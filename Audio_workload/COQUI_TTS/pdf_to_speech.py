"""
PDF -> Speech client, talks to a running Coqui tts-server over HTTP.

Prereq: a tts-server container/process already running and reachable
(see Dockerfile.server). This script does NOT load any model itself -
it just extracts text from the PDF, chunks it, and POSTs each chunk to
the server's /api/tts endpoint, then concatenates the returned WAV bytes.

Usage:
    python pdf_to_speech.py input.pdf output.wav
    python pdf_to_speech.py input.pdf output.wav --tts-host localhost --tts-port 5002
    python pdf_to_speech.py input.pdf output.wav --tts-host tts_server --tts-port 5002 --speaker_id p225

Install (once, client side - much lighter than the server, no torch needed):
    pip install requests pypdf --break-system-packages
"""

import argparse
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path

import requests
from pypdf import PdfReader


# Tacotron2's character set is built for plain ASCII. PDFs are full of
# "smart" typographic characters that will 500 the server if sent as-is.
_CHAR_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'",   # single curly quotes
    "\u201c": '"', "\u201d": '"',   # double curly quotes
    "\u2013": "-", "\u2014": "-",   # en dash, em dash
    "\u2026": "...",                # ellipsis
    "\u00a0": " ",                   # non-breaking space
}


def sanitize_text(text: str) -> str:
    for bad, good in _CHAR_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Fold any remaining non-ASCII (accented letters etc.) to closest ASCII equivalent;
    # drop anything that can't be represented at all.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    full_text = "\n".join(pages_text)

    full_text = re.sub(r"-\n", "", full_text)
    full_text = re.sub(r"\s*\n\s*", " ", full_text)
    full_text = re.sub(r"\s{2,}", " ", full_text).strip()
    full_text = sanitize_text(full_text)
    return full_text


def split_into_chunks(text: str, max_chars: int = 400):
    """Split text into one-sentence chunks (hard-capped at max_chars for very long sentences).

    Packing multiple sentences into one chunk causes Tacotron2's attention to drift
    partway through and start generating garbled audio - one sentence per chunk is
    far more reliable, even though it means more (shorter) requests overall.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Hard-split an unusually long single sentence.
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars])
    return chunks


class TTSClient:
    """Thin HTTP client for Coqui's tts-server /api/tts endpoint."""

    def __init__(self, host: str, port: int, timeout: int = 120):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def check_alive(self):
        """Confirm the server is up before burning time on chunking/synthesis."""
        try:
            r = requests.get(f"{self.base_url}/", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach tts-server at {self.base_url} ({e}). "
                f"Is the server container running and the port reachable?"
            )

    def synthesize(self, text: str, out_path: str, speaker_id: str = None, style_wav: str = None):
        params = {"text": text}
        if speaker_id:
            params["speaker_id"] = speaker_id
        if style_wav:
            params["style_wav"] = style_wav

        resp = requests.get(f"{self.base_url}/api/tts", params=params, timeout=self.timeout)
        resp.raise_for_status()
        Path(out_path).write_bytes(resp.content)


def concatenate_wavs(wav_paths, output_path):
    if not wav_paths:
        raise ValueError("No audio chunks were generated.")

    with wave.open(wav_paths[0], "rb") as first:
        params = first.getparams()

    with wave.open(output_path, "wb") as out_wav:
        out_wav.setparams(params)
        for wav_path in wav_paths:
            with wave.open(wav_path, "rb") as w:
                out_wav.writeframes(w.readframes(w.getnframes()))


def main():
    parser = argparse.ArgumentParser(description="Convert a PDF to speech via a running Coqui tts-server.")
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument("output_path", help="Path to the output WAV file.")
    parser.add_argument("--tts-host", default="localhost",
                         help="tts-server hostname (default: localhost). Use the container name if on the same Docker network.")
    parser.add_argument("--tts-port", type=int, default=5002,
                         help="tts-server port (default: 5002).")
    parser.add_argument("--speaker_id", default=None,
                         help="Speaker ID, for multi-speaker models served by tts-server.")
    parser.add_argument("--style_wav", default=None,
                         help="Path/URL to a style reference WAV, if the served model supports it.")
    parser.add_argument("--max_chars", type=int, default=400,
                         help="Max characters per synthesis chunk (default: 400).")
    parser.add_argument("--time-limit-minutes", type=float, default=None,
                         help="Stop synthesizing after this many minutes (for sizing benchmark runs). "
                              "Whatever chunks completed so far are still concatenated into the output.")
    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        sys.exit(f"Input PDF not found: {args.pdf_path}")

    client = TTSClient(args.tts_host, args.tts_port)
    print(f"Checking tts-server at {args.tts_host}:{args.tts_port} ...")
    client.check_alive()
    print("Server is reachable.")

    print(f"Extracting text from: {args.pdf_path}")
    text = extract_text_from_pdf(args.pdf_path)
    if not text.strip():
        sys.exit("No extractable text found in the PDF (it may be scanned/image-based; OCR it first).")
    print(f"Extracted {len(text)} characters.")

    chunks = split_into_chunks(text, max_chars=args.max_chars)
    print(f"Split into {len(chunks)} chunk(s) for synthesis.")

    tmp_dir = Path(args.output_path).parent / "._pdf_tts_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []

    try:
        synth_start = time.perf_counter()
        failed_chunks = []
        time_limit_sec = args.time_limit_minutes * 60 if args.time_limit_minutes else None
        stopped_early = False

        for idx, chunk in enumerate(chunks):
            if time_limit_sec is not None and (time.perf_counter() - synth_start) >= time_limit_sec:
                print(f"\n[time limit] Stopping after {idx}/{len(chunks)} chunks "
                      f"({args.time_limit_minutes:.1f} min elapsed).")
                stopped_early = True
                break

            chunk_wav = str(tmp_dir / f"chunk_{idx:05d}.wav")
            print(f"Synthesizing chunk {idx + 1}/{len(chunks)} ({len(chunk)} chars)...")
            try:
                client.synthesize(chunk, chunk_wav, speaker_id=args.speaker_id, style_wav=args.style_wav)
                chunk_paths.append(chunk_wav)
            except Exception as e:
                print(f"  WARNING: chunk {idx + 1} failed, skipping ({e})")
                failed_chunks.append((idx + 1, str(e)))
                continue

            if (idx + 1) % 50 == 0:
                elapsed = time.perf_counter() - synth_start
                rate = elapsed / (idx + 1)
                remaining = rate * (len(chunks) - (idx + 1))
                print(f"  [progress] {idx + 1}/{len(chunks)} done, "
                      f"{rate:.2f}s/chunk avg, ETA {remaining/60:.1f} min remaining "
                      f"(full document; ignores --time-limit-minutes)")

        synth_ms = (time.perf_counter() - synth_start) * 1000
        completed = len(chunk_paths) + len(failed_chunks)
        avg_ms = synth_ms / completed if completed else 0
        print(f"Synthesis stopped after {synth_ms/1000:.1f}s "
              f"({completed} chunk(s) attempted, {avg_ms:.0f}ms/chunk avg).")
        if stopped_early:
            print(f"[time limit] {len(chunks) - completed} chunk(s) not processed "
                  f"(would take an estimated {(len(chunks) - completed) * avg_ms / 1000 / 60:.1f} more min at this rate).")
        if failed_chunks:
            print(f"\n{len(failed_chunks)} chunk(s) failed and were skipped:")
            for chunk_num, err in failed_chunks[:20]:
                print(f"  chunk {chunk_num}: {err}")
            if len(failed_chunks) > 20:
                print(f"  ... and {len(failed_chunks) - 20} more")

        print(f"Concatenating {len(chunk_paths)} chunk(s) into: {args.output_path}")
        concatenate_wavs(chunk_paths, args.output_path)
        print("Done.")
    finally:
        for p in chunk_paths:
            Path(p).unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
