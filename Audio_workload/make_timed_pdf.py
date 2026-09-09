"""
Carve out a slice of a PDF sized to run for a target duration through the
Coqui TTS pipeline, based on a measured ms/chunk rate.

Usage:
    python make_timed_pdf.py input.pdf output_1hr.pdf --target-minutes 60 --ms-per-chunk 429
"""

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


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
    return full_text


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="Slice a PDF to a target TTS run duration.")
    parser.add_argument("input_pdf")
    parser.add_argument("output_pdf")
    parser.add_argument("--target-minutes", type=float, default=60)
    parser.add_argument("--ms-per-chunk", type=float, default=429,
                         help="Measured avg ms/chunk from a prior test run.")
    args = parser.parse_args()

    if not Path(args.input_pdf).exists():
        sys.exit(f"Input PDF not found: {args.input_pdf}")

    target_chunks = int((args.target_minutes * 60 * 1000) / args.ms_per_chunk)
    print(f"Target: {args.target_minutes} min at {args.ms_per_chunk}ms/chunk => ~{target_chunks} sentences needed")

    text = extract_text_from_pdf(args.input_pdf)
    sentences = split_sentences(text)
    print(f"Source PDF has {len(sentences)} total sentences.")

    if target_chunks >= len(sentences):
        print("Target exceeds full document length - using the whole document instead.")
        selected = sentences
    else:
        selected = sentences[:target_chunks]

    print(f"Writing {len(selected)} sentences to {args.output_pdf} "
          f"(~{len(selected) * args.ms_per_chunk / 1000 / 60:.1f} min estimated run time)")

    c = canvas.Canvas(args.output_pdf, pagesize=letter)
    width, height = letter
    margin = 60
    y = height - margin
    line_height = 14
    max_chars_per_line = 95

    for sentence in selected:
        # naive word-wrap
        words = sentence.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= max_chars_per_line:
                line = f"{line} {word}".strip()
            else:
                c.drawString(margin, y, line)
                y -= line_height
                if y < margin:
                    c.showPage()
                    y = height - margin
                line = word
        if line:
            c.drawString(margin, y, line)
            y -= line_height
            if y < margin:
                c.showPage()
                y = height - margin

    c.save()
    print("Done.")


if __name__ == "__main__":
    main()
