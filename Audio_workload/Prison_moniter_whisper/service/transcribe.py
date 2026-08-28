import time
import json
import requests
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .detector import ThreatDetector, AnalysisResult

SAMPLE_RATE = 16000


@dataclass
class TranscriptionResult:
    cell_id: str
    text: str
    language: str
    decode_ms: float
    segments: list = field(default_factory=list)
    analysis: AnalysisResult = None
    native_timings: dict = field(default_factory=dict)


class WhisperService:
    def __init__(self, model_size: str = "base", device: str = "cpu",
                 whisper_host: str = "localhost", whisper_port: int = 8062):
        """
        Initialize Whisper service to connect to HTTP server.
        
        Args:
            model_size: Model size (for reference, not used in HTTP mode)
            device: Device type (for reference, not used in HTTP mode)
            whisper_host: Hostname of Whisper server (default: localhost)
            whisper_port: Port of Whisper server (default: 8062)
        """
        self.model_size = model_size
        self.device = device
        self.whisper_host = whisper_host
        self.whisper_port = whisper_port
        self.whisper_url = f"http://{whisper_host}:{whisper_port}/inference"
        self.detector = ThreatDetector()
        self._verify_connection()

    def _verify_connection(self):
        """Test connection to Whisper server."""
        try:
            # Try a simple health check by making a small request
            requests.head(f"http://{self.whisper_host}:{self.whisper_port}", timeout=5)
            print(f"✅ Connected to Whisper server at {self.whisper_url}")
        except requests.exceptions.ConnectionError:
            print(f"⚠️  Warning: Cannot connect to Whisper server at {self.whisper_url}")
            print(f"   Make sure the server is running with:")
            print(f"   /usr/bin/time -v sudo -E ./target/debug/ultraedge --root /var/lib/ultraedge run \\")
            print(f"     --image <ECR_IMAGE> \\")
            print(f"     --mount /var/lib/ultraedge/whisper-store/models:/models:rw \\")
            print(f"     --publish 8062:8062 \\")
            print(f"     -- /bin/sh -lc 'exec /app/build/bin/whisper-server -m /models/ggml-base.bin --host 0.0.0.0 --port 8062'")
        except Exception as e:
            print(f"⚠️  Warning: {e}")

    def transcribe(self, cell_id: str, audio: np.ndarray,
                   language: str = None, analyze: bool = True) -> TranscriptionResult:
        """
        Transcribe audio by sending to Whisper HTTP server.
        
        Args:
            cell_id: Cell identifier
            audio: Audio data as float32 numpy array
            language: Language code (optional)
            analyze: Whether to run threat detection
            
        Returns:
            TranscriptionResult with transcription and analysis
        """
        t0 = time.perf_counter()

        # Convert audio to bytes for sending
        audio_bytes = self._audio_to_bytes(audio)

        try:
            # Send to Whisper server
            files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
            params = {}
            if language:
                params['language'] = language

            response = requests.post(self.whisper_url, files=files, params=params, timeout=300)
            response.raise_for_status()

            data = response.json()
            text = data.get("text", "").strip()
            detected_language = language or "auto"

            # Parse segments if available in response
            seg_list = []
            if "segments" in data:
                seg_list = [
                    {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", "").strip(),
                    }
                    for seg in data.get("segments", [])
                ]
            else:
                # Fallback: single segment
                seg_list = [{"start": 0.0, "end": 0.0, "text": text}]

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Whisper server at {self.whisper_url}\n"
                f"Make sure the server is running on port {self.whisper_port}"
            ) from e
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Whisper server request timed out (300s)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Whisper server error: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from Whisper server: {e}") from e

        decode_ms = (time.perf_counter() - t0) * 1000

        transcription = TranscriptionResult(
            cell_id=cell_id,
            text=text,
            language=detected_language,
            decode_ms=decode_ms,
            segments=seg_list,
            native_timings={},  # Not available from HTTP server
        )

        if analyze:
            transcription.analysis = self.detector.analyze(cell_id, text, seg_list)

        return transcription

    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float32 audio to WAV bytes."""
        import wave
        import io

        # Ensure audio is in [-1, 1] range
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767).astype(np.int16)

        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())

        wav_buffer.seek(0)
        return wav_buffer.read()
