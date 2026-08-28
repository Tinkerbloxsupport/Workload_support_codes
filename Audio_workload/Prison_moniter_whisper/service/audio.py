import ffmpeg
import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """
    Load audio file using ffmpeg and convert to float32 PCM.
    
    Args:
        path: Path to audio file (mp3, wav, m4a, flac, ogg, aac, mp4, mkv, mov)
        
    Returns:
        Audio as float32 numpy array, mono, 16kHz, values in [-1, 1]
        
    Raises:
        RuntimeError: If ffmpeg decoding fails
    """
    try:
        out, _ = (
            ffmpeg.input(path)
            .output("-", format="f32le", ac=1, ar=SAMPLE_RATE)
            .run(capture_stdout=True, quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {e.stderr.decode(errors='ignore')}")
    
    return np.frombuffer(out, dtype=np.float32)
