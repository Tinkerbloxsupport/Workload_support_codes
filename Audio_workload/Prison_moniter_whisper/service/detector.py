import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set


@dataclass
class Detection:
    """A single threat or profanity detection."""
    cell_id: str
    timestamp: float
    category: str  # "threat" or "profanity"
    matched_word: str
    context: str
    severity: str  # "high", "medium", "low"


@dataclass
class AnalysisResult:
    """Analysis result for a transcription."""
    cell_id: str
    transcript: str
    detections: List[Detection] = field(default_factory=list)
    is_flagged: bool = False
    threat_count: int = 0
    profanity_count: int = 0


class ThreatDetector:
    """Detects threat keywords and profanity in transcripts."""
    
    # High-severity threat keywords
    HIGH_SEVERITY = {
        "escape", "escaping", "tunnel", "weapon", "knife", "shank",
        "kill", "riot", "attack", "break out", "breakout"
    }
    
    def __init__(self, threat_file: str = "config/threat_words.txt",
                 profanity_file: str = "config/profanity.txt"):
        """
        Initialize threat detector with keyword lists.
        
        Args:
            threat_file: Path to threat keywords file (one per line)
            profanity_file: Path to profanity keywords file (one per line)
        """
        self.threat_words = self._load_words(threat_file)
        self.profanity_words = self._load_words(profanity_file)
        self.threat_pattern = self._build_pattern(self.threat_words)
        self.profanity_pattern = self._build_pattern(self.profanity_words)

    def _load_words(self, path: str) -> Set[str]:
        """Load keywords from file."""
        p = Path(path)
        if not p.exists():
            return set()
        return {
            line.strip().lower()
            for line in p.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

    def _build_pattern(self, words: Set[str]) -> re.Pattern:
        """Build regex pattern for word matching."""
        if not words:
            # Pattern that never matches
            return re.compile(r"(?!)")
        
        # Sort by length (longest first) for better matching
        sorted_words = sorted(words, key=len, reverse=True)
        escaped = [re.escape(w) for w in sorted_words]
        return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    def analyze(self, cell_id: str, transcript: str,
                segments: List[dict] = None) -> AnalysisResult:
        """
        Analyze transcript for threats and profanity.
        
        Args:
            cell_id: Cell identifier
            transcript: Full transcript text
            segments: List of {start, end, text} dicts with timestamps
            
        Returns:
            AnalysisResult with detections and flags
        """
        result = AnalysisResult(cell_id=cell_id, transcript=transcript)
        
        if segments:
            for seg in segments:
                text = seg.get("text", "")
                timestamp = seg.get("start", 0.0)
                self._scan_text(result, text, timestamp)
        else:
            self._scan_text(result, transcript, 0.0)
        
        result.is_flagged = result.threat_count > 0 or result.profanity_count > 0
        return result

    def _scan_text(self, result: AnalysisResult, text: str, timestamp: float):
        """Scan text for threats and profanity."""
        # Scan for threats
        for match in self.threat_pattern.finditer(text):
            word = match.group(1).lower()
            severity = "high" if word in self.HIGH_SEVERITY else "medium"
            result.detections.append(Detection(
                cell_id=result.cell_id,
                timestamp=timestamp,
                category="threat",
                matched_word=word,
                context=self._get_context(text, match.start(), match.end()),
                severity=severity,
            ))
            result.threat_count += 1

        # Scan for profanity
        for match in self.profanity_pattern.finditer(text):
            word = match.group(1).lower()
            result.detections.append(Detection(
                cell_id=result.cell_id,
                timestamp=timestamp,
                category="profanity",
                matched_word=word,
                context=self._get_context(text, match.start(), match.end()),
                severity="low",
            ))
            result.profanity_count += 1

    def _get_context(self, text: str, start: int, end: int, window: int = 40) -> str:
        """Extract context around matched word."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        prefix = "..." if ctx_start > 0 else ""
        suffix = "..." if ctx_end < len(text) else ""
        return prefix + text[ctx_start:ctx_end].strip() + suffix
