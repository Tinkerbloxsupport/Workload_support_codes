import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Dict, Optional, Tuple, TextIO

from service.audio import load_audio
from service.transcribe import WhisperService

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".mkv", ".mov"}

# Default configuration
DEFAULT_KEYWORDS_CONFIG = {
    "threat_keywords": {
        "high": ["tunnel", "shank", "weapon", "knife", "kill", "attack", "breakout"],
        "medium": ["escape", "run", "flee", "digging", "dig", "unlock", "gate", "fence", "break out", "getaway"],
        "low": ["plan", "tonight", "tomorrow", "signal", "outside", "freedom"]
    },
    "profanity_keywords": {
        "high": [],
        "medium": ["fuck", "shit"],
        "low": ["damn", "bastard", "bitch", "asshole", "crap", "dick", "piss", "hell"]
    },
    "contextual_innocence": {
        "outside": ["tea", "morning", "walk", "day", "spend", "air", "nature", "fresh", "sit"],
        "plan": ["weekend", "schedule", "dinner", "meeting", "trip", "work", "project"],
        "guard": ["security", "watch", "protect", "duty", "shift", "stand"],
        "tomorrow": ["meeting", "work", "appointment", "call", "visit", "day", "weather"],
        "gate": ["open", "lock", "security", "entrance", "exit"],
        "run": ["exercise", "jog", "errand", "coffee", "shower", "late"],
        "wall": ["paint", "repair", "picture", "hang", "decor"]
    }
}


@dataclass
class FileTimingStats:
    """Timing statistics for a single file."""
    cell_id: str
    file_name: str
    transcribe_ms: float  # Whisper transcription time
    filter_ms: float      # Filtering + phrase detection time
    total_ms: float       # Total time


class KeywordFilter:
    """Filters detections based on severity, context, and phrase matching."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_KEYWORDS_CONFIG
        self.threat_keywords = self.config.get("threat_keywords", {})
        self.profanity_keywords = self.config.get("profanity_keywords", {})
        self.contextual_innocence = self.config.get("contextual_innocence", {})
    
    def should_filter_detection(self, detection, transcript_text: str, min_severity: str = "medium") -> bool:
        """
        Determine if a detection should be filtered out (ignored).
        
        Returns True if should be filtered (ignored), False if should be kept.
        """
        word = detection.matched_word.lower()
        
        # Filter by severity threshold
        severity_order = {"low": 0, "medium": 1, "high": 2}
        if severity_order.get(detection.severity, 0) < severity_order.get(min_severity, 1):
            return True
        
        # Filter low-severity profanity in casual speech
        if detection.category == "profanity" and detection.severity == "low":
            return True
        
        # Context-aware filtering for commonly false-positive words
        if word in self.contextual_innocence:
            innocent_contexts = self.contextual_innocence[word]
            transcript_lower = transcript_text.lower()
            
            # If surrounded by innocent context words, filter it out
            if any(ctx in transcript_lower for ctx in innocent_contexts):
                return True
        
        # Never filter high-severity threats
        if detection.severity == "high":
            return False
        
        return False
    
    def extract_threat_phrases(self, transcript_text: str) -> List[Dict]:
        """
        Extract multi-word threat phrases from transcript.
        More accurate than single-word matching.
        """
        threat_phrases = [
            "escape tonight",
            "escape tomorrow",
            "break out",
            "digging tunnel",
            "tunnel behind",
            "need to escape",
            "running away",
            "get out tonight",
            "get out tomorrow",
            "steal weapon",
            "steal knife",
            "make shank",
        ]
        
        detections = []
        transcript_lower = transcript_text.lower()
        
        for phrase in threat_phrases:
            idx = transcript_lower.find(phrase)
            if idx != -1:
                # Extract context (50 chars before and after)
                start = max(0, idx - 50)
                end = min(len(transcript_text), idx + len(phrase) + 50)
                context = transcript_text[start:end].strip()
                
                detections.append({
                    "category": "threat",
                    "matched_phrase": phrase,
                    "context": context,
                    "severity": "high",
                    "type": "phrase"
                })
        
        return detections


def get_extended_context(full_text: str, matched_word: str, context_snippet: str = "", context_width: int = 100) -> str:
    """Extract more context around a matched word than what's in the detection."""
    if not full_text or not matched_word:
        return context_snippet
    
    idx = full_text.lower().find(matched_word.lower())
    if idx == -1:
        return context_snippet
    
    start = max(0, idx - context_width)
    end = min(len(full_text), idx + len(matched_word) + context_width)
    
    return full_text[start:end].strip()


def format_time(milliseconds: float) -> str:
    """Format milliseconds to human-readable time."""
    if milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    else:
        return f"{milliseconds/1000:.2f}s"


def log_message(message: str, cycle_log: Optional[TextIO] = None, summary_log: Optional[TextIO] = None, to_console: bool = True):
    """Write message to console, cycle log, and/or summary log."""
    if to_console:
        print(message)
    if cycle_log:
        cycle_log.write(message + "\n")
        cycle_log.flush()
    if summary_log:
        summary_log.write(message + "\n")
        summary_log.flush()


def cmd_monitor(args):
    """Monitor a single audio file and report detections."""
    try:
        svc = WhisperService(
            model_size=args.model,
            device=args.device,
            whisper_host=args.whisper_host,
            whisper_port=args.whisper_port
        )
    except Exception as e:
        sys.exit(f"Failed to initialize WhisperService: {e}")
    
    try:
        audio = load_audio(args.file)
    except Exception as e:
        sys.exit(f"Failed to load audio file: {e}")
    
    try:
        result = svc.transcribe(args.cell, audio, language=args.language)
    except Exception as e:
        sys.exit(f"Transcription failed: {e}")
    
    # Load keyword filter
    keywords_config = DEFAULT_KEYWORDS_CONFIG
    if args.keywords_file:
        try:
            with open(args.keywords_file) as f:
                keywords_config = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load keywords file: {e}")
    
    keyword_filter = KeywordFilter(keywords_config)
    
    analysis = result.analysis
    
    print(f"\n{'='*60}")
    print(f"CELL {args.cell} — Audio Monitoring Report")
    print(f"{'='*60}\n")
    
    if analysis.is_flagged:
        if analysis.threat_count > 0:
            print("🚨 STATUS: FLAGGED — THREAT KEYWORDS DETECTED\n")
        else:
            print("⚠️  STATUS: FLAGGED — PROFANITY DETECTED\n")
    else:
        print("✅ STATUS: CLEAN — No threats detected\n")
    
    print("--- Transcript ---")
    print(result.text)
    print()
    
    # Filter detections
    filtered_detections = [
        d for d in analysis.detections
        if not keyword_filter.should_filter_detection(d, result.text, args.min_severity)
    ]
    
    # Extract phrase-level detections
    phrase_detections = keyword_filter.extract_threat_phrases(result.text)
    
    if filtered_detections or phrase_detections:
        print("--- Detections ---")
        
        # Show phrase detections first (more reliable)
        for d in phrase_detections:
            print(f"  🔴 [PHRASE] {d['matched_phrase'].upper()}")
            print(f"      Context: \"{d['context']}\"")
        
        # Show filtered keyword detections
        for d in filtered_detections:
            icon = "🔴" if d.severity == "high" else "🟡" if d.severity == "medium" else "🟢"
            extended_context = get_extended_context(result.text, d.matched_word, d.context)
            print(f"  {icon} [{d.timestamp:5.1f}s] {d.category.upper()}: '{d.matched_word}'")
            print(f"      Context: \"{extended_context}\"")
        print()
    
    print("--- Stats ---")
    print(f"  Threats (before filter):   {analysis.threat_count}")
    print(f"  Threats (after filter):    {len([d for d in filtered_detections if d.category == 'threat'])}")
    print(f"  Profanity (before filter): {analysis.profanity_count}")
    print(f"  Profanity (after filter):  {len([d for d in filtered_detections if d.category == 'profanity'])}")
    print(f"  Phrase detections:         {len(phrase_detections)}")
    print(f"  Decode:                    {result.decode_ms:.0f}ms")


def cmd_sweep(args):
    """Batch process audio files in a directory (with cycle support)."""
    files = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in AUDIO_EXTS)[:args.limit]
    if not files:
        sys.exit(f"No audio files in {args.dir}")
    
    cycles = args.cycles if hasattr(args, 'cycles') and args.cycles else 1
    
    # Create log files
    cycle_log_path = Path("cycle_count_logs.txt")
    summary_log_path = Path("Final summary logs.txt")
    
    # Open log files
    cycle_log = open(cycle_log_path, 'w')
    summary_log = open(summary_log_path, 'w')
    
    header = f"\n[sweep] {len(files)} cells × {cycles} cycle(s) = {len(files) * cycles} total | host={args.whisper_host}:{args.whisper_port} model={args.model}\n"
    log_message(header, cycle_log=cycle_log, to_console=True)
    
    try:
        svc = WhisperService(
            model_size=args.model,
            device=args.device,
            whisper_host=args.whisper_host,
            whisper_port=args.whisper_port
        )
    except Exception as e:
        sys.exit(f"Failed to initialize WhisperService: {e}")
    
    # Load keyword filter
    keywords_config = DEFAULT_KEYWORDS_CONFIG
    if args.keywords_file:
        try:
            with open(args.keywords_file) as f:
                keywords_config = json.load(f)
                msg = f"[keywords] Loaded from {args.keywords_file}\n"
                log_message(msg, cycle_log=cycle_log, to_console=True)
        except Exception as e:
            msg = f"⚠️  Warning: Could not load keywords file: {e}\n"
            log_message(msg, cycle_log=cycle_log, to_console=True)
    
    keyword_filter = KeywordFilter(keywords_config)
    
    # Cycle-based processing
    all_results = []
    cycle_stats = []
    file_timing_stats = {}  # Track timing per file across cycles
    
    total_start = time.perf_counter()
    
    for cycle_num in range(1, cycles + 1):
        cycle_start = time.perf_counter()
        cycle_results = []
        flagged_cells = []
        failed_cells = []
        
        cycle_header = f"\n--- CYCLE {cycle_num}/{cycles} ---\n"
        log_message(cycle_header, cycle_log=cycle_log, to_console=True)
        
        for i, p in enumerate(files, 1):
            cell_id = p.stem.upper().replace("_", "-")
            
            try:
                file_start = time.perf_counter()
                
                audio = load_audio(str(p))
                r = svc.transcribe(cell_id, audio, language=args.language)
                
                transcribe_ms = r.decode_ms
                
                # Filter detections
                filter_start = time.perf_counter()
                filtered_detections = [
                    d for d in r.analysis.detections
                    if not keyword_filter.should_filter_detection(d, r.text, args.min_severity)
                ]
                
                # Extract phrase detections
                phrase_detections = keyword_filter.extract_threat_phrases(r.text)
                filter_ms = (time.perf_counter() - filter_start) * 1000
                
                # Determine if truly flagged after filtering
                threat_count_filtered = len([d for d in filtered_detections if d.category == "threat"])
                profanity_count_filtered = len([d for d in filtered_detections if d.category == "profanity"])
                is_flagged_filtered = (threat_count_filtered > 0) or (profanity_count_filtered > 0) or len(phrase_detections) > 0
                
                total_file_ms = (time.perf_counter() - file_start) * 1000
                
                row = {
                    "cell_id": cell_id,
                    "file": p.name,
                    "cycle": cycle_num,
                    "decode_ms": r.decode_ms,
                    "filter_ms": filter_ms,
                    "total_ms": total_file_ms,
                    "is_flagged": is_flagged_filtered,
                    "threat_count": threat_count_filtered,
                    "profanity_count": profanity_count_filtered,
                    "threat_count_raw": r.analysis.threat_count,
                    "profanity_count_raw": r.analysis.profanity_count,
                    "phrase_detections": phrase_detections,
                    "detections": [asdict(d) for d in filtered_detections],
                }
                cycle_results.append(row)
                
                # Track timing per file
                if cell_id not in file_timing_stats:
                    file_timing_stats[cell_id] = []
                file_timing_stats[cell_id].append(total_file_ms)
                
                if is_flagged_filtered:
                    flagged_cells.append(cell_id)
                    if threat_count_filtered > 0 or len(phrase_detections) > 0:
                        icon = "🚨"
                    else:
                        icon = "⚠️"
                    msg = f"  {icon} [{i:3d}/{len(files)}] {cell_id}: FLAGGED (threats={threat_count_filtered}, profanity={profanity_count_filtered}, phrases={len(phrase_detections)}) [{format_time(total_file_ms)}]"
                    log_message(msg, cycle_log=cycle_log, to_console=True)
                elif i % 10 == 0 or i == len(files):
                    msg = f"  ✅ [{i:3d}/{len(files)}] {cell_id}: clean [{format_time(total_file_ms)}]"
                    log_message(msg, cycle_log=cycle_log, to_console=True)
            
            except Exception as e:
                failed_cells.append((cell_id, str(e)))
                msg = f"  ❌ [{i:3d}/{len(files)}] {cell_id}: ERROR - {e}"
                log_message(msg, cycle_log=cycle_log, to_console=True)
        
        cycle_duration = time.perf_counter() - cycle_start
        
        # Cycle summary
        cycle_flagged = len(flagged_cells)
        cycle_processed = len(cycle_results)
        cycle_summary = {
            "cycle": cycle_num,
            "duration_seconds": cycle_duration,
            "cells_processed": cycle_processed,
            "cells_flagged": cycle_flagged,
            "cells_failed": len(failed_cells),
            "throughput": cycle_processed / cycle_duration if cycle_duration > 0 else 0,
            "avg_ms_per_file": (cycle_duration * 1000) / cycle_processed if cycle_processed > 0 else 0,
            "flagged_cells": flagged_cells,
        }
        cycle_stats.append(cycle_summary)
        
        msg1 = f"\n  ⏱️  Cycle {cycle_num} completed in {format_time(cycle_duration * 1000)}"
        msg2 = f"  Flagged: {cycle_flagged}/{cycle_processed} | Avg/file: {format_time(cycle_summary['avg_ms_per_file'])}"
        log_message(msg1, cycle_log=cycle_log, to_console=True)
        log_message(msg2, cycle_log=cycle_log, to_console=True)
        
        all_results.extend(cycle_results)
    
    total_duration = time.perf_counter() - total_start
    
    # Overall statistics
    separator = f"\n{'='*60}"
    msg1 = f"SWEEP COMPLETE — {len(all_results)}/{len(files) * cycles} files processed in {format_time(total_duration * 1000)}"
    msg2 = f"{'='*60}\n"
    
    log_message(separator, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg1, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg2, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    msg3 = f"  Total cycles: {cycles}"
    msg4 = f"  Files per cycle: {len(files)}"
    msg5 = f"  Total files processed: {len(all_results)}"
    msg6 = f"  Overall throughput: {len(all_results) / total_duration:.2f} files/sec"
    
    log_message(msg3, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg4, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg5, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg6, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    # Per-file timing statistics
    msg_timing = f"\n--- Per-File Timing (Average across {cycles} cycle(s)) ---"
    log_message(msg_timing, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    avg_file_times = {}
    min_file_times = {}
    max_file_times = {}
    
    for cell_id, times in file_timing_stats.items():
        avg_file_times[cell_id] = sum(times) / len(times)
        min_file_times[cell_id] = min(times)
        max_file_times[cell_id] = max(times)
    
    # Sort by average time (slowest first)
    sorted_files = sorted(avg_file_times.items(), key=lambda x: x[1], reverse=True)
    
    msg_slowest = f"\n  Slowest files (top 5):"
    log_message(msg_slowest, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    for cell_id, avg_time in sorted_files[:5]:
        min_t = min_file_times[cell_id]
        max_t = max_file_times[cell_id]
        msg = f"    {cell_id}: avg={format_time(avg_time)}, min={format_time(min_t)}, max={format_time(max_t)}"
        log_message(msg, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    msg_fastest = f"\n  Fastest files (bottom 5):"
    log_message(msg_fastest, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    for cell_id, avg_time in sorted_files[-5:]:
        min_t = min_file_times[cell_id]
        max_t = max_file_times[cell_id]
        msg = f"    {cell_id}: avg={format_time(avg_time)}, min={format_time(min_t)}, max={format_time(max_t)}"
        log_message(msg, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    # Cycle timing summary
    if cycles > 1:
        msg_cycle_summary = f"\n--- Cycle-by-Cycle Summary ---"
        log_message(msg_cycle_summary, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
        for cs in cycle_stats:
            msg = f"  Cycle {cs['cycle']}: {format_time(cs['duration_seconds'] * 1000)} ({cs['throughput']:.2f} files/sec, avg/file: {format_time(cs['avg_ms_per_file'])})"
            log_message(msg, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    # Detection statistics across all cycles
    total_threat_raw = sum(r["threat_count_raw"] for r in all_results)
    total_threat_filtered = sum(r["threat_count"] for r in all_results)
    total_profanity_raw = sum(r["profanity_count_raw"] for r in all_results)
    total_profanity_filtered = sum(r["profanity_count"] for r in all_results)
    
    msg_detection = f"\n--- Detection Stats (All Cycles) ---"
    msg_threat_raw = f"  Threats (raw):          {total_threat_raw}"
    msg_threat_filtered = f"  Threats (filtered):     {total_threat_filtered}"
    msg_false_pos = f"  False positives removed: {total_threat_raw - total_threat_filtered}"
    msg_profanity_raw = f"  Profanity (raw):        {total_profanity_raw}"
    msg_profanity_filtered = f"  Profanity (filtered):   {total_profanity_filtered}"
    
    log_message(msg_detection, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg_threat_raw, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg_threat_filtered, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg_false_pos, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg_profanity_raw, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    log_message(msg_profanity_filtered, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    # Flagged cells (unique across all cycles)
    unique_flagged = set()
    for r in all_results:
        if r["is_flagged"]:
            unique_flagged.add(r["cell_id"])
    
    msg_unique = f"\n  Unique flagged cells: {len(unique_flagged)}/{len(files)}"
    log_message(msg_unique, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    if unique_flagged:
        msg_review = f"  ⚠️  Cells requiring review: {', '.join(sorted(unique_flagged))}"
        log_message(msg_review, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    # Save JSON output
    if args.json_out:
        out = {
            "host": args.whisper_host,
            "port": args.whisper_port,
            "model": args.model,
            "min_severity": args.min_severity,
            "cycles": cycles,
            "n_files_per_cycle": len(files),
            "n_total_files": len(all_results),
            "n_cells_failed": sum(cs["cells_failed"] for cs in cycle_stats),
            "total_seconds": total_duration,
            "overall_throughput": len(all_results) / total_duration if total_duration > 0 else 0,
            "per_file_timing": {
                cell_id: {
                    "avg_ms": avg_file_times[cell_id],
                    "min_ms": min_file_times[cell_id],
                    "max_ms": max_file_times[cell_id],
                    "samples": len(file_timing_stats[cell_id])
                }
                for cell_id in sorted(file_timing_stats.keys())
            },
            "cycle_stats": cycle_stats,
            "summary": {
                "threat_count_raw": total_threat_raw,
                "threat_count_filtered": total_threat_filtered,
                "false_positives_removed": total_threat_raw - total_threat_filtered,
                "profanity_count_raw": total_profanity_raw,
                "profanity_count_filtered": total_profanity_filtered,
                "unique_flagged_cells": sorted(list(unique_flagged)),
            },
            "results": all_results
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        msg_json = f"\n  [saved] {args.json_out}"
        log_message(msg_json, cycle_log=cycle_log, summary_log=summary_log, to_console=True)
    
    # Save log file messages
    msg_cycle_saved = f"\n  [saved] {cycle_log_path}"
    msg_summary_saved = f"  [saved] {summary_log_path}"
    log_message(msg_cycle_saved, cycle_log=cycle_log, to_console=True)
    log_message(msg_summary_saved, summary_log=summary_log, to_console=True)
    
    # Close log files
    cycle_log.close()
    summary_log.close()
    
    print(f"\n✅ All logs saved successfully!")


def main():
    p = argparse.ArgumentParser(
        prog="prison-monitor",
        description="Audio monitoring system for threat and profanity detection with cycle support and timing analysis"
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="Device type (for reference only in HTTP mode)")
    p.add_argument("--model", default="base",
                   help="Whisper model (for reference only in HTTP mode)")
    p.add_argument("--whisper-host", default="localhost",
                   help="Whisper server hostname (default: localhost)")
    p.add_argument("--whisper-port", type=int, default=8062,
                   help="Whisper server port (default: 8062)")
    p.add_argument("--language", default=None,
                   help="Language code (e.g., en, ta, hi)")
    p.add_argument("--min-severity", choices=["low", "medium", "high"], default="medium",
                   help="Minimum severity level to report (default: medium)")
    p.add_argument("--keywords-file", type=str, default=None,
                   help="Path to JSON file with threat/profanity keywords configuration")
    
    sub = p.add_subparsers(dest="command", required=True)
    
    m = sub.add_parser("monitor", help="Monitor one cell")
    m.add_argument("cell", help="Cell ID")
    m.add_argument("file", help="Audio file path")
    m.set_defaults(func=cmd_monitor)
    
    s = sub.add_parser("sweep", help="Batch process directory with cycle support")
    s.add_argument("dir", help="Directory of audio files")
    s.add_argument("--limit", type=int, default=500,
                   help="Maximum number of files to process per cycle (default: 500)")
    s.add_argument("--cycles", type=int, default=1,
                   help="Number of times to process the directory (default: 1). Example: --cycles 10 runs all files 10 times")
    s.add_argument("--json-out", help="Save results to JSON file")
    s.set_defaults(func=cmd_sweep)
    
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
