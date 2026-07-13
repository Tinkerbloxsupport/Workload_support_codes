"""
Core Agentic AI Code Generation Pipeline — Multi-Model (Hybrid) v11
====================================================================

What changed in v11 (functional integrity + real tool execution):

11a. FUNCTIONAL-FULFILLMENT CHECK (quality reviewer)
    The quality reviewer now verifies the code ACTUALLY implements the task —
    if the task names a library/protocol (asyncpg, COPY, Pydantic strict), the
    real code must use it, not fake it in comments/placeholders. Stubbed core
    functionality is an automatic rejection.
11b. ANTI-STUB GUARD (validator + deterministic routing)
    A high-precision phrase detector flags "in a real implementation we
    would…"-style placeholders in the source; the reviewer routes these straight
    to the developer with "implement it for real" feedback. Closes the escape
    where the dev gutted required logic to make tests pass.
11c. SOURCE-SCOPED COVERAGE GATE (tester)
    Coverage now targets the source modules (--cov=<source>) instead of '.', so
    a ~100%-covered test file can no longer inflate the total and carry an
    under-covered source past the 80% gate.
11d. ASYNC-MOCK ROUTING (reviewer)
    "MagicMock can't be used in 'await'" / "coroutine was never awaited" is a
    test-mocking defect → routed deterministically to QA (use AsyncMock), not
    the developer. This was what triggered the source-gutting failure mode.
11e. REAL TOOL EXECUTION (run_react_loop)
    Text-format tool calls (web_search, fetch_url, …) the runtime didn't parse
    are now recovered AND executed in-loop generically, so research actually
    runs instead of leaking markup. Deduped to avoid loops.
11f. BATCH RESILIENCE ON UNSTABLE NETWORKS
    An unstable connection could hang an Ollama call forever; interrupting the
    hang raised KeyboardInterrupt (a BaseException), which slipped past the
    loop's `except Exception` and killed the WHOLE batch — so prompts after a
    broken one never ran. Fixes: (a) LLM_TIMEOUT_S sets a per-request timeout so
    a stalled/broken connection raises a normal catchable error instead of
    hanging; (b) the batch loop now contains any per-prompt failure (Exception
    or BaseException) and moves on, while a deliberate Ctrl+C stops the batch
    gracefully and still prints the summary.

Earlier (v10) changes retained below.

Core Agentic AI Code Generation Pipeline — Multi-Model (Hybrid) v10
====================================================================

What changed vs v9 (critical correctness fixes):

0. SOURCE-FILE CLOBBER BUG FIXED (the big one)
   In v9 the QA node could call `write_file` on the PRIMARY SOURCE file
   instead of a `test_` file. A weak local model would happily overwrite
   a correct `main.py` with a trivial stub (e.g. `return "Hello, World!"`),
   which then "passed" all downstream gates (validator, tester at 100%
   coverage of 2 statements, pragmatic quality review) and got saved as
   the final artifact. Two defenses now prevent this:
     a) A QA WRITE GUARD: while the QA node is active, `write_file`
        REJECTS any filename that is not `test_`-prefixed, so source
        files are frozen and cannot be overwritten by the test author.
     b) A PLACEHOLDER GUARD in the validator: if the primary source is a
        detectable stub/placeholder (and the task isn't itself trivial),
        validation FAILS and routes back to the developer instead of
        sailing through to approval.

0b. STRUCTURED-OUTPUT REVIEWERS HARDENED
   Local models often fail `with_structured_output()`. Previously the
   reviewer then emitted a useless "parsing failed" message (losing the
   real error) and, worse, the quality reviewer DEFAULTED TO APPROVE on
   any parse glitch — a silent rubber stamp. Both now fall back to a
   plain-text JSON salvage (`_salvage_verdict`); the reviewer hands the
   developer the actual pipeline error when it still can't parse, and the
   quality gate only approves-to-avoid-loops as a last resort, loudly.

0c. REVIEWER MODEL -> qwen3:14b (temp 0)
   Swapped the reviewer from gemma4 to qwen3:14b for stronger JSON/structured
   adherence, using Ollama's constrained decoding (`method="json_schema"`).
   qwen3 is a reasoning model, so plain-text reviewer calls (filename pick,
   language detection, supervisor routing) now go through `_reviewer_text`,
   which appends '/no_think' and strips any leaked <think> block so reasoning
   tokens can never poison parsing.

0d. DETERMINISTIC SYNTAX-ERROR ROUTING + REPORTING POLISH
   - A SYNTAX ERROR in a source file now routes straight to the developer
     (and a test-file syntax error to QA) WITHOUT asking the reviewer LLM.
     Previously the LLM mis-routed a main.py syntax error to QA, which can't
     edit frozen source, deadlocking for several repair cycles.
   - The developer's inline-code path is no longer logged as a scary
     [WARNING]; code models emitting a fenced block is normal. The extractor
     now picks the longest PARSEABLE block for the primary file (fewer
     spurious syntax errors) and honours the QA write-guard so the fallback
     can't clobber source either.
   - The code review report gained a Coverage Analysis section: when coverage
     is below 100% it lists exactly which lines are untested (with source);
     at 100% it says so in one line. Tester now runs with term-missing.

0e. SOURCE-FOCUSED COVERAGE  (parse-guard re-prompt superseded by 0f)
   - Coverage Analysis now details only SOURCE-file gaps; gaps that live solely
     in test files (their own pytest.main()/__main__ guard) are noted as
     immaterial, or — when source is fully covered — summarised as "no action
     needed" rather than flagged as a problem.

0f. ROOT CAUSE FOUND: TEXT-FORMAT TOOL CALLS  +  SEPARATION OF CONCERNS RESTORED
   - The recurring "syntax error at the last line" was never bad Python: the
     dev model (qwen3-coder) emits write_file as a TEXT tool call in XML/JSON
     form, the runtime didn't parse it, and the closing
     </parameter></function></tool_call> tags leaked into the file. The same
     cause produced the "write_file never called" message. Fixed at the source:
     `_extract_toolcall_writes` recovers the real filename+content from the text
     tool call, and `_strip_toolcall_markup` scrubs any stray tags as a backstop
     — so the written file is clean and there is no syntax error to chase.
   - With the root cause gone, the developer's in-node re-prompt (added in 0e)
     was REMOVED to restore separation of concerns. The developer only writes;
     a genuine syntax error is now owned by the validator + deterministic
     routing (one visible, reportable bounce back to the developer). A purely
     advisory [PARSE CHECK] log line remains — it never fixes or routes.

0g. STATIC ANALYSIS NO LONGER WALKS VIRTUAL-ENVS  (fixes the validator hang)
   When a task pulled in real dependencies (e.g. "write a LangChain pipeline"),
   a venv could land in the workspace. The old walker only skipped `.git`, so
   get_workspace_files() pulled in thousands of site-packages files; the
   validator then built its module set from all of them and ran an
   unresolvable-import check over the entire dependency tree — embedding the
   full module list in every error — which exploded into a multi-MB hang.
   Fixes:
   - EXCLUDED_DIRS + pruning: the walker skips venvs (by name or pyvenv.cfg
     marker), site-packages, caches, build/VCS dirs. Reports and LLM prompts
     stay clean too.
   - The validator analyses only the workspace's own files, recognises INSTALLED
     packages via importlib.util.find_spec (so `from langchain_core …` no longer
     false-fails), and CAPS unresolvable-import output at 10 (no giant lists).
   - The tester adds `--ignore=<dir>` for any venv/dep dir in the workspace so
     pytest won't collect or measure coverage inside it.

0h. QA QUALITY + DETERMINISM, PACKAGE HINTS, CLEAN OUTPUT
   Triggered by dependency-heavy tasks (LangChain / asyncpg) that thrashed:
   - QA now writes ONE canonical test file (test_<primary>.py). Stale test/
     verify files are wiped at the start of each QA pass and consolidated to a
     single file at the end — so tests can't proliferate and a stale broken
     test from an earlier iteration can no longer doom the run.
   - QA is instructed to write BEHAVIOURAL tests that import the module and
     exercise it (real coverage), mocking external systems (DB/network/LLM via
     unittest.mock / monkeypatch). Source-text string-match "tests" are banned.
   - Validator unresolvable-import errors now suggest INSTALLED packages in the
     same family (e.g. langchain_openai → langchain_core/langchain_ollama) so
     the developer stops guessing at libraries that aren't there.
   - Researcher output is scrubbed of unparsed tool-call markup before it lands
     in research_notes / the report.
   - final_output is cleared at the start of every run, so artifacts from a
     previous task no longer linger.

0i. COVERAGE DEAD-LOOP BROKEN (asyncpg-style tasks)
   Symptom: tests PASS but the primary source sat at 0% coverage every cycle,
   so the 80% gate failed forever and the pipeline looped to the step cap. Root
   cause: the source imported an UNINSTALLED package (e.g. asyncpg) at module
   level, so `import main` would crash collection — QA couldn't cover it and
   kept re-implementing logic inline. Fixes:
   - Reviewer: a TEST_FAILED where tests PASS and only the coverage threshold is
     unmet now routes deterministically to QA (not the developer — rewriting
     working code can't raise coverage), with the concrete sys.modules-mock
     technique to import a module whose deps aren't installed.
   - QA prompt now teaches that same technique proactively.
   - Supervisor: stall detection ends the run early (after STALL_LIMIT=3 cycles)
     when the PRIMARY SOURCE's own coverage doesn't improve — turning a 30-step
     dead-loop into a fast, honest termination. (Researcher tool-call recovery
     intentionally deferred to V11.)

0j. BATCH INPUT FROM input.txt → PER-PROMPT RESULT FOLDERS
   The pipeline no longer asks for one task interactively. It reads numbered
   prompts from `input.txt` (or a path passed as argv[1]), runs each in order,
   and writes each prompt's artifact + reports into its own folder under
   final_output/<slug>/, where <slug> is a short name summarising the task
   (e.g. 'bitcoin', 'pydantic'). A master summary prints which prompts passed
   and where each landed. Falls back to interactive input if input.txt is
   absent.

What changed in v9 (kept):

1. FIXED WORKFLOW -> SUPERVISOR-ROUTED GRAPH
   The old graph had hardcoded edges (planner -> developer -> qa ->
   validator -> tester -> reviewer -> ...). Every worker node now
   returns to a `supervisor` node. The supervisor is an LLM call with
   a structured Literal output choosing the next node from:
       researcher | developer | qa | validator | tester | reviewer |
       save_artifact | end
   based on the full current state. A hard step cap (MAX_TOTAL_STEPS)
   prevents an indecisive model from looping forever — once hit, the
   supervisor is forced to "end" regardless of its own output.

2. FIXED TOOLS -> TOOL MENU
   developer/qa previously only had write_file/delete_file. They now
   have a full bound toolset and choose per step:
       write_file, delete_file   (filesystem)
       run_shell                 (generic shell command)
       execute_code              (run a specific file, capture output)
       git_tool                  (init/add/commit/diff/log)
       install_package           (pip / cargo)
       web_search, fetch_url     (retrieval, see #3)

3. NO RETRIEVAL -> RESEARCHER NODE + WEB TOOLS
   web_search hits DuckDuckGo's HTML endpoint (no API key) and returns
   titles+snippets; fetch_url retrieves a page and strips it to plain
   text. A dedicated `researcher` node can be dispatched by the
   supervisor when a task names a specific library/version/"best
   practices", and writes findings into state["research_notes"], which
   is included in every later developer/qa prompt. developer/qa can
   also call these tools directly mid-task if they hit something they
   don't know.

Both v8 reports (pipeline execution + code review) are kept and now
also log every supervisor decision.
"""

import subprocess
import shlex
import os
import sys
import re
import json
import shutil
import ast
import datetime as dt
import urllib.parse
from pathlib import Path
from typing import Dict, List, TypedDict, Literal, Optional

import requests
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

# ═════════════════════════════════════════════════════════════
# 1. WORKSPACE & LLM CONFIGURATION
# ═════════════════════════════════════════════════════════════
WORKSPACE_DIR = Path("./workspace")
BASE_OUTPUT_DIR = Path("./final_output")   # all per-prompt result folders live here
OUTPUT_DIR    = BASE_OUTPUT_DIR            # repointed per-prompt to BASE_OUTPUT_DIR/<slug>
REPORTS_DIR   = OUTPUT_DIR / "reports"


def _set_task_output_dirs(task_dir: Path) -> None:
    """Point the output + reports dirs at a specific per-prompt folder.

    Nodes and report generators read these as module globals at call time, so
    repointing them between prompts routes each prompt's artifact and reports
    into its own folder."""
    global OUTPUT_DIR, REPORTS_DIR
    OUTPUT_DIR = task_dir
    REPORTS_DIR = task_dir / "reports"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:8081")

MAX_TOTAL_STEPS = int(os.getenv("MAX_TOTAL_STEPS", "30"))
# Stop a repair loop after this many consecutive cycles where the primary
# source's coverage doesn't improve (kills the coverage dead-loop early).
STALL_LIMIT = int(os.getenv("STALL_LIMIT", "3"))
SHELL_TIMEOUT_S = 30
# Per-request timeout for the Ollama HTTP calls. Without this an unstable
# connection can hang a call FOREVER; the hang then only ends via Ctrl+C
# (a KeyboardInterrupt), which used to kill the whole batch. With a timeout a
# broken/stalled connection raises a normal, catchable exception instead, so the
# batch loop can log the prompt as ERROR and move on. Generous by default
# because CPU-only generation is slow.
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "600"))

DEV_LLM = ChatOllama(
    model="qwen3-coder:30b",
    temperature=0.15,
    base_url=OLLAMA_BASE_URL,
    num_predict=8192,   # qwen3-coder handles long files; give it room
    keep_alive="10m",
    num_ctx=16384,      # larger context — qwen3 supports it
    client_kwargs={"timeout": LLM_TIMEOUT_S},
)

REVIEWER_LLM = ChatOllama(
    model="qwen3:14b",
    temperature=0,          # temp 0 → maximum JSON-schema adherence
    base_url=OLLAMA_BASE_URL,
    num_predict=4096,
    keep_alive="10m",
    reasoning=False,        # qwen3 is a thinking model — try to disable CoT.
                            # langchain's flag is unreliable for qwen3, so the
                            # plain-text reads below also strip <think> blocks.
    client_kwargs={"timeout": LLM_TIMEOUT_S},
)

# ═════════════════════════════════════════════════════════════
# 2. STATE & SCHEMA DEFINITIONS
# ═════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    task:             str
    language:         Literal["python", "rust"]
    primary_artifact: str
    plan:             str
    files:            List[str]  # Authoritative filesystem metadata
    test_output:      str
    review_feedback:  str
    quality_feedback: str
    repair_history:   List[str]
    repair_target:    Literal["developer", "qa"]
    status:           Literal["PENDING", "VALIDATION_FAILED", "TEST_FAILED", "TESTS_PASSED", "QUALITY_FAILED", "APPROVED"]
    iterations:        int
    total_steps:       int
    research_notes:    str
    last_supervisor_reasoning: str
    best_src_coverage: float   # highest coverage% seen for the primary source
    stall_count:       int     # consecutive no-progress repair cycles
    _next:             str


class ReviewVerdict(BaseModel):
    review_feedback: str = Field(
        description="Detailed explanation of what failed and how to fix it."
    )
    repair_target: Literal["developer", "qa"] = Field(
        description=(
            "Who needs to fix this? "
            "'developer' if source code logic is wrong or files are missing. "
            "'qa' ONLY if tests import correctly but are stale or have wrong assertions."
        )
    )


class QualityVerdict(BaseModel):
    is_approved: bool = Field(
        description="True if the code architecture and quality are good, False if it requires major refactoring or has security flaws."
    )
    feedback: str = Field(
        description="If rejected, explain what must be fixed. If approved, provide a short positive note."
    )

# ═════════════════════════════════════════════════════════════
# 2b. RUN-WIDE LOGS (for reporting only)
# ═════════════════════════════════════════════════════════════
EXECUTION_LOG: List[dict] = []
REVIEW_LOG:    List[dict] = []
SUPERVISOR_LOG: List[dict] = []


def _log_event(node: str, detail: str):
    EXECUTION_LOG.append({
        "time": dt.datetime.now().strftime("%H:%M:%S"),
        "node": node,
        "detail": detail,
    })


def _log_review(iteration: int, status: str, output: str, feedback: str, target: str):
    REVIEW_LOG.append({
        "iteration": iteration, "status": status,
        "output": output, "feedback": feedback, "target": target,
    })


def _log_supervisor(step: int, status: str, decision: str, reasoning: str):
    SUPERVISOR_LOG.append({
        "step": step, "status": status,
        "decision": decision, "reasoning": reasoning,
    })

# ═════════════════════════════════════════════════════════════
# 3. AGENT TOOLS
# ═════════════════════════════════════════════════════════════
# QA WRITE GUARD
# While active, write_file refuses any non-`test_` filename. This freezes
# the developer's source files during the QA (test-authoring) phase so a
# confused local model cannot overwrite e.g. main.py with a stub.
_QA_WRITE_GUARD = {"active": False}


@tool
def write_file(filename: str, content: str) -> str:
    """[filesystem] Write or overwrite a file in the workspace."""
    base = os.path.basename(filename)
    if _QA_WRITE_GUARD["active"] and not base.startswith("test_"):
        return (
            f"REJECTED: During QA you may ONLY write test files. "
            f"'{filename}' is not a test file and source files are frozen. "
            f"Re-issue write_file with a name starting with 'test_' "
            f"(e.g. 'test_{base}'). Do NOT modify or recreate source files."
        )
    file_path = WORKSPACE_DIR / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully written to {filename}"


@tool
def delete_file(filename: str) -> str:
    """[filesystem] Delete a file from the workspace if no longer needed."""
    file_path = WORKSPACE_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return f"Successfully deleted {filename}"
    return f"File {filename} not found."


@tool
def run_shell(command: str) -> str:
    """[shell] Run an arbitrary shell command inside the workspace directory
    (e.g. 'ls -la', 'mkdir src', 'find . -name "*.py"'). Use this for
    inspecting the workspace or any OS-level operation that isn't covered
    by a more specific tool. Timeout is 30s; output is truncated to 4000 chars."""
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=SHELL_TIMEOUT_S,
        )
        out = f"Exit: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return out[:4000]
    except subprocess.TimeoutExpired:
        return f"Command timed out after {SHELL_TIMEOUT_S}s."
    except Exception as e:
        return f"Shell error: {e}"


@tool
def execute_code(filename: str, cmd_args: str = "") -> str:
    """[code execution] Run a specific source file in the workspace and
    capture its output. Picks the interpreter based on file extension
    (.py -> python3, .rs -> requires a prior 'cargo build'). Use this to
    sanity-check a script works before/independent of the formal test suite.
    Timeout 30s."""
    target = WORKSPACE_DIR / filename
    if not target.exists():
        return f"File {filename} not found in workspace."
    if filename.endswith(".py"):
        cmd = f"python3 {shlex.quote(filename)} {cmd_args}".strip()
    elif filename.endswith(".rs"):
        cmd = "cargo run -- " + cmd_args if cmd_args else "cargo run"
    else:
        return f"Don't know how to execute {filename}."
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=SHELL_TIMEOUT_S,
        )
        return (
            f"Exit: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )[:4000]
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {SHELL_TIMEOUT_S}s."
    except Exception as e:
        return f"Execution error: {e}"


@tool
def git_tool(operation: str, cmd_args: str = "") -> str:
    """[git] Run a git operation inside the workspace. operation is one of:
    'init', 'add', 'commit', 'diff', 'log', 'status'. cmd_args are extra
    arguments, e.g. operation='commit', cmd_args='-m "initial commit"'."""
    allowed = {"init", "add", "commit", "diff", "log", "status"}
    if operation not in allowed:
        return f"Operation '{operation}' not permitted. Allowed: {sorted(allowed)}"
    cmd = f"git {operation} {cmd_args}".strip()
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=SHELL_TIMEOUT_S,
        )
        return f"Exit: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"[:4000]
    except Exception as e:
        return f"Git error: {e}"


@tool
def install_package(manager: str, package: str) -> str:
    """[package installer] Install a dependency. manager is 'pip' or 'cargo'.
    For pip, runs 'pip install <package>'. For cargo, runs 'cargo add <package>'
    (requires an existing Cargo.toml in the workspace)."""
    if manager not in ("pip", "cargo"):
        return "manager must be 'pip' or 'cargo'."
    cmd = f"pip install {shlex.quote(package)}" if manager == "pip" else f"cargo add {shlex.quote(package)}"
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=60,
        )
        return f"Exit: {result.returncode}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-1000:]}"
    except subprocess.TimeoutExpired:
        return "Install timed out after 60s."
    except Exception as e:
        return f"Install error: {e}"


@tool
def web_search(query: str) -> str:
    """[documentation search] Search the web for up-to-date information —
    library docs, framework "best practices", API references, version notes.
    Returns a short list of titles + snippets + URLs. Use fetch_url afterward
    to read a specific page in full."""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        html = resp.text
        # crude extraction: result titles/snippets/links from DDG's lite HTML
        links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html)
        results = []
        for i, (url, title) in enumerate(links[:5]):
            title_clean = re.sub("<.*?>", "", title).strip()
            snippet_clean = re.sub("<.*?>", "", snippets[i]).strip() if i < len(snippets) else ""
            url_clean = urllib.parse.unquote(url)
            results.append(f"{i+1}. {title_clean}\n   {snippet_clean}\n   {url_clean}")
        if not results:
            return "No results parsed (search backend may have changed its HTML)."
        return "\n".join(results)
    except requests.exceptions.RequestException as e:
        return f"web_search unavailable (no network or blocked): {e}"


@tool
def fetch_url(url: str) -> str:
    """[browser] Fetch a web page and return its visible text, stripped of
    HTML, truncated to ~4000 chars. Use this to read a doc page found via
    web_search."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except requests.exceptions.RequestException as e:
        return f"fetch_url failed (no network or blocked): {e}"


FILESYSTEM_TOOLS = [write_file, delete_file]
FULL_TOOLSET     = [write_file, delete_file, run_shell, execute_code,
                    git_tool, install_package, web_search, fetch_url]
RESEARCH_TOOLS   = [web_search, fetch_url]


def reset_workspace():
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


# Directories we must never treat as workspace source: virtual-envs, vendored
# dependencies, caches, build output, VCS. Walking a real site-packages tree
# (e.g. when a task installs LangChain) produces thousands of files and, without
# this guard, exploded static analysis into a multi-megabyte hang.
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "myenv",
    "virtualenv", ".virtualenv", "site-packages", "node_modules", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "dist", "build", ".eggs",
    "__pypackages__", "target", ".idea", ".vscode",
}


def _is_dependency_path(rel_path) -> bool:
    """True if a (relative) path lives inside an excluded dependency/venv dir."""
    parts = Path(rel_path).parts
    return any(part in EXCLUDED_DIRS for part in parts) or "site-packages" in str(rel_path)


_INSTALLED_TOP_LEVEL = None


def _installed_top_level() -> set:
    """Top-level importable names of INSTALLED distributions (cached).

    Used to tell the developer which packages are actually available when an
    import can't be resolved — so it can pick an installed library (e.g.
    langchain_ollama) instead of guessing at one that isn't there."""
    global _INSTALLED_TOP_LEVEL
    if _INSTALLED_TOP_LEVEL is not None:
        return _INSTALLED_TOP_LEVEL
    tops: set = set()
    try:
        import importlib.metadata as _im
        for dist in _im.distributions():
            try:
                tl = dist.read_text("top_level.txt")
            except Exception:
                tl = None
            if tl:
                tops.update(t.strip() for t in tl.splitlines() if t.strip())
            name = (dist.metadata.get("Name") or "").strip()
            if name:
                tops.add(name.replace("-", "_"))
    except Exception:
        pass
    _INSTALLED_TOP_LEVEL = tops
    return tops


def _suggest_installed(missing_top: str, limit: int = 8) -> list:
    """Installed packages related to a missing import (same family prefix),
    e.g. 'langchain_openai' -> ['langchain_core', 'langchain_ollama', ...]."""
    installed = _installed_top_level()
    family = missing_top.split("_")[0].lower()
    if len(family) < 3:
        return []
    related = sorted(
        p for p in installed
        if p.lower().startswith(family) and p.lower() != missing_top.lower()
    )
    return related[:limit]


def get_workspace_files() -> List[str]:
    """Return relative file paths in the workspace, skipping dependency,
    virtual-env, cache and VCS directories. Pruning happens during the walk so
    os.walk never descends into (potentially enormous) site-packages trees."""
    file_list = []
    if not WORKSPACE_DIR.exists():
        return file_list
    for root, dirnames, filenames in os.walk(WORKSPACE_DIR):
        # Prune unwanted dirs in place: by name, or by the definitive venv
        # marker file `pyvenv.cfg` (catches venvs with non-standard names).
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS
            and not (Path(root) / d / "pyvenv.cfg").exists()
        ]
        for name in filenames:
            path = Path(root) / name
            rel = path.relative_to(WORKSPACE_DIR)
            file_list.append(str(rel))
    return file_list


def _is_test_artifact(rel_path) -> bool:
    """A pytest test file or a stray verify_* script that QA tends to spawn."""
    base = os.path.basename(rel_path)
    return base.endswith(".py") and (
        base.startswith("test_") or base.endswith("_test.py") or base.startswith("verify_")
    )


def _cleanup_test_artifacts(keep=None) -> None:
    """Delete test/verify artifacts from the workspace except `keep` (a basename).
    Prevents tests from accumulating across QA iterations — which is how a stale,
    broken test from an earlier pass kept failing forever."""
    for f in get_workspace_files():
        if _is_test_artifact(f) and os.path.basename(f) != keep:
            try:
                (WORKSPACE_DIR / f).unlink()
            except Exception:
                pass


def _consolidate_tests(canonical: str) -> None:
    """Guarantee exactly ONE test file, named `canonical`. If the model wrote
    test(s) under other names, the largest is renamed to canonical; the rest are
    removed. No-op-safe when canonical already exists."""
    tests = [f for f in get_workspace_files() if _is_test_artifact(f)]
    if not any(os.path.basename(t) == canonical for t in tests) and tests:
        largest = max(tests, key=lambda t: (WORKSPACE_DIR / t).stat().st_size)
        try:
            (WORKSPACE_DIR / largest).rename(WORKSPACE_DIR / canonical)
        except Exception:
            pass
    _cleanup_test_artifacts(keep=canonical)


def _format_workspace_contents() -> str:
    """Read directly from the authoritative filesystem for LLM prompts."""
    file_list = get_workspace_files()
    if not file_list:
        return "(workspace is empty)"
    
    output = []
    for rel_path in file_list:
        path = WORKSPACE_DIR / rel_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            output.append(f"--- {rel_path} ---\n{content}")
        except UnicodeDecodeError:
            continue  # skip binary files
    return "\n".join(output)

# ═════════════════════════════════════════════════════════════
# 4. MICRO-AGENT ENGINE
# ═════════════════════════════════════════════════════════════
def _looks_like_code(text: str) -> bool:
    """Heuristic: does this text look like Python/Rust source code?"""
    code_signals = [
        r'\bdef \w+\s*\(',        # Python function
        r'\bclass \w+',           # Python class
        r'\bif __name__',         # Python main guard
        r'\bimport \w+',          # Python import
        r'\bfn \w+\s*\(',         # Rust function
        r'\blet mut \b',          # Rust variable
        r'#\[derive\(',           # Rust derive
    ]
    for sig in code_signals:
        if re.search(sig, text):
            return True
    return False


def _is_placeholder_stub(content: str) -> bool:
    """
    Heuristic: does this source file look like a throwaway stub rather than a
    real solution? Catches the classic clobber stub (e.g. a lone function that
    just returns "Hello, World!") so it can never reach approval.
    Conservative on purpose — only fires on clearly-trivial files.
    """
    c = content.strip()
    if not c:
        return True
    low = c.lower()
    if "hello, world" in low and len(c) < 200:
        return True
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False  # syntax errors are caught separately by the validator
    # A module whose entire body is a single function with a one-line trivial
    # body (pass / bare return / a single literal expression or return-literal).
    top = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    funcs = [n for n in top if isinstance(n, ast.FunctionDef)]
    if len(top) == 1 and len(funcs) == 1:
        body = funcs[0].body
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Return) and (
                stmt.value is None or isinstance(stmt.value, ast.Constant)
            ):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                return True
    return False


# Phrases that signal the real logic was faked/stubbed rather than implemented.
# Kept high-precision: these rarely appear in genuine complete solutions, so the
# anti-stub guard won't false-trip on real code and start a loop.
_STUB_PHRASES = (
    "in a real implementation",
    "in a real environment",
    "in a production environment",
    "in an actual implementation",
    "would connect to the database",
    "we would connect",
    "we would use the copy",
    "we would insert",
    "for demonstration purposes",
    "this is a demonstration",
    "this is a demo script",
    "replace with your actual",
    "replace with your real",
    "todo: implement",
    "not implemented yet",
    "placeholder implementation",
    "this would be implemented",
    "in a working environment",
)


def _detect_stub_phrase(content: str):
    """Return the first stub/placeholder phrase found in the source, or None.

    Targets the observed failure where the developer dodges a hard requirement
    by commenting out the real calls and leaving 'In a real implementation, we
    would …' text plus a print().
    """
    low = content.lower()
    for phrase in _STUB_PHRASES:
        if phrase in low:
            return phrase
    return None


_TOOLCALL_TAG_RE = re.compile(
    r'</?(?:tool_call|function|parameter)\b[^>]*>', re.IGNORECASE
)


def _strip_toolcall_markup(text: str) -> str:
    """Remove stray hermes/XML tool-call tags (<tool_call>, <function=...>,
    <parameter=...> and their closing tags) that local models emit as plain text
    when the runtime fails to parse them as real tool calls. Leaves the inner
    text intact so any embedded code survives clean."""
    return _TOOLCALL_TAG_RE.sub("", text)


def _extract_toolcall_writes(text: str) -> list:
    """Recover write_file calls a model emitted as TEXT rather than as a parsed
    tool call (the root cause of leaked </parameter></function></tool_call> tags).

    Handles the two qwen3 formats:
      XML:  <function=write_file><parameter=filename>x.py</parameter>
            <parameter=content>CODE</parameter></function>
      JSON: <tool_call>{"name":"write_file","arguments":{"filename":..,"content":..}}</tool_call>

    Returns a list of (filename_or_None, content). Empty list if none found,
    so callers fall through to the normal fenced/whole-response strategies.
    """
    writes: list = []

    for block in re.findall(
        r'<function\s*=\s*["\']?write_file["\']?\s*>(.*?)</function>',
        text, re.DOTALL | re.IGNORECASE,
    ):
        cm = re.search(
            r'<parameter\s*=\s*["\']?content["\']?\s*>(.*?)</parameter>',
            block, re.DOTALL | re.IGNORECASE,
        )
        if not cm:
            continue
        fm = re.search(
            r'<parameter\s*=\s*["\']?(?:filename|path|file)["\']?\s*>(.*?)</parameter>',
            block, re.DOTALL | re.IGNORECASE,
        )
        writes.append((fm.group(1).strip() if fm else None, cm.group(1)))

    for raw in re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("name") == "write_file":
            args = obj.get("arguments") or obj.get("args") or {}
            if isinstance(args, dict) and args.get("content") is not None:
                writes.append((
                    args.get("filename") or args.get("path") or args.get("file"),
                    str(args.get("content")),
                ))

    return writes


def _extract_toolcall_invocations(text: str, valid_names: set) -> list:
    """Recover ANY tool call a model emitted as TEXT (not parsed by the runtime).

    Generalises _extract_toolcall_writes to every tool (web_search, fetch_url,
    write_file, …) so research and other tools actually execute instead of
    leaking <function=…></tool_call> markup. Returns [(name, args_dict)].
    Only calls whose name is in `valid_names` are returned.
    """
    calls: list = []
    for m in re.finditer(
        r'<function\s*=\s*["\']?([\w\-]+)["\']?\s*>(.*?)</function>',
        text, re.DOTALL | re.IGNORECASE,
    ):
        name = m.group(1).strip()
        if valid_names and name not in valid_names:
            continue
        args = {}
        for pm in re.finditer(
            r'<parameter\s*=\s*["\']?([\w\-]+)["\']?\s*>(.*?)</parameter>',
            m.group(2), re.DOTALL | re.IGNORECASE,
        ):
            args[pm.group(1).strip()] = pm.group(2).strip()
        calls.append((name, args))

    for raw in re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name or (valid_names and name not in valid_names):
            continue
        args = obj.get("arguments") or obj.get("args") or {}
        if isinstance(args, dict):
            calls.append((name, args))

    return calls
    """
    Fallback: extract code from LLM plain-text output and write it to disk when
    write_file was never called as a tool.

    Strategy:
    1. Collect fenced ```...``` blocks. Blocks with an explicit filename hint
       (leading `# foo.py` comment, or "file: foo.py" just before the block)
       go to that name.
    2. Unhinted blocks: pick the single BEST one for the primary file — the
       longest block that parses as Python — instead of blindly writing block 0
       and dumping the rest into junk 'extra_' files. This avoids spurious
       syntax errors from trailing prose or example snippets.
    3. If there are no fences at all but the whole response looks like code,
       use that.
    The QA write-guard is honoured here too: during QA this path will not write
    a non-`test_` file, so it can't clobber frozen source.
    """
    if not text or not text.strip():
        return False

    # ── Strategy 0: the model emitted a TEXT-FORMAT write_file tool call that
    #    the runtime didn't parse (qwen3 XML/JSON style). Recognise it and write
    #    the clean content directly. This is the ROOT CAUSE of the leaked
    #    </parameter></tool_call> tags that broke parsing every run. ──
    tc_writes = _extract_toolcall_writes(text)
    if tc_writes:
        wrote_any = False
        for fname, content in tc_writes:
            fname = (fname or default_filename).strip() or default_filename
            content = content.strip("\n")
            if not content.strip():
                continue
            base = os.path.basename(fname)
            if _QA_WRITE_GUARD["active"] and not base.startswith("test_"):
                print(f"  [FALLBACK WRITER] Skipped '{fname}' — source is frozen during QA.")
                continue
            fp = WORKSPACE_DIR / fname
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            print(f"  [FALLBACK WRITER] Recovered text tool-call → wrote '{fname}' "
                  f"({len(content)} chars)")
            wrote_any = True
        if wrote_any:
            return True

    # Defensive: strip any residual tool-call tags so they can't pollute the
    # fenced/whole-response extraction below (no-op when there are none).
    text = _strip_toolcall_markup(text)

    is_py = default_filename.endswith(".py")
    hinted: list = []       # (filename, code) with an explicit name
    unhinted: list = []     # code blocks with no filename hint

    fenced = re.compile(
        r'```(?:python|py|rust|sh|bash)?\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE,
    )
    for code in fenced.findall(text):
        code = code.strip()
        if not code or len(code) < 20:
            continue
        fname_match = re.match(r'^#\s*([\w./\\-]+\.(?:py|rs))\s*\n', code)
        if fname_match:
            hinted.append((fname_match.group(1).strip(), code[fname_match.end():]))
            continue
        pre_text = text[:text.find(code)]
        pre_fname = re.search(
            r'(?:file(?:name)?|save\s+(?:as|to))[\s:]+`?([\w./\\-]+\.(?:py|rs))`?',
            pre_text[-200:], re.IGNORECASE,
        )
        if pre_fname:
            hinted.append((pre_fname.group(1).strip(), code))
        else:
            unhinted.append(code)

    # Strategy 3: no fences, but the whole response is raw source
    if not hinted and not unhinted and _looks_like_code(text):
        lines = text.splitlines()
        code_start = 0
        for idx, line in enumerate(lines):
            if re.match(r'\s*(def |class |import |from |fn |#!|use |let |pub )', line):
                code_start = idx
                break
        code = "\n".join(lines[code_start:]).strip()
        if code and len(code) > 30:
            unhinted.append(code)

    blocks_to_write: list = list(hinted)

    # Choose ONE primary block for default_filename from the unhinted blocks:
    # prefer the longest that parses (py), else the longest overall.
    if unhinted and not any(fn == default_filename for fn, _ in blocks_to_write):
        def _score(c: str):
            parses = True
            if is_py:
                try:
                    ast.parse(c)
                except SyntaxError:
                    parses = False
            return (parses, len(c))
        primary = max(unhinted, key=_score)
        blocks_to_write.append((default_filename, primary))

    written = False
    for fname, code in blocks_to_write:
        if not code.strip():
            continue
        base = os.path.basename(fname)
        if _QA_WRITE_GUARD["active"] and not base.startswith("test_"):
            print(f"  [FALLBACK WRITER] Skipped '{fname}' — source is frozen during QA.")
            continue
        file_path = WORKSPACE_DIR / fname
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code, encoding="utf-8")
        print(f"  [FALLBACK WRITER] Extracted and wrote '{fname}' ({len(code)} chars)")
        written = True

    return written



def _strip_thinking(text: str) -> str:
    """
    Qwen3-coder emits <think>...</think> blocks before its real response.
    Strip them so downstream code/fallback parsers only see the actual output.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _extract_first_json_obj(text: str) -> Optional[dict]:
    """Pull the first balanced {...} JSON object out of arbitrary text.

    Handles ```json fenced blocks and bare objects. Returns a dict or None.
    Used to recover a verdict when a local model ignores structured-output
    formatting and emits prose around (or instead of) clean JSON.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    break
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _salvage_verdict(llm: ChatOllama, prompt: str, keys: tuple) -> Optional[dict]:
    """Fallback for when llm.with_structured_output() throws on a local model.

    Re-asks in plain text demanding a single bare JSON object, then extracts it.
    Returns a (possibly partial) dict, or None if nothing parseable came back.
    """
    ask = (
        prompt
        + "\n\nRespond with ONLY a single JSON object containing the keys "
        + ", ".join(repr(k) for k in keys)
        + ". No prose, no markdown, no code fences. /no_think"
    )
    try:
        raw = llm.invoke([HumanMessage(content=ask)]).content or ""
    except Exception:
        return None
    return _extract_first_json_obj(_strip_thinking(raw))


def _reviewer_text(prompt: str) -> str:
    """Invoke REVIEWER_LLM for a short PLAIN-TEXT answer.

    qwen3 is a reasoning model: even with reasoning=False (unreliable in
    langchain for qwen3) a <think> block can leak into .content and poison
    downstream parsing (filename pick, language detection, routing). We append
    the documented '/no_think' soft switch AND strip any think block that still
    slips through, so callers always get clean text.
    """
    resp = REVIEWER_LLM.invoke(prompt + "\n\n/no_think")
    return _strip_thinking(resp.content or "")


def run_react_loop(
    llm: ChatOllama,
    tools: list,
    system_msg: str,
    user_msg: str,
    max_steps: int = 12,
    require_write: bool = False,
    default_filename: str = "main.py",
) -> str:
    """
    ReAct loop. If require_write=True and no write_file tool call was made
    by the end of the loop, attempt to extract code from plain-text output
    as a fallback before returning.
    """
    bound = llm.bind_tools(tools)
    messages = [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]
    tool_map = {t.name: t for t in tools}

    response_content = ""
    write_tool_called = False
    executed_text_calls: set = set()   # dedupe recovered text tool-calls

    for step_i in range(max_steps):
        try:
            response = bound.invoke(messages)
        except Exception as e:
            print(f"  [LLM ERROR step {step_i}]: {e}")
            break

        messages.append(response)

        if response.content:
            response_content = _strip_thinking(response.content)

        # Normalize tool_calls — some Ollama versions return dicts, some objects
        raw_calls = getattr(response, "tool_calls", None) or []
        tool_calls = []
        for tc in raw_calls:
            if isinstance(tc, dict):
                tool_calls.append(tc)
            else:
                tool_calls.append({
                    "name": getattr(tc, "name", ""),
                    "args": getattr(tc, "args", {}) or {},
                    "id":   getattr(tc, "id", "") or f"tc_{step_i}",
                })

        if not tool_calls:
            # No PARSED tool calls. The model may have emitted a TEXT-FORMAT tool
            # call the runtime didn't parse (qwen3 style). Recover and EXECUTE it
            # so tools like web_search actually run instead of leaking markup.
            recovered = _extract_toolcall_invocations(response_content, set(tool_map.keys()))
            pending = []
            for name, args in recovered:
                key = (name, tuple(sorted((k, str(v)) for k, v in args.items())))
                if key not in executed_text_calls:
                    executed_text_calls.add(key)
                    pending.append((name, args))
            if not pending:
                break   # nothing new to execute — the model is done
            for name, args in pending:
                fn = tool_map.get(name)
                if name == "write_file":
                    write_tool_called = True
                if fn:
                    try:
                        clean = {k: v for k, v in args.items() if not k.startswith("v__")}
                        result = fn.invoke(clean)
                    except Exception as e:
                        result = f"Tool execution error ({name}): {e}"
                else:
                    result = f"Unknown tool '{name}'."
                # Feed the result back as a human turn (format-safe — the AI msg
                # had no parsed tool_calls to attach a ToolMessage to).
                messages.append(HumanMessage(
                    content=f"[recovered tool '{name}' executed] Result:\n{str(result)[:4000]}"
                ))
            continue   # let the model use the results

        for tc in tool_calls:
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", {}) or {}
            tc_id   = tc.get("id", "") or f"tc_{step_i}_{tc_name}"
            fn      = tool_map.get(tc_name)

            if tc_name == "write_file":
                write_tool_called = True

            if fn:
                try:
                    clean_args = {k: v for k, v in tc_args.items() if not k.startswith("v__")}
                    result = fn.invoke(clean_args)
                except Exception as e:
                    result = f"Tool execution error ({tc_name}): {e}"
            else:
                result = f"Unknown tool '{tc_name}'. Available: {list(tool_map.keys())}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))

    # ── Fallback: code models often emit a fenced block, or a TEXT-FORMAT
    #    tool call the runtime didn't parse, instead of a real write_file call.
    #    Both are recovered here — expected, not an error. ──
    if require_write and not write_tool_called:
        print("  [INFO] No parsed write_file call — recovering code from the "
              "response (inline block or text tool-call; normal for code models).")
        _extract_code_blocks_and_write(response_content, default_filename)

    return response_content

# ═════════════════════════════════════════════════════════════
# 5. NODES
# ═════════════════════════════════════════════════════════════
def _direct_write_fallback(
    llm: ChatOllama,
    task: str,
    filename: str,
    plan: str = "",
    repair_feedback: str = "",
    is_test: bool = False,
    source_stem: str = "",
) -> bool:
    """
    NUCLEAR FALLBACK: Call LLM with NO tools at all — just ask it to output
    raw file content. The host process writes it directly to disk.
    Used when bind_tools / tool-calling fails entirely on the local model.
    Returns True if a file was written.
    """
    role = "QA engineer writing pytest tests" if is_test else "Python developer"
    target_desc = (
        f"a pytest test file named `{filename}` that imports from `{source_stem}` and tests it"
        if is_test
        else f"a Python source file named `{filename}` that solves the task"
    )
    repair_section = f"\nPREVIOUS FEEDBACK TO FIX:\n{repair_feedback}\n" if repair_feedback else ""
    plan_section   = f"\nPLAN:\n{plan}\n" if plan and not is_test else ""

    prompt = (
        f"You are a {role}.\n"
        f"OUTPUT RULES — follow exactly:\n"
        f"1. Output ONLY the complete contents of {target_desc}.\n"
        f"2. Start your response with the line: ```python\n"
        f"3. End your response with the line: ```\n"
        f"4. Do NOT include any explanation, commentary, or text outside the code block.\n"
        f"5. The code must be complete and runnable as-is.\n"
        f"{plan_section}"
        f"{repair_section}"
        f"\nTASK: {task}\n"
    )

    print(f"  [DIRECT WRITER] Bypassing tool-calling. Asking LLM for raw `{filename}` content...")
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = _strip_thinking(response.content or "")
    except Exception as e:
        print(f"  [DIRECT WRITER ERROR] LLM call failed: {e}")
        return False

    if not text.strip():
        print("  [DIRECT WRITER] Empty response from LLM.")
        return False

    # Try the aggressive extractor first
    wrote = _extract_code_blocks_and_write(text, filename)
    if wrote:
        return True

    # Last resort: if the model output looks like code, write it raw
    if _looks_like_code(text):
        file_path = WORKSPACE_DIR / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text.strip(), encoding="utf-8")
        print(f"  [DIRECT WRITER] Wrote raw LLM output to '{filename}' ({len(text)} chars)")
        return True

    print("  [DIRECT WRITER] Could not extract code from LLM response.")
    print(f"  [DIRECT WRITER] Response preview: {text[:300]!r}")
    return False


def _detect_language(task: str) -> str:
    task_lower = task.lower()
    has_python = re.search(r"\bpython\b", task_lower) is not None
    has_rust   = re.search(r"\brust\b", task_lower) is not None

    if has_python and not has_rust:
        return "python"
    if has_rust and not has_python:
        return "rust"

    lang_resp = _reviewer_text(
        f"Task: {task}\n"
        "Which programming language should this be implemented in? "
        "Reply with EXACTLY one word, lowercase: python or rust. "
        "No punctuation, no explanation."
    ).strip().lower()
    first_word = re.split(r"\W+", lang_resp.strip())[0] if lang_resp else ""
    return "rust" if first_word == "rust" else "python"


def planner(state: AgentState):
    print("\n[NODE] Planner")
    reset_workspace()
    # Clear stale artifacts from previous runs (e.g. a leftover script or test
    # file) so final_output only ever reflects the current task. Reports are
    # regenerated at the end, so wiping the whole dir here is safe.
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            try:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            except Exception:
                pass
    EXECUTION_LOG.clear()
    REVIEW_LOG.clear()
    SUPERVISOR_LOG.clear()

    language = _detect_language(state["task"])
    _log_event("planner", f"Detected language: {language}")

    artifact_resp = _reviewer_text(
        f"Task: {state['task']}\nLanguage: {language}\n"
        "Based on the task, determine the best primary filename.\n"
        "Output ONLY the target filename (e.g., 'main.py', 'app.py', or 'src/main.rs'). "
        "One line, no explanation."
    ).strip()
    primary_artifact = next(
        (ln.strip() for ln in artifact_resp.splitlines() if ln.strip()), "main.py"
    )

    plan = _reviewer_text(
        f"Task: {state['task']}\nLanguage: {language}\nCreate a step-by-step coding plan."
    )

    _log_event("planner", f"Primary artifact: {primary_artifact}")

    return {
        "plan": plan,
        "language": language,
        "primary_artifact": primary_artifact,
        "iterations": 0,
        "total_steps": 0,
        "files": [],
        "test_output": "",
        "review_feedback": "",
        "quality_feedback": "",
        "repair_history": [],
        "research_notes": "",
        "repair_target": "developer",
        "status": "PENDING",
        "last_supervisor_reasoning": "",
        "_next": "",
    }


def researcher(state: AgentState):
    print("\n[NODE] Researcher (External Retrieval)")
    sys_msg = (
        "You are a technical researcher. Use 'web_search' to find current "
        "documentation, library versions, and best practices relevant to the "
        "task, then 'fetch_url' to read the most promising 1-2 results in full. "
        "Finish by writing a concise plain-text summary of the concrete, "
        "actionable findings (no tool calls in your final message)."
    )
    user_msg = (
        f"Task: {state['task']}\nLanguage: {state['language']}\n\n"
        "Research anything in this task that depends on current external "
        "knowledge (specific libraries, frameworks, version-specific APIs, "
        "'best practices' for a given year, etc). If nothing in the task "
        "requires external knowledge, say so briefly instead of searching."
    )
    # Reduced max steps to 3 to improve execution speed for trivial tasks
    notes = run_react_loop(DEV_LLM, RESEARCH_TOOLS, sys_msg, user_msg, max_steps=3)

    # The model may emit web_search/fetch_url as TEXT tool-calls the runtime
    # didn't parse; scrub that markup so raw <function=…></tool_call> tags don't
    # leak into research_notes (and from there into the report).
    notes = _strip_toolcall_markup(notes or "")

    # Force findings block even if blank, stopping empty `has_research` check issues
    if not notes or not notes.strip():
        notes = "No specific findings generated."
        
    _log_event("researcher", "Findings recorded")
    prior = state.get("research_notes", "")
    combined = (prior + "\n\n---\n\n" + notes).strip() if prior else notes
    return {"research_notes": combined}


def developer(state: AgentState):
    iters = state.get("iterations", 0) + 1
    print(f"\n[NODE] Developer (Iteration {iters})")

    # The developer writes/overwrites source files, so the QA freeze must be off.
    _QA_WRITE_GUARD["active"] = False

    primary = state.get("primary_artifact", "main.py")
    history = state.get("repair_history", [])
    history_str = (
        "\n".join(f"Attempt {i+1} Feedback:\n{fb}\n---" for i, fb in enumerate(history))
        if history else "None — first attempt."
    )

    sys_msg = (
        "You are an expert software developer. Your ONLY job is to write code files to disk.\n"
        "━━━ MANDATORY PROTOCOL ━━━\n"
        "1. ALWAYS call the `write_file` tool. Never just print code — it will be discarded.\n"
        "2. Write the complete file content in a single `write_file` call.\n"
        "3. Do NOT write test files (filenames starting with 'test_') — QA handles those.\n"
        "4. After writing, call `execute_code` to sanity-check the file runs without errors.\n"
        "5. If execution shows an error, fix it with another `write_file` call.\n"
        "━━━ AVAILABLE TOOLS ━━━\n"
        "write_file, delete_file, run_shell, execute_code, git_tool, install_package, web_search, fetch_url"
    )

    user_msg = (
        f"TASK: {state['task']}\n\n"
        f"PRIMARY FILENAME (you MUST use this exact name): `{primary}`\n\n"
        f"PLAN:\n{state['plan']}\n\n"
        f"RESEARCH NOTES:\n{state.get('research_notes', '(none)')}\n\n"
        f"CURRENT WORKSPACE:\n{_format_workspace_contents()}\n\n"
        f"REPAIR HISTORY (avoid repeating these mistakes):\n{history_str}\n\n"
        f"⚠️  ACTION REQUIRED: Call `write_file` with filename=`{primary}` and the complete source code NOW.\n"
        "Then call `execute_code` to verify it. Do not stop until write_file has been called."
    )

    run_react_loop(
        DEV_LLM, FULL_TOOLSET, sys_msg, user_msg,
        max_steps=14, require_write=True, default_filename=primary,
    )

    files_written = get_workspace_files()
    has_source = any(
        f.endswith(".py") and "test" not in f.lower() for f in files_written
    )

    # ── Escalation 1: minimal tool-calling prompt ──
    if not has_source:
        print("  [GUARD] No source file found after loop. Sending emergency write prompt...")
        emergency_sys = (
            "You are a code-writing assistant. Your ONLY action must be to call `write_file`.\n"
            "Do not explain, do not think out loud. Just call the tool immediately."
        )
        emergency_user = (
            f"Write the solution for this task to the file `{primary}`:\n{state['task']}\n\n"
            "Call `write_file` right now with the complete code."
        )
        run_react_loop(
            DEV_LLM, [write_file, execute_code], emergency_sys, emergency_user,
            max_steps=4, require_write=True, default_filename=primary,
        )
        files_written = get_workspace_files()
        has_source = any(f.endswith(".py") and "test" not in f.lower() for f in files_written)

    # ── Escalation 2: nuclear fallback — bypass tool-calling entirely ──
    if not has_source:
        repair_feedback = history[-1] if history else ""
        wrote = _direct_write_fallback(
            DEV_LLM,
            task=state["task"],
            filename=primary,
            plan=state.get("plan", ""),
            repair_feedback=repair_feedback,
            is_test=False,
        )
        if wrote:
            files_written = get_workspace_files()

    # ── Parse check (ADVISORY ONLY): if the primary source still doesn't parse,
    #    just log it. We do NOT fix it here — the validator detects the syntax
    #    error and the reviewer's deterministic routing sends it straight back to
    #    the developer, all visible in the reports. Separation of concerns. ──
    primary_path = WORKSPACE_DIR / primary
    if not (primary_path.suffix == ".py" and primary_path.exists()):
        cand = next((f for f in files_written
                     if f.endswith(".py") and "test" not in f.lower()), None)
        primary_path = (WORKSPACE_DIR / cand) if cand else None

    if primary_path and primary_path.exists() and primary_path.suffix == ".py":
        try:
            ast.parse(primary_path.read_text(encoding="utf-8"))
        except SyntaxError as se:
            print(f"  [PARSE CHECK] {primary_path.name} has a syntax error at "
                  f"line {se.lineno}: {se.msg} — leaving it for the validator; "
                  "deterministic routing will return here to fix it.")

    _log_event("developer", f"Iteration {iters} complete — files: {files_written}")
    return {"files": files_written, "iterations": iters, "status": "PENDING"}


def qa(state: AgentState):
    print("\n[NODE] QA (Test Generation/Repair)")

    # Ensure source files exist before asking QA to write tests
    source_files = [f for f in state.get("files", [])
                    if (f.endswith(".py") or f.endswith(".rs")) and "test" not in f.lower()]
    if not source_files:
        print("  [QA GUARD] No source files found — skipping QA, sending back to developer.")
        _log_event("qa", "Skipped — no source files to test")
        return {"files": get_workspace_files(), "status": "VALIDATION_FAILED",
                "test_output": "VALIDATION FAILED:\nCRITICAL: No source files exist. QA cannot write tests."}

    # Freeze source files for the duration of QA: write_file now rejects any
    # non-`test_` filename, so the test author cannot clobber the real source.
    _QA_WRITE_GUARD["active"] = True

    # Canonical, deterministic test filename. Everything consolidates to this so
    # tests cannot proliferate across iterations (which previously let a stale,
    # broken test from an earlier pass doom the whole run).
    primary = state.get("primary_artifact", source_files[0])
    primary_stem = Path(primary).stem
    canonical_test = f"test_{primary_stem}.py"

    # Start each QA pass from a clean slate — drop all stale test/verify files.
    _cleanup_test_artifacts(keep=None)

    source_list = "\n".join(f"  - {f}" for f in source_files)
    history = state.get("repair_history", [])
    history_str = (
        "\n".join(f"Attempt {i+1} Feedback:\n{fb}\n---" for i, fb in enumerate(history))
        if history else "None — write fresh tests."
    )

    sys_msg = (
        "You are an expert QA engineer. Write ONE pytest file that actually "
        "EXERCISES the code.\n"
        "━━━ MANDATORY PROTOCOL ━━━\n"
        f"1. Call `write_file` with filename `{canonical_test}`. Do NOT create any "
        "other test files.\n"
        "2. IMPORT the module under test and call its functions/classes so real "
        "code runs (coverage must come from execution, not inspection).\n"
        "3. FORBIDDEN: 'tests' that read the source as text and string-match it "
        "(e.g. `assert \"foo\" in open('main.py').read()`). They prove nothing.\n"
        "4. MOCK external systems so tests need no real infrastructure: use "
        "`unittest.mock` (Mock/MagicMock/AsyncMock, patch) or pytest monkeypatch "
        "for databases (e.g. asyncpg.connect), network, and LLM/API clients; use "
        "tmp_path / small fixtures for file I/O.\n"
        "   If the source imports a third-party package that may NOT be installed "
        "(e.g. asyncpg, langchain_*), inject a fake into sys.modules BEFORE you "
        "import the module, so it stays importable and its lines get covered:\n"
        "       import sys\n"
        "       from unittest.mock import MagicMock\n"
        "       sys.modules.setdefault('asyncpg', MagicMock())\n"
        "       import main   # then call main.<functions>() with mocks\n"
        "5. Test behaviour and edge cases. Use stdlib + pytest + unittest.mock "
        "only; do NOT install extra packages.\n"
        "━━━ AVAILABLE TOOLS ━━━\n"
        "write_file, delete_file, run_shell, execute_code, git_tool, install_package, web_search, fetch_url"
    )

    user_msg = (
        f"TASK: {state['task']}\n\n"
        "SOURCE FILES YOU MUST IMPORT FROM (use exact stem names):\n"
        f"{source_list}\n\n"
        f"CURRENT WORKSPACE:\n{_format_workspace_contents()}\n\n"
        f"REPAIR HISTORY (avoid repeating these mistakes):\n{history_str}\n\n"
        f"⚠️  ACTION REQUIRED: Call `write_file` with filename `{canonical_test}` "
        "containing pytest tests that IMPORT the module and exercise its functions "
        "(mock any DB/network/LLM). Then run `pytest --collect-only` to verify they load."
    )

    run_react_loop(
        DEV_LLM, FULL_TOOLSET, sys_msg, user_msg,
        max_steps=14, require_write=True, default_filename=canonical_test,
    )

    files_written = get_workspace_files()
    test_files = [f for f in files_written if "test" in f.lower() and f.endswith(".py")]

    # ── Escalation 1: minimal tool-calling prompt ──
    if not test_files:
        print("  [GUARD] No test file found after loop. Sending emergency write prompt...")
        emergency_sys = (
            "You are a test-writing assistant. Your ONLY action must be to call `write_file`.\n"
            "Do not explain. Just call the tool immediately."
        )
        emergency_user = (
            f"Write basic pytest tests for the module `{primary_stem}` to the file "
            f"`{canonical_test}`. IMPORT `{primary_stem}` and call its functions; mock any "
            "DB/network/LLM. Do not string-match the source text. Call `write_file` right now."
        )
        run_react_loop(
            DEV_LLM, [write_file, execute_code], emergency_sys, emergency_user,
            max_steps=4, require_write=True, default_filename=canonical_test,
        )
        files_written = get_workspace_files()
        test_files = [f for f in files_written if "test" in f.lower() and f.endswith(".py")]
    if not test_files:
        repair_feedback = history[-1] if history else ""
        _direct_write_fallback(
            DEV_LLM,
            task=state["task"],
            filename=canonical_test,
            repair_feedback=repair_feedback,
            is_test=True,
            source_stem=primary_stem,
        )

    # Consolidate to exactly ONE canonical test file (rename/dedupe), so nothing
    # proliferates and no stale broken test survives into the next iteration.
    _consolidate_tests(canonical_test)
    files_written = get_workspace_files()

    _log_event("qa", f"Test generation complete — files: {files_written}")
    # Unfreeze source files now that test authoring is done.
    _QA_WRITE_GUARD["active"] = False
    return {"files": files_written, "status": "PENDING"}


def validator(state: AgentState):
    print("\n[NODE] Validator (Static Analysis)")
    files = state.get("files", [])
    errors = []

    if state["language"] == "rust":
        if "Cargo.toml" not in files:
            errors.append("CRITICAL: Missing Cargo.toml.")
        else:
            try:
                cargo = (WORKSPACE_DIR / "Cargo.toml").read_text(encoding="utf-8")
                if "[package]" not in cargo or "name" not in cargo:
                    errors.append("CRITICAL: Cargo.toml is malformed.")
            except Exception as e:
                errors.append(f"CRITICAL: Could not read Cargo.toml: {e}")
        if "src/main.rs" not in files and "src/lib.rs" not in files:
            errors.append("CRITICAL: Missing src/main.rs or src/lib.rs.")
    else:
        py_files = [f for f in files if f.endswith(".py") and not _is_dependency_path(f)]
        test_files = [f for f in py_files if "test" in f.lower()]

        if not py_files:
            errors.append("CRITICAL: No .py files generated.")
        if not test_files:
            errors.append("CRITICAL: No test files detected.")

        # ── Placeholder/stub guard: don't let a clobbered or trivial primary
        #    source sail through to approval. Skip when the task itself is
        #    genuinely trivial (e.g. literally "write hello world").
        task_text = state.get("task", "").lower()
        task_is_trivial = "hello" in task_text and len(task_text) < 60
        if not task_is_trivial:
            primary = state.get("primary_artifact", "")
            src_non_test = [f for f in py_files if "test" not in f.lower()]
            check_target = (
                primary if primary in src_non_test
                else (src_non_test[0] if src_non_test else None)
            )
            if check_target:
                try:
                    pc = (WORKSPACE_DIR / check_target).read_text(encoding="utf-8")
                    if _is_placeholder_stub(pc):
                        errors.append(
                            f"CRITICAL: Primary source '{check_target}' is a "
                            "placeholder/stub and does not implement the task. "
                            "The developer must write the real, complete solution."
                        )
                    else:
                        stub_phrase = _detect_stub_phrase(pc)
                        if stub_phrase:
                            errors.append(
                                f"STUB DETECTED in '{check_target}': the code contains "
                                f"placeholder text (\"{stub_phrase}\") instead of a real "
                                "implementation — the required logic appears faked, "
                                "commented out, or replaced with print/pass. Implement "
                                "the ACTUAL functionality the task asks for. If a required "
                                "library isn't installed, still write the real code that "
                                "uses it (QA will mock it for tests)."
                            )
                except Exception:
                    pass

        import sys as _sys
        import importlib.util as _ilu
        stdlib = getattr(_sys, "stdlib_module_names", set())

        # Only ever analyse the workspace's OWN files — never vendored deps.
        # (get_workspace_files already prunes these, but stay defensive in case
        # a stray venv path reached `files` some other way.)
        analyzable = [f for f in py_files if not _is_dependency_path(f)]

        local_modules: set = set()
        for f in analyzable:
            p = Path(f)
            stem = p.stem
            if stem != "__init__":
                local_modules.add(stem)
            for part in p.parts[:-1]:
                local_modules.add(part)

        _resolve_cache: dict = {}

        def _resolvable(top: str) -> bool:
            # stdlib or a local workspace module → fine
            if not top or top in stdlib or top in local_modules:
                return True
            if top in _resolve_cache:
                return _resolve_cache[top]
            # installed third-party package (langchain, requests, …) → fine
            try:
                ok = _ilu.find_spec(top) is not None
            except Exception:
                ok = False
            _resolve_cache[top] = ok
            return ok

        unresolved: list = []   # (path, module) — deduped, capped on output
        for path in analyzable:
            try:
                content = (WORKSPACE_DIR / path).read_text(encoding="utf-8")
                tree = ast.parse(content)
            except SyntaxError as e:
                errors.append(f"SYNTAX ERROR in {path}: {e}")
                continue
            except Exception as e:
                errors.append(f"READ ERROR in {path}: {e}")
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    top = (node.module or "").split(".")[0]
                    if top and not _resolvable(top):
                        pair = (path, top)
                        if pair not in unresolved:
                            unresolved.append(pair)

        # Report unresolvable imports, but CAP the output so a pathological case
        # can never flood the console (and never dump the local-module list).
        for path, top in unresolved[:10]:
            msg = (
                f"CRITICAL: Unresolvable import '{top}' in {path} "
                "(not a standard-library, installed, or local module)."
            )
            related = _suggest_installed(top)
            if related:
                msg += (" Installed packages in the same family you CAN use "
                        f"instead: {', '.join(related)}.")
            else:
                msg += (" This package is NOT installed in the environment — "
                        "use an already-installed library, or a stdlib approach.")
            errors.append(msg)
        if len(unresolved) > 10:
            errors.append(f"...and {len(unresolved) - 10} more unresolvable import(s).")

    if errors:
        error_text = "VALIDATION FAILED:\n" + "\n".join(errors)
        print("  Errors:\n  " + "\n  ".join(errors))
        _log_event("validator", f"FAILED — {len(errors)} error(s)")
        return {"status": "VALIDATION_FAILED", "test_output": error_text}

    print("  Validation passed.")
    _log_event("validator", "Passed")
    return {"status": "PENDING"}


def tester(state: AgentState):
    print("\n[NODE] Tester (Coverage & Execution)")
    if state["language"] == "rust":
        cmd = "cargo tarpaulin --fail-under 80"
    else:
        # Don't let pytest collect or measure coverage inside a vendored venv
        # that a task may have created in the workspace (pytest's default
        # norecursedirs misses non-standard names like 'myenv').
        ignores = ""
        try:
            for d in WORKSPACE_DIR.iterdir():
                if d.is_dir() and (d.name in EXCLUDED_DIRS or (d / "pyvenv.cfg").exists()):
                    ignores += f" --ignore={shlex.quote(d.name)}"
        except Exception:
            pass

        # Scope coverage to SOURCE modules only — measuring '.' lets a ~100%-
        # covered test file inflate the total and carry an under-covered source
        # over the 80% gate. With --cov=<source> the gate reflects real source
        # coverage.
        src_files = [
            f for f in get_workspace_files()
            if f.endswith(".py") and not _is_test_artifact(f) and not _is_dependency_path(f)
        ]
        cov_targets = " ".join(
            f"--cov={shlex.quote(f[:-3].replace(os.sep, '.'))}" for f in src_files
        ) or "--cov=."
        cmd = (f"pytest {cov_targets} --cov-report=term-missing "
               "--cov-fail-under=80 -v" + ignores)
    try:
        result = subprocess.run(
            shlex.split(cmd), cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=45,
        )
        output = (
            f"Exit Code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        status = "TESTS_PASSED" if result.returncode == 0 else "TEST_FAILED"
        if "collected 0 items" in result.stdout or "0 tests run" in result.stdout:
            status = "TEST_FAILED"
            output += "\nERROR: 0 tests collected. Check file naming and test function prefixes."
    except Exception as e:
        output = f"Execution failed: {e}"
        status = "TEST_FAILED"

    print(f"  Test status: {status}")
    _log_event("tester", f"Status: {status}")
    return {"test_output": output, "status": status}


def quality_reviewer(state: AgentState):
    print("\n[NODE] Quality Reviewer (Architecture & Best Practices)")
    
    history = state.get("repair_history", [])
    history_str = "\n".join(f"- {fb}" for fb in history if "Quality/Architecture" in fb)
    
    prompt = (
        f"Task: {state['task']}\n\n"
        "The code has successfully passed all testing. Review it for architecture and best practices.\n\n"
        f"Workspace Files:\n{_format_workspace_contents()}\n\n"
        f"Previous Quality Rejections (if any):\n{history_str or 'None'}\n\n"
        "CRITICAL REVIEW INSTRUCTIONS:\n"
        "1. FUNCTIONAL FULFILLMENT (most important): verify the code ACTUALLY "
        "implements the task. If the task names a library, API, or protocol "
        "(e.g. asyncpg, the COPY protocol, Pydantic v2 strict mode, requests), "
        "the real code must USE it — not in comments, not in 'in a real "
        "implementation we would…' placeholders, not faked with print(). REJECT "
        "(is_approved=False) if the core required functionality is stubbed, "
        "commented out, simulated, or replaced with placeholders.\n"
        "2. BE PRAGMATIC about STYLE: if the task is a simple script, do NOT "
        "reject it for lacking classes, argparse, or 'enterprise' architecture.\n"
        "3. Approve (is_approved=True) only if the code cleanly, safely, and "
        "really solves the requested task.\n"
        "4. Otherwise reject for: stubbed/faked core functionality (per #1), "
        "critical security flaws, unmaintainable spaghetti, or obvious bugs.\n"
        "5. Don't loop on pure style: if the developer addressed your previous "
        "feedback and only minor stylistic nitpicks remain, APPROVE — but a "
        "stubbed implementation is NEVER a minor nitpick."
    )
    
    structured_llm = REVIEWER_LLM.with_structured_output(QualityVerdict, method="json_schema")
    try:
        verdict = structured_llm.invoke(prompt)
        is_approved = verdict.is_approved
        feedback = verdict.feedback
    except Exception:
        # Don't blindly approve on a parse glitch — that's how bad code gets
        # rubber-stamped. Try a plain-text JSON salvage first.
        salvaged = _salvage_verdict(REVIEWER_LLM, prompt, ("is_approved", "feedback"))
        if salvaged is not None and "is_approved" in salvaged:
            val = salvaged.get("is_approved")
            if isinstance(val, str):
                is_approved = val.strip().lower() in ("true", "yes", "approved", "1")
            else:
                is_approved = bool(val)
            feedback = salvaged.get("feedback") or "(no feedback text provided)"
        else:
            # Still unparseable. The step cap is the real backstop against loops,
            # so approve as a last resort but flag it loudly rather than silently.
            is_approved = True
            feedback = (
                "Quality verdict could not be parsed even after a plain-text retry; "
                "approving to avoid an endless loop. MANUAL REVIEW RECOMMENDED."
            )
            print("  [WARN] Quality verdict unparseable after retry — approving as last resort.")
        
    if is_approved:
        print("  -> Quality Approved!")
        _log_event("quality_reviewer", "Quality check passed")
        return {"status": "APPROVED", "quality_feedback": feedback}
    else:
        print("  -> Quality Rejected! Sending back to Developer.")
        _log_event("quality_reviewer", f"Rejected: {feedback[:50]}...")
        
        history = state.get("repair_history", [])
        new_history = history + [f"Quality/Architecture Review Failure: {feedback}"]
        
        return {
            "status": "QUALITY_FAILED", 
            "review_feedback": feedback, 
            "quality_feedback": feedback,
            "repair_target": "developer", 
            "repair_history": new_history
        }


def reviewer(state: AgentState):
    print(f"\n[NODE] Reviewer (Debugging {state['status']})")
    status = state["status"]
    test_out = state.get("test_output", "")

    # Shortcut: if no .py files exist at all, no need to ask the LLM
    files = state.get("files", [])
    py_files = [f for f in files if f.endswith(".py") and "test" not in f.lower()]
    if not py_files and status == "VALIDATION_FAILED":
        target = "developer"
        feedback = (
            "No source .py files were found in the workspace. "
            "The developer must write the main source file using the `write_file` tool. "
            "Reminder: the file must be named exactly as the primary_artifact and "
            "must NOT start with 'test_'."
        )
        print(f"  -> Shortcut: no source files — routing to developer")
        _log_event("reviewer", "Shortcut: no source files → developer")
        _log_review(state.get("iterations", 0), status, test_out, feedback, target)
        history = state.get("repair_history", [])
        return {"review_feedback": feedback, "repair_target": target,
                "repair_history": history + [feedback]}

    # Deterministic shortcut: a SYNTAX ERROR in a *source* file is ALWAYS a
    # developer problem; one in a *test* file is ALWAYS a QA problem. Leaving
    # this to the LLM caused a deadlock — it routed a main.py syntax error to
    # QA, which can't touch frozen source, burning three repair cycles.
    syn = re.search(r"SYNTAX ERROR in (\S+?):", test_out)
    if syn and status == "VALIDATION_FAILED":
        bad_file = os.path.basename(syn.group(1))
        det_target = "qa" if bad_file.startswith("test_") else "developer"
        det_feedback = (
            f"A syntax error was detected in `{bad_file}`. Rewrite the file so it "
            f"parses correctly. Exact pipeline error:\n\n{test_out.strip()}"
        )
        print(f"  -> Deterministic routing (syntax error in {bad_file}) → {det_target.upper()}")
        _log_event("reviewer", f"Deterministic: syntax error in {bad_file} → {det_target}")
        _log_review(state.get("iterations", 0), status, test_out, det_feedback, det_target)
        history = state.get("repair_history", [])
        return {"review_feedback": det_feedback, "repair_target": det_target,
                "repair_history": history + [det_feedback]}

    # Deterministic: a STUB/PLACEHOLDER was detected in the source (the dev
    # faked the hard requirement). That's always a developer fix — re-implement
    # for real. Never let the LLM soften this into a QA test tweak.
    if "STUB DETECTED" in test_out and status == "VALIDATION_FAILED":
        det_feedback = (
            "The source contains placeholder/stubbed logic instead of a real "
            "implementation. Do NOT comment out or fake the required functionality "
            "to make tests pass — implement it for real. Exact issue:\n\n"
            + test_out.strip()
        )
        print("  -> Deterministic routing (stub detected) → DEVELOPER")
        _log_event("reviewer", "Deterministic: stub detected → developer")
        _log_review(state.get("iterations", 0), status, test_out, det_feedback, "developer")
        history = state.get("repair_history", [])
        return {"review_feedback": det_feedback, "repair_target": "developer",
                "repair_history": history + [det_feedback]}

    # Deterministic: an async-mocking error ("MagicMock can't be used in 'await'"
    # or "coroutine was never awaited") is a TEST defect — QA must use AsyncMock.
    # Sending it to the developer caused the gutting-the-source failure mode.
    if status == "TEST_FAILED" and (
        "can't be used in 'await'" in test_out
        or "was never awaited" in test_out
        or "AsyncMock" in test_out
    ):
        det_feedback = (
            "The test failure is an async-mocking defect in the TEST file, not a "
            "code bug — do NOT change the source. The test mocks an async call "
            "with a non-async mock. Fix the test: use `unittest.mock.AsyncMock` "
            "for awaited methods (e.g. the connection / its async methods), and "
            "`await` (or `asyncio.run(...)`) the coroutine under test so its body "
            "actually executes. Exact error:\n\n" + test_out.strip()[:1500]
        )
        print("  -> Deterministic routing (async-mock test defect) → QA")
        _log_event("reviewer", "Deterministic: async-mock defect → qa")
        _log_review(state.get("iterations", 0), status, test_out, det_feedback, "qa")
        history = state.get("repair_history", [])
        return {"review_feedback": det_feedback, "repair_target": "qa",
                "repair_history": history + [det_feedback]}

    # Deterministic: tests PASSED and the ONLY failure is the coverage threshold.
    # That's a test-completeness gap (QA), not a code bug — routing it to the
    # developer just rewrites working code (the dead-loop we observed). Send it
    # to QA with the concrete technique to cover a module whose imports may not
    # be installed.
    if status == "TEST_FAILED":
        cov_only = (
            ("fail-under" in test_out or "Coverage failure" in test_out)
            and re.search(r"\b\d+ passed\b", test_out)
            and not re.search(r"\b\d+ (failed|error)", test_out)
        )
        if cov_only:
            cov_m = re.search(r"total coverage:\s*([\d.]+)%", test_out, re.IGNORECASE)
            cov_pct = (cov_m.group(1) + "%") if cov_m else "below threshold"
            det_feedback = (
                "All tests PASS — the only failure is the COVERAGE threshold "
                f"(currently {cov_pct}). This is a TEST gap, not a code bug, so do "
                "NOT rewrite the source. The tests must IMPORT the source module "
                "and call its functions so the source lines actually execute.\n"
                "If the module imports third-party packages that are NOT installed "
                "(e.g. asyncpg, langchain_*), inject fakes into sys.modules BEFORE "
                "importing it, then import and exercise the real functions with "
                "mocks:\n"
                "    import sys\n"
                "    from unittest.mock import MagicMock, AsyncMock\n"
                "    sys.modules.setdefault('asyncpg', MagicMock())\n"
                "    import main            # now importable → its lines get covered\n"
                "    # then call main.read_config(), main.batch_insert_data(), etc.\n"
                "Do NOT re-implement the module's logic inside the test."
            )
            print("  -> Deterministic routing (coverage-only failure) → QA")
            _log_event("reviewer", "Deterministic: coverage-only failure → qa")
            _log_review(state.get("iterations", 0), status, test_out, det_feedback, "qa")
            history = state.get("repair_history", [])
            return {"review_feedback": det_feedback, "repair_target": "qa",
                    "repair_history": history + [det_feedback]}

    routing_hint = (
        "NOTE: This is a VALIDATION_FAILED (static analysis). "
        "If source files are missing or have syntax errors -> target 'developer'. "
        "If test imports are broken -> target 'qa'."
        if status == "VALIDATION_FAILED"
        else
        "NOTE: This is a TEST_FAILED (runtime). "
        "If the implementation logic is wrong -> target 'developer'. "
        "If tests have wrong assertions or coverage gaps -> target 'qa'."
    )
    prompt = (
        f"{routing_hint}\n\nFailure Type: {status}\n"
        f"System Output:\n{test_out}\n\n"
        f"Workspace Files:\n{_format_workspace_contents()}"
    )

    structured_llm = REVIEWER_LLM.with_structured_output(ReviewVerdict, method="json_schema")
    try:
        verdict = structured_llm.invoke(prompt)
        target, feedback = verdict.repair_target, verdict.review_feedback
    except Exception:
        # Structured parse failed (common on local models). Try a plain-text
        # JSON salvage before giving up.
        salvaged = _salvage_verdict(REVIEWER_LLM, prompt, ("repair_target", "review_feedback"))
        target = (salvaged or {}).get("repair_target", "developer")
        feedback = (salvaged or {}).get("review_feedback") or ""
        if target not in ("developer", "qa"):
            target = "developer"
        if not feedback:
            # Don't hand the developer a meta-message about JSON — give it the
            # actual pipeline error so the repair has something to act on.
            feedback = (
                "The automated reviewer could not produce structured feedback. "
                "Fix the issue reported by the pipeline below:\n\n"
                + (test_out or "(no diagnostic output was captured)")
            )

    print(f"  -> Suggested repair target: {target.upper()}")
    _log_event("reviewer", f"Suggested {target}")
    _log_review(state.get("iterations", 0), status, test_out, feedback, target)

    history = state.get("repair_history", [])
    new_history = history + [feedback]

    return {"review_feedback": feedback, "repair_target": target, "repair_history": new_history}


def save_artifact(state: AgentState):
    print(f"\n[NODE] Save Artifact (Extracting: {state.get('primary_artifact')})")
    if state.get("status") != "APPROVED":
        print("  -> Blocked: status is not APPROVED. Nothing saved.")
        _log_event("save_artifact", "Blocked — status not APPROVED")
        return {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_file = state.get("primary_artifact", "")
    file_list = state.get("files", [])

    if target_file and target_file in file_list:
        src = WORKSPACE_DIR / target_file
        dest = OUTPUT_DIR / Path(target_file).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  -> Saved {target_file} -> {dest}")
        _log_event("save_artifact", f"Saved {target_file}")
    else:
        print(f"  -> '{target_file}' not found. Saving full workspace as fallback.")
        for rel_path in file_list:
            src = WORKSPACE_DIR / rel_path
            dest = OUTPUT_DIR / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        _log_event("save_artifact", "Saved full workspace (fallback)")
    return {}


def supervisor(state: AgentState):
    """Hybrid deterministic router: 95% State Machine + 5% LLM for ambiguity."""
    step = state.get("total_steps", 0) + 1
    print(f"\n[SUPERVISOR] Step {step}")

    if step > MAX_TOTAL_STEPS:
        reasoning = "Step cap reached."
        print(f"  -> end :: {reasoning}")
        _log_supervisor(step, state.get("status", ""), "end", reasoning)
        return {"total_steps": step, "last_supervisor_reasoning": reasoning, "_next": "end"}

    last_node = EXECUTION_LOG[-1]["node"] if EXECUTION_LOG else "planner"
    status = state.get("status", "PENDING")
    all_files = state.get("files", [])
    lang = state.get("language", "python")

    # A source file must exist (not just any file — not test files, not reports)
    if lang == "rust":
        files_exist = any("main.rs" in f or "lib.rs" in f for f in all_files)
    else:
        files_exist = any(
            f.endswith(".py") and "test" not in f.lower()
            for f in all_files
        )

    next_node = "end"
    reasoning = ""

    # ---- Coverage stall detection: break the dead-loop ----
    # If the PRIMARY SOURCE file's own coverage isn't improving across repair
    # cycles, we're stuck (e.g. the source can't be imported/run without absent
    # infrastructure). Stop early instead of burning the whole step cap.
    cov_track: dict = {}
    if status == "TEST_FAILED" and last_node == "tester":
        primary = os.path.basename(state.get("primary_artifact", "") or "")
        cur_src = None
        if primary:
            row = re.search(rf"(?m)^{re.escape(primary)}\s+\d+\s+\d+\s+(\d+)%",
                            state.get("test_output", ""))
            if row:
                cur_src = float(row.group(1))
        if cur_src is not None:
            best = state.get("best_src_coverage", -1.0)
            if cur_src > best + 0.01:
                cov_track = {"best_src_coverage": cur_src, "stall_count": 0}
            else:
                sc = state.get("stall_count", 0) + 1
                cov_track = {"stall_count": sc}
                if sc >= STALL_LIMIT:
                    reasoning = (
                        f"`{primary}` has stayed at {int(best)}% coverage for {sc} "
                        "repair cycles with no improvement — the tests aren't "
                        "exercising the source (it likely can't run without external "
                        "infrastructure). Stopping early rather than looping to the "
                        "step cap."
                    )
                    print(f"  -> end :: {reasoning}")
                    _log_supervisor(step, status, "end", reasoning)
                    return {"total_steps": step, "last_supervisor_reasoning": reasoning,
                            "_next": "end", **cov_track}

    # ====================================================
    # 1. DETERMINISTIC RULES (Fast, Free, 100% Reliable)
    # ====================================================
    if status == "APPROVED":
        next_node = "save_artifact"
        reasoning = "Status is APPROVED. Ready to save."
        
    elif last_node == "save_artifact":
        next_node = "end"
        reasoning = "Artifact saved. Ending pipeline."
        
    elif status in ("VALIDATION_FAILED", "TEST_FAILED"):
        if last_node != "reviewer":
            next_node = "reviewer"
            reasoning = f"Status is {status}. Routing to reviewer for diagnosis."
        else:
            next_node = state.get("repair_target", "developer")
            reasoning = f"Reviewer completed diagnosis. Routing to {next_node} to fix."

    elif status == "QUALITY_FAILED":
        next_node = "developer"
        reasoning = "Quality review failed. Routing back to developer for architectural fixes."
            
    elif last_node == "researcher":
        next_node = "developer"
        reasoning = "Research complete. Moving to development."
        
    elif last_node == "developer":
        if not files_exist:
            # Count how many times developer ran without producing files
            dev_failures = sum(
                1 for e in EXECUTION_LOG
                if e["node"] == "developer" and "Iteration" in e["detail"]
            )
            if dev_failures >= 3:
                next_node = "end"
                reasoning = (
                    f"Developer failed to write source files after {dev_failures} attempts. "
                    "The local LLM appears unable to use tools reliably. Ending pipeline."
                )
            else:
                next_node = "validator"
                reasoning = (
                    "Developer did not write source files. "
                    "Sending to validator to generate a VALIDATION_FAILED error, "
                    "which will trigger the reviewer → developer repair loop."
                )
        else:
            next_node = "qa"
            reasoning = "Development complete. Moving to QA to write tests."
            
    elif last_node == "qa":
        next_node = "validator"
        reasoning = "QA complete. Moving to validation."
        
    elif last_node == "validator":
        # If we reached here, status is not VALIDATION_FAILED (handled above), so it passed
        next_node = "tester"
        reasoning = "Validation passed. Running test suite."

    elif last_node == "tester":
        if status == "TESTS_PASSED":
            next_node = "quality_reviewer"
            reasoning = "Tests passed. Routing to quality reviewer for architecture check."
        else:
            # Fallback for unexpected states, though caught above typically
            next_node = "reviewer"
            reasoning = "Tests failed. Routing to reviewer."

    # ====================================================
    # 2. LLM FALLBACK (Only for ambiguous routing: Planner -> ?)
    # ====================================================
    elif last_node == "planner":
        prompt = (
            f"Task: {state['task']}\n\n"
            "Does this programming task require looking up external documentation "
            "(e.g., specific library versions, APIs, or best practices) before writing code?\n"
            "Reply with EXACTLY one word: 'researcher' if yes, or 'developer' if no."
        )
        try:
            response = _reviewer_text(prompt).strip().lower()
            if "researcher" in response:
                next_node = "researcher"
                reasoning = "LLM determined external research is needed."
            else:
                next_node = "developer"
                reasoning = "LLM determined no research needed; routing straight to developer."
        except Exception as e:
            next_node = "developer"
            reasoning = f"LLM routing failed ({e}); defaulting to developer."
            
    else:
        next_node = "end"
        reasoning = f"Unknown state transition from {last_node}. Forcing end."

    print(f"  -> {next_node} :: {reasoning}")
    _log_supervisor(step, status, next_node, reasoning)
    return {"total_steps": step, "last_supervisor_reasoning": reasoning,
            "_next": next_node, **cov_track}

# ═════════════════════════════════════════════════════════════
# 5b. REPORT GENERATION
# ═════════════════════════════════════════════════════════════
def generate_pipeline_execution_report(final_state: AgentState) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "pipeline_execution_report.md"

    lines = [
        "# Pipeline Execution Report",
        f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Task",
        f"- **Task:** {final_state.get('task', '')}",
        f"- **Detected language:** {final_state.get('language', '')}",
        f"- **Primary artifact:** {final_state.get('primary_artifact', '')}",
        f"- **Developer/QA iterations:** {final_state.get('iterations', 0)}",
        f"- **Total supervisor steps:** {final_state.get('total_steps', 0)}",
        f"- **Final status:** {final_state.get('status', 'UNKNOWN')}",
        "",
        "## Supervisor Decisions",
        "",
        "| Step | Status at decision | Routed to | Reasoning |",
        "|---|---|---|---|",
    ]
    for s in SUPERVISOR_LOG:
        lines.append(f"| {s['step']} | {s['status']} | {s['decision']} | {s['reasoning']} |")

    lines += [
        "",
        "## Node Timeline",
        "",
        "| Time | Node | Detail |",
        "|---|---|---|",
    ]
    for ev in EXECUTION_LOG:
        lines.append(f"| {ev['time']} | {ev['node']} | {ev['detail']} |")

    lines += [
        "",
        "## Outcome",
        (
            f"Pipeline reached **APPROVED** status and the artifact was saved to "
            f"`{OUTPUT_DIR.absolute()}`."
            if final_state.get("status") == "APPROVED"
            else
            f"Pipeline halted with status **{final_state.get('status', 'UNKNOWN')}** "
            f"after {final_state.get('total_steps', 0)} supervisor step(s). "
            "No artifact was approved."
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _expand_line_ranges(spec: str) -> list:
    """'45-46, 50, 60-62' -> [45,46,50,60,61,62]. Ignores branch arrows."""
    out: list = []
    for part in spec.split(","):
        part = part.strip()
        if not part or "->" in part:   # skip branch-coverage entries
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


def _coverage_section(test_output: str) -> list:
    """Build a human-readable Coverage Analysis section.

    - Returns a single 'full coverage' line when total == 100%.
    - When below 100%, lists the uncovered lines per file with their source so
      the reader sees exactly what isn't tested.
    - Returns [] when there's no coverage data to talk about (e.g. Rust, or no
      run), so the report just omits the section.
    """
    total = None
    m = re.search(r"Total coverage:\s*([\d.]+)%", test_output)
    if m:
        total = float(m.group(1))
    else:
        m2 = re.search(r"^TOTAL\s+\d+\s+\d+\s+([\d.]+)%", test_output, re.MULTILINE)
        if m2:
            total = float(m2.group(1))
    if total is None:
        return []

    if total >= 100.0:
        return ["## Coverage Analysis", "",
                "Full line coverage (100%) — every statement is exercised by the "
                "test suite. No gaps to report.", ""]

    # term-missing rows:  name.py   Stmts   Miss   Cover%   Missing
    row_re = re.compile(
        r"^(\S+\.py)\s+\d+\s+(\d+)\s+\d+%\s+([\d,\-\s>]+?)\s*$", re.MULTILINE
    )
    source_rows, test_rows = [], []
    for fname, miss, missing in row_re.findall(test_output):
        if int(miss) == 0:
            continue
        bucket = test_rows if "test" in os.path.basename(fname).lower() else source_rows
        bucket.append((fname, int(miss), missing.strip()))

    section = ["## Coverage Analysis", ""]

    # When every gap is inside test files, source is effectively fully covered —
    # say so plainly instead of flagging a test file's own pytest.main() guard.
    if not source_rows:
        section += [
            f"Overall line coverage is **{total:.2f}%**. All **source** files have "
            "full line coverage — the only uncovered lines live inside the test "
            "files themselves (e.g. a `pytest.main()` / `__main__` guard), which "
            "don't need testing. No action needed.", "",
        ]
        return section

    section += [
        f"Overall line coverage is **{total:.2f}%**, below 100%. The **source** "
        "statements below were never executed during testing:", "",
    ]
    for fname, miss, missing in source_rows:
        section.append(f"**`{fname}`** — {miss} uncovered line(s): `{missing}`")
        section.append("")
        line_nums = _expand_line_ranges(missing)
        try:
            src = (WORKSPACE_DIR / fname).read_text(encoding="utf-8").splitlines()
            shown = ["```python"]
            for ln in line_nums:
                if 1 <= ln <= len(src):
                    shown.append(f"{ln:>4}  {src[ln - 1]}")
            shown.append("```")
            if len(shown) > 2:
                section += shown
                section.append("")
        except Exception:
            pass

    if test_rows:
        section += [
            "_(Test files also show minor gaps — their own `pytest.main()` / "
            '`if __name__ == "__main__"` lines — which are immaterial and excluded '
            "from the analysis above.)_", "",
        ]

    section += [
        "_Untested source lines are commonly error-handling branches, the "
        '`if __name__ == "__main__"` guard, or unreached edge cases. If any '
        "represent real behaviour you care about, add a targeted test to cover "
        "them._", "",
    ]
    return section


def generate_code_review_report(final_state: AgentState) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "code_review_report.md"

    lines = [
        "# Code Review Report",
        f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"Task: {final_state.get('task', '')}",
        f"Language: {final_state.get('language', '')}",
        "",
    ]

    if final_state.get("research_notes"):
        lines += ["## Research Notes", "", final_state["research_notes"].strip(), ""]

    if not REVIEW_LOG:
        lines.append("No review cycles were triggered — the pipeline passed validation "
                      "and tests without needing a repair loop.")
    else:
        for i, entry in enumerate(REVIEW_LOG, start=1):
            lines += [
                f"## Review Cycle {i} (after iteration {entry['iteration']})",
                f"- **Failure type:** {entry['status']}",
                f"- **Routed to:** {entry['target']}",
                "",
                "**System output at time of failure:**",
                "```",
                entry["output"].strip(),
                "```",
                "",
                "**Reviewer feedback:**",
                "",
                entry["feedback"].strip(),
                "",
            ]

    lines += ["## Final Test Coverage & Execution Output", "```text", final_state.get("test_output", "(No tests were run)"), "```", ""]

    # Explain coverage gaps when below 100% (one line when fully covered).
    lines += _coverage_section(final_state.get("test_output", ""))

    lines += ["## Final Files in Workspace", ""]
    files = final_state.get("files", [])
    if files:
        for fname in sorted(files):
            lines.append(f"- `{fname}`")
    else:
        lines.append("(no files in workspace)")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_quality_report(final_state: AgentState) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "quality_review_report.md"
    
    lines = [
        "# Quality & Architecture Review",
        f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"**Final Decision:** {'APPROVED' if final_state.get('status') == 'APPROVED' else 'REJECTED'}",
        "",
        "## Senior Staff Engineer Feedback",
        final_state.get("quality_feedback", "(No quality review completed)"),
        ""
    ]
    
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

# ═════════════════════════════════════════════════════════════
# 6. GRAPH CONSTRUCTION
# ═════════════════════════════════════════════════════════════
def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("developer", developer)
    workflow.add_node("qa", qa)
    workflow.add_node("validator", validator)
    workflow.add_node("tester", tester)
    workflow.add_node("quality_reviewer", quality_reviewer)
    workflow.add_node("reviewer", reviewer)
    workflow.add_node("save_artifact", save_artifact)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "supervisor")

    # Every worker reports back to the supervisor, which decides what's next.
    for worker in ("researcher", "developer", "qa", "validator", "tester", "quality_reviewer", "reviewer"):
        workflow.add_edge(worker, "supervisor")

    def route_from_supervisor(state: AgentState):
        decision = state.get("_next", "end")
        if decision == "end":
            return END
        if decision == "save_artifact":
            return "save_artifact"
        return decision
    workflow.add_conditional_edges("supervisor", route_from_supervisor)

    # save_artifact always ends the run (whether it actually saved or was blocked).
    workflow.add_edge("save_artifact", END)

    return workflow.compile()

# ═════════════════════════════════════════════════════════════
# BATCH INPUT: read prompts from input.txt, one result folder each
# ═════════════════════════════════════════════════════════════
def _read_prompts(path: Path) -> List[str]:
    """Parse a prompt file into an ordered list of task strings.

    Accepts numbered lines like '1)task...', '2. task...', '3 - task...'.
    A non-numbered line is treated as a continuation of the previous prompt,
    so a wrapped prompt still parses as one task. Blank lines are ignored.
    """
    prompts: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^\d+\s*[\)\.\:\-]\s*(.+)$", s)
        if m:
            prompts.append(m.group(1).strip())
        elif prompts:
            prompts[-1] += " " + s          # continuation of the previous prompt
        else:
            prompts.append(s)               # unnumbered first line
    return [p for p in prompts if p]


_TASK_STOPWORDS = {
    "write", "a", "an", "the", "python", "script", "that", "code", "program",
    "to", "and", "from", "using", "use", "with", "of", "for", "in", "is", "it",
    "please", "without", "any", "library", "libraries", "file", "create", "build",
    "make", "your", "this", "which", "reads", "read", "writes", "write", "save",
    "saves", "current", "data", "new",
}


def _slugify_task(task: str, fallback: str = "task") -> str:
    """Short, filesystem-safe folder name summarising a task (e.g. 'bitcoin').

    Tries the reviewer model for a 1-2 word name, then falls back to the first
    few significant keywords from the task. Always returns something usable.
    """
    name = ""
    try:
        name = _reviewer_text(
            "Give a SHORT folder name (1 or 2 words, lowercase snake_case, no "
            "spaces or punctuation) that summarises this coding task. Output ONLY "
            f"the name, nothing else.\n\nTask: {task}"
        ).strip()
    except Exception:
        name = ""
    name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    name = "_".join([t for t in name.split("_") if t][:3])[:40]
    if not name:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", task.lower())
        keep = [w for w in words if w not in _TASK_STOPWORDS][:3]
        name = "_".join(keep)[:40]
    return name or fallback


def _fresh_initial_state(task: str) -> "AgentState":
    return {
        "task": task, "language": "python", "primary_artifact": "", "plan": "",
        "files": [], "test_output": "", "review_feedback": "", "quality_feedback": "",
        "repair_history": [], "repair_target": "developer", "status": "PENDING",
        "iterations": 0, "total_steps": 0, "research_notes": "",
        "last_supervisor_reasoning": "", "best_src_coverage": -1.0,
        "stall_count": 0, "_next": "",
    }


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n🚀 Multi-Agent Coding Pipeline v11 (Dynamic Supervisor)\n{'='*60}")

    # Prompts come from a file (default ./input.txt, or pass a path as arg 1).
    # Each line like "1)..." is one task; results go to final_output/<slug>/.
    prompt_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input.txt")

    if prompt_file.exists():
        prompts = _read_prompts(prompt_file)
        if not prompts:
            print(f"No prompts found in {prompt_file.absolute()}. Exiting.")
            raise SystemExit(1)
        print(f"\nLoaded {len(prompts)} prompt(s) from {prompt_file.absolute()}")
    else:
        # Backward-compatible fallback: no input.txt → ask interactively.
        print(f"\n(No '{prompt_file}' found — falling back to interactive input.)")
        one = input("\nEnter the programming task you want the AI agents to solve:\n> ").strip()
        if not one:
            print("Task cannot be empty. Exiting.")
            raise SystemExit(1)
        prompts = [one]

    app = build_workflow()
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    used_slugs: set = set()
    run_summary: list = []
    interrupted = False

    for idx, task in enumerate(prompts, 1):
        # Compute the slug/folder OUTSIDE the main try so a failure still has a
        # place to record against; _slugify_task is itself network-guarded.
        try:
            slug = _slugify_task(task)
        except Exception:
            slug = f"task_{idx}"
        base_slug, n = slug, 2
        while slug in used_slugs:            # guarantee a unique folder per prompt
            slug = f"{base_slug}_{n}"; n += 1
        used_slugs.add(slug)

        task_dir = BASE_OUTPUT_DIR / slug
        _set_task_output_dirs(task_dir)

        print("\n" + "█" * 60)
        print(f"PROMPT {idx}/{len(prompts)}  →  folder: {slug}/")
        print(f"  {task}")
        print("█" * 60)

        status = "ERROR"
        final_state = {**_fresh_initial_state(task), "status": "ERROR"}
        try:
            final_state = app.invoke(_fresh_initial_state(task),
                                     config={"recursion_limit": 100})
            status = final_state.get("status", "UNKNOWN")
        except KeyboardInterrupt:
            # A deliberate Ctrl+C stops the whole batch — but gracefully: record
            # this prompt, still write its reports and the master summary below.
            print(f"\n[INTERRUPTED] Ctrl+C during prompt {idx}. Stopping the batch.")
            status = "INTERRUPTED"
            final_state = {**_fresh_initial_state(task), "status": "INTERRUPTED"}
            interrupted = True
        except BaseException as e:
            # ANY other failure (network drop/timeout, Ollama segfault→500, langgraph
            # interrupt, etc.) is contained to THIS prompt so the batch continues.
            print(f"\n[ERROR] Prompt {idx} crashed: {type(e).__name__}: {e}")
            status = "ERROR"
            final_state = {**_fresh_initial_state(task), "status": "ERROR"}

        # Reports always land in this prompt's own folder (best-effort).
        try:
            generate_pipeline_execution_report(final_state)
            generate_code_review_report(final_state)
            generate_quality_report(final_state)
        except Exception as e:
            print(f"[WARN] Report generation failed for prompt {idx}: {e}")

        run_summary.append((idx, slug, status, task_dir))

        print("\n" + "=" * 50)
        print(f"PROMPT {idx} TERMINATED — Status: {status}")
        if status == "APPROVED":
            print(f"  Artifact + reports: {task_dir.absolute()}")
        else:
            print(f"  No artifact saved. Reports: {(task_dir / 'reports').absolute()}")
        print("=" * 50)

        if interrupted:
            print("\n[INTERRUPTED] Skipping remaining prompts at user request.")
            break

    # ── Master summary across all prompts ──
    print("\n" + "█" * 60)
    print("BATCH COMPLETE — SUMMARY")
    print("█" * 60)
    for idx, slug, status, task_dir in run_summary:
        mark = "✅" if status == "APPROVED" else "❌"
        print(f"  {mark}  Prompt {idx:>2}  [{status:<16}]  → {task_dir}/")
    approved = sum(1 for *_, s, _ in run_summary if s == "APPROVED")
    print(f"\n  {approved}/{len(run_summary)} prompt(s) approved. "
          f"All results under: {BASE_OUTPUT_DIR.absolute()}/")
    print("█" * 60 + "\n")
