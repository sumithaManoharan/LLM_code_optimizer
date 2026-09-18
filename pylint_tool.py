"""
pylint_tool.py

A LangChain tool for the Evaluation Agent (LLM 2). Runs pylint against
original_code and optimized_code and compares their scores/issues, so the
agent can report whether the optimization also affected code quality/style
— not just correctness or performance.

Design notes:
- pylint only STATICALLY analyzes code (parses it, never executes it), so
  this is inherently safer than correctness_tool/runtime_tool/memory_tool.
  Still run in an isolated subprocess anyway, for consistency and to keep
  pylint's internal state (registered plugins, sys.path changes) from
  leaking across repeated calls within the same agent process.
- Uses pylint's Python API (not just the CLI) so both the numeric score
  and the structured issue list come from a single run, rather than
  shelling out twice.
- No function_name or test_cases needed — the full source of each version
  is linted directly, not executed.
"""

import json
import multiprocessing
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import tool

DEFAULT_TIMEOUT_SECONDS = 15.0


def _run_pylint_in_process(code: str, result_queue: "multiprocessing.Queue") -> None:
    """Writes `code` to a temp file and lints it with pylint's Python API. Runs in a child process."""
    try:
        import io
        from pylint.lint import Run
        from pylint.reporters.json_reporter import JSONReporter

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            output = io.StringIO()
            reporter = JSONReporter(output)
            run = Run([temp_path], reporter=reporter, exit=False)

            score = run.linter.stats.global_note
            raw_issues = json.loads(output.getvalue() or "[]")

            issues = [
                {
                    "type": issue.get("type"),
                    "symbol": issue.get("symbol"),
                    "message": issue.get("message"),
                    "line": issue.get("line"),
                }
                for issue in raw_issues
            ]

            result_queue.put({
                "status": "success",
                "score": round(score, 2) if score is not None else None,
                "issue_count": len(issues),
                "issues": issues,
            })
        finally:
            Path(temp_path).unlink(missing_ok=True)

    except Exception:
        result_queue.put({"status": "error", "error": traceback.format_exc()})


def _lint_with_timeout(code: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Runs pylint in an isolated subprocess with a hard timeout."""
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_run_pylint_in_process, args=(code, result_queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {"status": "timeout", "error": f"Pylint exceeded {timeout}s"}

    if not result_queue.empty():
        return result_queue.get()

    return {"status": "error", "error": "Process terminated unexpectedly with no result."}


@tool
def pylint_tool(original_code: str, optimized_code: str) -> Dict[str, Any]:
    """
    Lints original_code and optimized_code with pylint and compares results.

    Unlike correctness_tool/runtime_tool/memory_tool, this does not execute
    either version — pylint only performs static analysis. No function_name
    or test_cases needed; the full source of each version is linted as-is.

    Args:
        original_code: Full source of the original code.
        optimized_code: Full source of the optimized code.

    Returns:
        A dict with:
          - original: {"score": float 0-10 or None, "issue_count": int,
            "issues": [{"type", "symbol", "message", "line"}, ...]}
          - optimized: same shape, for the optimized version
          - score_delta: optimized_score - original_score (positive =
            quality improved, negative = quality got worse, None if
            either run failed)
    """
    print("\n[pylint_tool] CALLED", flush=True)

    orig_result = _lint_with_timeout(original_code)
    opt_result = _lint_with_timeout(optimized_code)

    score_delta = None
    if (
        orig_result.get("status") == "success"
        and opt_result.get("status") == "success"
        and orig_result.get("score") is not None
        and opt_result.get("score") is not None
    ):
        score_delta = round(opt_result["score"] - orig_result["score"], 2)

    return {
        "original": orig_result,
        "optimized": opt_result,
        "score_delta": score_delta,
    }


if __name__ == "__main__":
    original = """import os
def add_list(nums):
    l = 0
    for i in range(len(nums)):
        l = l + nums[i]
    return l
"""
    optimized = '''"""Sum a list of numbers."""


def add_list(nums):
    """Return the sum of nums."""
    return sum(nums)
'''

    result = pylint_tool.func(original_code=original, optimized_code=optimized)
    print(json.dumps(result, indent=2, default=str))