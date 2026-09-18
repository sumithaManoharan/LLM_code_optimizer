"""
runtime_tool.py

A LangChain tool for the Evaluation Agent (LLM 2). Measures and compares the
execution time of original_code vs optimized_code across the same shape of
test cases used by correctness_tool.

Design notes:
- Same subprocess-isolation pattern as correctness_tool.py, but timing
  happens *inside* a single subprocess across `runs` repeated calls, so
  process-spawn overhead doesn't pollute the measurement.
- Reports min/avg/median time per test case. `min` is the number to trust
  most for comparison — system noise only ever adds time, never removes it.
- Only call this after correctness_tool has confirmed the optimization is
  correct. Timing broken code is meaningless — enforce that in your agent's
  system prompt, not in this tool.
"""

import multiprocessing
import statistics
import time
import traceback
from typing import Any, Dict, List

from langchain_core.tools import tool

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RUNS = 5


def _summarize_value(value: Any, max_len: int = 120) -> Any:
    """
    Returns a compact, LLM-friendly stand-in for a value that might be huge
    (e.g. a million-element list). Small values pass through unchanged;
    large ones get reported as a type + size summary instead of being
    dumped in full. This only affects what's reported back in `details` —
    the *actual* args used to call the function are never touched.
    """
    if isinstance(value, (list, tuple)) and len(value) > 20:
        return {
            "_summary": True,
            "type": type(value).__name__,
            "length": len(value),
            "first_items": value[:5],
        }
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"... [{len(value)} chars total]"
    return value


def _summarize_case(args: list, kwargs: dict) -> Dict[str, Any]:
    return {
        "args": [_summarize_value(a) for a in args],
        "kwargs": {k: _summarize_value(v) for k, v in kwargs.items()},
    }


def _run_function_timed_in_process(
    code: str,
    function_name: str,
    args: list,
    kwargs: dict,
    runs: int,
    result_queue: "multiprocessing.Queue",
) -> None:
    """
    Defines `function_name` from `code`, then calls it `runs` times back to
    back, timing each call. Runs inside a child process.
    """
    try:
        namespace: Dict[str, Any] = {}
        exec(code, namespace)

        if function_name not in namespace or not callable(namespace[function_name]):
            result_queue.put({
                "status": "error",
                "error": f"Function '{function_name}' not found (or not callable) in code.",
            })
            return

        func = namespace[function_name]

        # One untimed warmup call, so first-call effects don't skew the first timed run.
        func(*args, **kwargs)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)

        result_queue.put({
            "status": "success",
            "times_seconds": times,
            "min_seconds": min(times),
            "avg_seconds": statistics.mean(times),
            "median_seconds": statistics.median(times),
        })

    except Exception:
        result_queue.put({"status": "error", "error": traceback.format_exc()})


def _time_with_timeout(
    code: str,
    function_name: str,
    args: list,
    kwargs: dict,
    runs: int = DEFAULT_RUNS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Runs the timed subprocess call with a hard timeout — same pattern as correctness_tool."""
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_function_timed_in_process,
        args=(code, function_name, args, kwargs, runs, result_queue),
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {"status": "timeout", "error": f"Execution exceeded {timeout}s"}

    if not result_queue.empty():
        return result_queue.get()

    return {"status": "error", "error": "Process terminated unexpectedly with no result."}


@tool
def runtime_tool(
    original_code: str,
    optimized_code: str,
    function_name: str,
    test_cases: List[Dict[str, Any]],
    runs: int = DEFAULT_RUNS,
) -> Dict[str, Any]:
    """
    Measures and compares execution time of original_code vs optimized_code.

    For each test case, both versions of `function_name` are called `runs`
    times each (in isolated subprocesses) and timed. Only call this AFTER
    correctness_tool has confirmed the optimization is correct — timing
    incorrect code is meaningless.

    Args:
        original_code: Full source of the original code. Must define a
            top-level function named `function_name`.
        optimized_code: Full source of the optimized code. Must define a
            top-level function with the same name.
        function_name: Name of the function to call in both versions.
        test_cases: List of inputs to time, shaped like
            {"args": [...], "kwargs": {...}}. Prefer inputs large or complex
            enough to produce a measurable runtime — trivial inputs (empty
            lists, single elements) tend to show near-zero, noisy timings.
        runs: How many times to call the function per test case, for timing
            stability (default 5). Higher is more stable but slower to run.

    Returns:
        A dict with:
          - total (int): number of test cases timed
          - overall_avg_speedup (float | None): mean speedup ratio across
            test cases where both versions succeeded (>1 = optimized is
            faster, <1 = optimized is slower, None if nothing succeeded)
          - details (list): per-test-case timing for both versions plus
            each one's speedup_ratio
    """
    details = []
    speedups = []

    for i, case in enumerate(test_cases):
        args = case.get("args", [])
        kwargs = case.get("kwargs", {})

        orig_result = _time_with_timeout(original_code, function_name, args, kwargs, runs=runs)
        opt_result = _time_with_timeout(optimized_code, function_name, args, kwargs, runs=runs)

        speedup_ratio = None
        if orig_result["status"] == "success" and opt_result["status"] == "success":
            orig_time = orig_result["min_seconds"]
            opt_time = opt_result["min_seconds"]
            if opt_time > 0:
                speedup_ratio = orig_time / opt_time
                speedups.append(speedup_ratio)

        details.append({
            "test_case_index": i,
            **_summarize_case(args, kwargs),
            "original_result": orig_result,
            "optimized_result": opt_result,
            "speedup_ratio": speedup_ratio,
        })

    return {
        "total": len(test_cases),
        "overall_avg_speedup": statistics.mean(speedups) if speedups else None,
        "details": details,
    }


if __name__ == "__main__":
    # Quick manual test, no agent required. Uses a large input so the
    # difference between a Python loop and sum() is actually measurable.
    original = """
def add_list(nums):
    total = 0
    for n in nums:
        total = total + n
    return total
"""
    optimized = """
def add_list(nums):
    return sum(nums)
"""
    test_cases = [
        {"args": [list(range(1_000_000))], "kwargs": {}},
    ]

    result = runtime_tool.func(
        original_code=original,
        optimized_code=optimized,
        function_name="add_list",
        test_cases=test_cases,
        runs=5,
    )
    import json
    print(json.dumps(result, indent=2, default=str))