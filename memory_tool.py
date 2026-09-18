"""
memory_tool.py

A LangChain tool for the Evaluation Agent (LLM 2). Measures and compares
peak memory usage of original_code vs optimized_code across test cases.

Design notes:
- Uses tracemalloc (Python stdlib, no extra dependency) to measure peak
  memory allocated by a single function call. Unlike timing, memory is
  fairly deterministic for a pure function, so this measures once per
  test case rather than averaging across repeated runs.
- Same subprocess-isolation pattern as correctness_tool/runtime_tool.
- One untracked warmup call happens before tracemalloc starts, so a
  function's first-call side effects (e.g. a lazy import inside the
  function) don't get counted as part of its "real" memory footprint.
"""

import multiprocessing
import traceback
import tracemalloc
from typing import Any, Dict, List

from langchain_core.tools import tool

DEFAULT_TIMEOUT_SECONDS = 10.0


def _summarize_value(value: Any, max_len: int = 120) -> Any:
    """Compact stand-in for a value that might be huge — same helper as the other tools."""
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


def _run_function_memory_in_process(
    code: str,
    function_name: str,
    args: list,
    kwargs: dict,
    result_queue: "multiprocessing.Queue",
) -> None:
    """
    Defines `function_name` from `code`, calls it once (untracked) to warm
    up, then calls it again inside tracemalloc to measure peak memory.
    Runs inside a child process.
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

        # Untracked warmup call.
        func(*args, **kwargs)

        tracemalloc.start()
        func(*args, **kwargs)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result_queue.put({
            "status": "success",
            "peak_bytes": peak_bytes,
            "peak_kb": round(peak_bytes / 1024, 2),
            "current_bytes": current_bytes,
        })

    except Exception:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        result_queue.put({"status": "error", "error": traceback.format_exc()})


def _measure_memory_with_timeout(
    code: str,
    function_name: str,
    args: list,
    kwargs: dict,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Runs the memory-measuring subprocess call with a hard timeout."""
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_function_memory_in_process,
        args=(code, function_name, args, kwargs, result_queue),
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
def memory_tool(
    original_code: str,
    optimized_code: str,
    function_name: str,
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Measures and compares peak memory usage of original_code vs
    optimized_code.

    For each test case, both versions of `function_name` are called once
    (in isolated subprocesses) and their peak memory allocation is measured
    with tracemalloc. Only call this AFTER correctness_tool has confirmed
    the optimization is correct — measuring broken code is meaningless.

    Args:
        original_code: Full source of the original code. Must define a
            top-level function named `function_name`.
        optimized_code: Full source of the optimized code. Must define a
            top-level function with the same name.
        function_name: Name of the function to call in both versions.
        test_cases: List of inputs to measure, shaped like
            {"args": [...], "kwargs": {...}}. Prefer inputs large or
            complex enough to produce measurable memory use — trivial
            inputs may show near-zero allocation for both versions.

    Returns:
        A dict with:
          - total (int): number of test cases measured
          - overall_avg_reduction_ratio (float | None): mean of
            original_peak / optimized_peak across test cases where both
            versions succeeded (>1 = optimized uses less memory, <1 =
            optimized uses more, None if nothing succeeded)
          - details (list): per-test-case peak memory for both versions
            plus each one's reduction_ratio
    """
    print(f"\n[memory_tool] CALLED — function_name={function_name!r}, "
          f"{len(test_cases)} test case(s)", flush=True)

    details = []
    ratios = []

    for i, case in enumerate(test_cases):
        args = case.get("args", [])
        kwargs = case.get("kwargs", {})

        orig_result = _measure_memory_with_timeout(original_code, function_name, args, kwargs)
        opt_result = _measure_memory_with_timeout(optimized_code, function_name, args, kwargs)

        reduction_ratio = None
        if orig_result["status"] == "success" and opt_result["status"] == "success":
            orig_peak = orig_result["peak_bytes"]
            opt_peak = opt_result["peak_bytes"]
            if opt_peak > 0:
                reduction_ratio = orig_peak / opt_peak
                ratios.append(reduction_ratio)

        details.append({
            "test_case_index": i,
            **_summarize_case(args, kwargs),
            "original_result": orig_result,
            "optimized_result": opt_result,
            "reduction_ratio": reduction_ratio,
        })

    return {
        "total": len(test_cases),
        "overall_avg_reduction_ratio": sum(ratios) / len(ratios) if ratios else None,
        "details": details,
    }


if __name__ == "__main__":
    # Quick manual test, no agent required. Big list comprehension vs
    # generator-based sum, so there's an actual memory difference to see.
    original = """
def add_list(nums):
    doubled = [n * 2 for n in nums]
    return sum(doubled)
"""
    optimized = """
def add_list(nums):
    return sum(n * 2 for n in nums)
"""
    test_cases = [
        {"args": [list(range(1_000_000))], "kwargs": {}},
    ]

    result = memory_tool.func(
        original_code=original,
        optimized_code=optimized,
        function_name="add_list",
        test_cases=test_cases,
    )
    import json
    print(json.dumps(result, indent=2, default=str))