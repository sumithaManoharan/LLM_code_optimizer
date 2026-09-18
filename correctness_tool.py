import math
import multiprocessing, traceback
from typing import Any, Dict, List
from langchain_core.tools import tool

DEFAULT_TIMEOUT = 10.0


def _values_close(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """
    Compares two values, treating tiny floating-point discrepancies as
    equal instead of failing on them. This matters because a numerically
    equivalent computation can legitimately produce a different bit
    pattern (e.g. summing in a different order), so exact `==` on floats
    produces false failures that don't reflect an actual bug.

    - Numbers (int/float, not bool): compared with math.isclose. A true
      logic bug (wrong by a real amount, e.g. 0.0 vs 1.0) still fails —
      only negligible differences (e.g. 0.6 vs 0.6000000000000001) pass.
    - Lists/tuples: compared element-wise, recursively, so a float buried
      inside a larger structure still gets tolerance treatment.
    - Dicts: compared key-by-key, recursively.
    - Everything else (str, bool, None, ...): exact `==`, since tolerance
      doesn't make sense for non-numeric types.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_close(x, y, rel_tol, abs_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_values_close(a[k], b[k], rel_tol, abs_tol) for k in a)
    return a == b


def _summarize_value(value: Any, max_len: int = 120) -> Any:
    """
    Returns a compact, LLM-friendly stand-in for a value that might be huge
    (e.g. a million-element list). Small values pass through unchanged;
    large ones get reported as a type + size summary instead of being
    dumped in full. Only affects what's reported back — the actual args
    used to call the function, and the actual comparison logic, are never
    touched by this.
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


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a copy of an execution result with a large 'output' summarized, if present."""
    if "output" not in result:
        return result
    return {**result, "output": _summarize_value(result["output"])}

def _run_function_in_process(code: str,function_name: str,args: list,kwargs: dict, result_queue: "multiprocessing.Queue"):
    try:
        namespace={}
        exec(code,namespace)

        if function_name not in namespace or not callable(namespace[function_name]):
            result_queue.put({
                "status": "error",
                "error": f"Function '{function_name}' not found (or not callable) in code.",
            })
            return
        func = namespace[function_name]
        output = func(*args,**kwargs)
        result_queue.put({"status": "success", "output": output})

    except Exception:
        result_queue.put({"status": "error", "error": traceback.format_exc()})

def _execute_with_timeout(code: str,function_name: str,args: list,kwargs: dict,timeout: float = DEFAULT_TIMEOUT):
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_run_function_in_process,args=(code, function_name, args, kwargs, result_queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {"status": "timeout", "error": f"Execution exceeded {timeout}s"}

    if not result_queue.empty():
        return result_queue.get()
    return {"status": "error", "error": "Process terminated unexpectedly with no result."}

def _exception_type(error_text: str) -> str:
    if not error_text:
        return ""
    last_line = error_text.strip().splitlines()[-1]
    return last_line.split(":")[0].strip()

@tool
def correctness_tool(
    original_code: str,
    optimized_code: str,
    function_name: str,
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Checks whether optimized_code behaves the same as original_code.

    For each test case, both versions of `function_name` are executed
    (each in an isolated subprocess with a timeout) using the same inputs,
    and their results are compared. A test case counts as passing if both
    versions return equal output, or both raise the same exception type.

    Args:
        original_code: Full source of the original code. Must define a
            top-level function named `function_name`.
        optimized_code: Full source of the LLM-optimized code. Must define
            a top-level function with the same name.
        function_name: Name of the function to call in both versions.
        test_cases: List of test cases to run, each shaped like
            {"args": [1, 2], "kwargs": {}} describing how to call the
            function. Generate a diverse set covering typical inputs,
            edge cases (empty input, zero, negative numbers), and at
            least one case likely to trigger an error path if relevant.

    Returns:
        A dict with:
          - passed (bool): True only if every test case matched
          - total (int): number of test cases run
          - passed_count (int)
          - failed_count (int)
          - details (list): per-test-case breakdown, useful for the agent
            to explain *why* something failed
    """
    details = []
    passed_count = 0

    for i, case in enumerate(test_cases):
        args = case.get("args", [])
        kwargs = case.get("kwargs", {})

        orig_result = _execute_with_timeout(original_code, function_name, args, kwargs)
        opt_result = _execute_with_timeout(optimized_code, function_name, args, kwargs)

        case_passed = False
        reason = ""

        if orig_result["status"] == "success" and opt_result["status"] == "success":
            if orig_result["output"] == opt_result["output"]:
                case_passed = True
            elif _values_close(orig_result["output"], opt_result["output"]):
                case_passed = True
                reason = (
                    f"Passed within floating-point tolerance (exact values "
                    f"differed negligibly): original={orig_result['output']!r}, "
                    f"optimized={opt_result['output']!r}"
                )
            else:
                reason = (
                    f"Output mismatch: original={orig_result['output']!r}, "
                    f"optimized={opt_result['output']!r}"
                )

        elif orig_result["status"] == "error" and opt_result["status"] == "error":
            orig_exc = _exception_type(orig_result["error"])
            opt_exc = _exception_type(opt_result["error"])
            if orig_exc == opt_exc:
                case_passed = True
            else:
                reason = f"Different exceptions: original={orig_exc}, optimized={opt_exc}"

        elif orig_result["status"] == "timeout" or opt_result["status"] == "timeout":
            reason = "Timed out on original or optimized run."

        else:
            reason = (
                f"Original status={orig_result['status']!r}, "
                f"Optimized status={opt_result['status']!r}"
            )

        if case_passed:
            passed_count += 1

        details.append({
            "test_case_index": i,
            **_summarize_case(args, kwargs),
            "passed": case_passed,
            "original_result": _summarize_result(orig_result),
            "optimized_result": _summarize_result(opt_result),
            "reason": reason,
        })

    total = len(test_cases)
    return {
        "passed": total > 0 and passed_count == total,
        "total": total,
        "passed_count": passed_count,
        "failed_count": total - passed_count,
        "details": details,
    }


if __name__ == "__main__":
    # Quick manual test, no agent required.
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
        {"args": [[1, 2, 3]], "kwargs": {}},
        {"args": [[]], "kwargs": {}},
        {"args": [[-5, 5, 10]], "kwargs": {}},
    ]

    result = correctness_tool.func(
        original_code=original,
        optimized_code=optimized,
        function_name="add_list",
        test_cases=test_cases,
    )
    import json
    print(json.dumps(result, indent=2, default=str))