"""
complexity_tool.py

A LangChain tool for the Evaluation Agent (LLM 2). Compares cyclomatic
complexity and maintainability index between original_code and
optimized_code using radon.

Design notes:
- Like pylint_tool, this only STATICALLY analyzes code — radon parses the
  AST, it never executes anything. Safe to run directly, no subprocess
  isolation needed (there's no arbitrary code execution to isolate).
- Cyclomatic complexity: counts independent paths through the function
  (more branching/loops = higher number = harder to test and reason
  about). radon also assigns a letter rank (A best, F worst).
- Maintainability Index: a 0-100 composite score radon derives from
  complexity, volume, and lines of code (higher = more maintainable).
- No test_cases needed — this analyzes structure, not runtime behavior.
"""

from typing import Any, Dict

from langchain_core.tools import tool
from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit


def _analyze(code: str, function_name: str) -> Dict[str, Any]:
    """Returns complexity + maintainability metrics for one code string."""
    try:
        blocks = cc_visit(code)
        target = next((b for b in blocks if b.name == function_name), None)

        if target is None:
            return {
                "status": "error",
                "error": f"Function '{function_name}' not found by radon in this code.",
            }

        mi_score = mi_visit(code, multi=True)

        return {
            "status": "success",
            "cyclomatic_complexity": target.complexity,
            "complexity_rank": cc_rank(target.complexity),
            "maintainability_index": round(mi_score, 2),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def complexity_tool(original_code: str, optimized_code: str, function_name: str) -> Dict[str, Any]:
    """
    Compares cyclomatic complexity and maintainability index between
    original_code and optimized_code, using static analysis (radon).

    Does not execute either version. No test_cases needed — this analyzes
    the code's structure, not its runtime behavior.

    Args:
        original_code: Full source of the original code. Must define a
            top-level function named `function_name`.
        optimized_code: Full source of the optimized code. Must define a
            top-level function with the same name.
        function_name: Name of the function to analyze in both versions.

    Returns:
        A dict with:
          - original: {"cyclomatic_complexity": int, "complexity_rank":
            "A"-"F", "maintainability_index": float 0-100}
          - optimized: same shape, for the optimized version
          - complexity_delta: optimized - original cyclomatic complexity
            (negative = optimized is simpler, positive = more complex)
          - maintainability_delta: optimized - original maintainability
            index (positive = optimized is more maintainable)
    """
    print(f"\n[complexity_tool] CALLED — function_name={function_name!r}", flush=True)

    orig_result = _analyze(original_code, function_name)
    opt_result = _analyze(optimized_code, function_name)

    complexity_delta = None
    maintainability_delta = None
    if orig_result["status"] == "success" and opt_result["status"] == "success":
        complexity_delta = opt_result["cyclomatic_complexity"] - orig_result["cyclomatic_complexity"]
        maintainability_delta = round(
            opt_result["maintainability_index"] - orig_result["maintainability_index"], 2
        )

    return {
        "original": orig_result,
        "optimized": opt_result,
        "complexity_delta": complexity_delta,
        "maintainability_delta": maintainability_delta,
    }


if __name__ == "__main__":
    import json

    original = """
def add_list(nums):
    total = 0
    for n in nums:
        if n is not None:
            total = total + n
        else:
            continue
    return total
"""
    optimized = """
def add_list(nums):
    return sum(n for n in nums if n is not None)
"""

    # result = complexity_tool.func(
    #     original_code=original,
    #     optimized_code=optimized,
    #     function_name="add_list",
    # )
    # print(json.dumps(result, indent=2, default=str))