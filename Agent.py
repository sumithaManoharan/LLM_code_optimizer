"""
agent.py

LLM 2 — the Evaluation Agent. All 5 tools are wired up: correctness_tool,
runtime_tool, memory_tool, pylint_tool, complexity_tool.
"""

import os
import textwrap


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from correctness_tool import correctness_tool
from runtime_tool import runtime_tool
from memory_tool import memory_tool
from pylint_tool import pylint_tool
from complexity_tool import complexity_tool

load_dotenv(override=True)

EVAL_SYSTEM_PROMPT = """You are a code evaluation agent. You are given an
ORIGINAL Python function and an OPTIMIZED version of it, produced by another
agent.

You have five tools: correctness_tool, runtime_tool, memory_tool,
pylint_tool, complexity_tool. Follow this sequence exactly:

1. Read both versions of the function.
2. Design ONE comprehensive set of test cases, covering typical inputs,
   edge cases (empty input, zero, negative numbers, boundary values,
   floating-point precision edge cases if the function does numeric work),
   and at least one case likely to trigger an error path if one exists.
   Build this test suite ONCE, up front — do not test a small batch, get
   a result, then test more. Shape it as
   [{"args": [...], "kwargs": {...}}, ...].
3. Call correctness_tool EXACTLY ONCE with original_code, optimized_code,
   function_name, and that full test suite. Do not call correctness_tool
   again after this, regardless of the result.
4. If correctness_tool reports the optimization is NOT correct: stop here.
   Do not call any other tool. Report the failure, explaining which test
   cases failed and why, using the tool's `details`/`reason` fields.
5. If correctness_tool confirms the optimization IS correct, call each of
   these four tools EXACTLY ONCE, in this order:
   a. runtime_tool — same original_code/optimized_code/function_name.
      For test_cases, prefer a large or complex input (e.g. a big list)
      over trivial ones — trivial inputs produce near-zero, noisy timings.
   b. memory_tool — same arguments as runtime_tool, reusing the same
      test_cases is fine.
   c. pylint_tool — takes ONLY original_code and optimized_code. No
      function_name, no test_cases.
   d. complexity_tool — takes original_code, optimized_code, and
      function_name. No test_cases.
6. Report your final answer as a markdown table, in EXACTLY this shape —
   two data columns, "Your Code" (the original) and "Optimized Code":

   | **Metric**      | **Your Code**                                       | **Optimized Code**                                    |
   | --------------- | ----------------------------------------------------| ------------------------------------------------------|
   | **Correctness** | Reference (ground truth)                            | Passed X/Y test cases — or "Failed, see below"        |
   | **Runtime**     | <original median time, e.g. 0.0244s>                | <optimized median time> (<speedup>x faster/slower)    |
   | **Memory**      | <original peak, e.g. 39.5 MB>                       | <optimized peak> (<ratio>x less/more)                 |
   | **Pylint**      | <original score>/10, <N> issue(s)                   | <optimized score>/10, <N> issue(s)                    |
   | **Complexity**  | CC=<original complexity> (<rank>), MI=<original MI> | CC=<optimized complexity> (<rank>), MI=<optimized MI> |

   Fill every cell using the tools' actual returned numbers — never invent
   or round from memory. If correctness FAILED (step 4), still show this
   table: fill in the Correctness row with the failure detail, and write
   "N/A — not run" in both columns for Runtime, Memory, Pylint, and
   Complexity, since those tools were never called.

   Some correctness_tool test cases may PASS but include a `reason` noting
   they matched only within floating-point tolerance (not an exact match).
   That is not a failure — do not treat it as one, and do not stop the
   sequence for it. But still mention it briefly in the Correctness cell,
   e.g. "Passed 13/13 (1 case matched within floating-point tolerance,
   negligible difference)" — the person should be informed even though it
   isn't a real problem.

   After the table, add summary of the overall verdict. if there is a fail in any test explain what and why failed.
   Do not repeat the table's numbers in prose, and do not add any other
   sections.

Never guess at any of these yourself — always verify with the tools first.
Never call runtime_tool, memory_tool, pylint_tool, or complexity_tool
before correctness_tool has passed.
"""

model = ChatOpenAI(
    model="nex-agi/nex-n2.5-pro:free",
    temperature=0,  # keep test-case generation deterministic-ish
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

agent = create_agent(
    model=model,
    tools=[correctness_tool, runtime_tool, memory_tool, pylint_tool, complexity_tool],
    system_prompt=EVAL_SYSTEM_PROMPT,
)


def evaluate(original_code: str, optimized_code: str, function_name: str) -> str:
    """Runs the evaluation agent on one optimizer output, returns its final report."""
    user_message = f"""ORIGINAL:
```python
{original_code}
```

OPTIMIZED:
```python
{optimized_code}
```

function_name: {function_name}
"""
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    
    original ="""
    def find_duplicates(numbers):
        duplicates = []

        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if numbers[i] == numbers[j] and numbers[i] not in duplicates:
                    duplicates.append(numbers[i])

        return duplicates
    """

    optimized ="""
    def find_duplicates(numbers):
    seen = set()
    duplicates_set = set()

    for num in numbers:
        if num in seen:
            duplicates_set.add(num)
        else:
            seen.add(num)

    return list(duplicates_set)
    """

    print(evaluate(original, optimized, "add_list"))