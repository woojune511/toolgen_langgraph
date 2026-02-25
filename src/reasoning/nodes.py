from langchain_openai import ChatOpenAI
import json
import re
from src.config import BASE_URL, API_KEY, MODEL_NAME
from src.reasoning.state import ReasoningState
from src.logger import get_logger
from src.utils.jupyter_sandbox import AgentSandbox

logger = get_logger(__name__)

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0.7,
    max_retries=2,
    request_timeout=120,
    model_kwargs={
        "extra_body": {
            "reasoning": {"effort": "none"}
        }
    },
)


def cot_reasoner(state: ReasoningState):
    """
    Step 1: CoT 추론
    LLM에게 문제를 단계별로 풀게 한다.
    재시도 시에는 이전 코드 검증 결과를 힌트로 제공한다.
    """
    problem = state["problem"]
    attempt = state.get("attempt", 0)

    logger.info(f"🧠 CoT Reasoning (attempt {attempt + 1})...")

    if attempt == 0:
        # 첫 시도: 순수 CoT
        prompt = f"""You are a math expert. Solve the following problem step by step.

## Problem:
{problem}

## Instructions:
1. Think through the problem carefully, step by step.
2. Show all your work and reasoning.
3. At the end, provide your final answer inside \\boxed{{}}.

## Solution:
"""
    else:
        # 재시도: 이전 코드 결과를 힌트로 제공
        prev_code_result = state.get("code_result", "")
        prev_cot_answer = state.get("cot_answer", "")
        judge_reasoning = state.get("judge_reasoning", "")

        prompt = f"""You are a math expert. Your previous answer to this problem was INCORRECT.

## Problem:
{problem}

## Your Previous Answer: {prev_cot_answer}
## Code Verification Result: {prev_code_result}
## Why it was wrong: {judge_reasoning}

## Instructions:
1. Carefully reconsider the problem. Your previous approach had an error.
2. Use the code verification result as a hint — the code computed a different answer.
3. Think through the problem again step by step.
4. At the end, provide your corrected answer inside \\boxed{{}}.

## Corrected Solution:
"""

    response = llm.invoke(prompt).content
    logger.info(f"   CoT output length: {len(response)} chars")

    # boxed answer 추출 (nested braces 처리)
    def extract_boxed(text):
        """\boxed{} 안의 내용을 추출. 중첩 중괄호도 처리."""
        results = []
        for m in re.finditer(r'\\boxed\{', text):
            start = m.end()
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                results.append(text[start:i-1])
        return results

    boxed_match = extract_boxed(response)
    cot_answer = boxed_match[-1] if boxed_match else ""

    if not cot_answer:
        # fallback: 마지막 줄에서 답 추출 시도
        lines = response.strip().split('\n')
        cot_answer = lines[-1].strip() if lines else ""

    logger.info(f"   CoT Answer: {cot_answer}")

    return {
        "cot_reasoning": response,
        "cot_answer": cot_answer,
        "attempt": attempt + 1,
    }


def code_verifier(state: ReasoningState, sandbox: AgentSandbox):
    """
    Step 2: 코드 검증
    CoT 추론 결과를 검증하는 Python 코드를 생성하고 실행한다.
    """
    problem = state["problem"]
    cot_reasoning = state["cot_reasoning"]
    cot_answer = state["cot_answer"]

    logger.info(f"💻 Generating verification code...")

    prompt = f"""You are a Python programmer. Your task is to write Python code that independently solves the following math problem to verify a given answer.

## Problem:
{problem}

## Proposed Answer (from reasoning): {cot_answer}

## Reasoning Process:
{cot_reasoning}

## Instructions:
1. Write Python code that solves this problem computationally.
2. The code should calculate the answer INDEPENDENTLY — do NOT just print the proposed answer.
3. Use libraries like `math`, `fractions`, `itertools`, `sympy` as needed.
4. At the end, print ONLY the final answer (nothing else).
5. If the answer is a fraction, use `fractions.Fraction` and print it in the form "a/b".
6. Keep the code simple and correct.

```python
# Your verification code here
```
"""

    response = llm.invoke(prompt).content

    # 코드 추출
    code_match = re.search(r'```python(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # fallback: 전체를 코드로 취급
        code = response.strip()

    logger.info(f"   Code length: {len(code)} chars")

    # 코드 실행
    try:
        result = sandbox.run_code(code, mode="temporary")
        sandbox.cleanup_test_kernel()

        if result["stderr"]:
            logger.warning(f"   ⚠️ Code Error: {result['stderr'][:200]}")
            return {
                "code": code,
                "code_result": "",
                "code_error": result["stderr"],
            }

        code_result = result["stdout"].strip()
        logger.info(f"   Code Result: {code_result}")

        return {
            "code": code,
            "code_result": code_result,
            "code_error": None,
        }

    except Exception as e:
        logger.error(f"   ❌ Code execution failed: {e}")
        return {
            "code": code,
            "code_result": "",
            "code_error": str(e),
        }


def judge(state: ReasoningState):
    """
    Step 3: Judge
    CoT 답과 코드 결과를 비교하여 최종 답을 결정한다.
    """
    cot_answer = state["cot_answer"]
    code_result = state.get("code_result", "")
    code_error = state.get("code_error")
    attempt = state["attempt"]

    logger.info(f"⚖️ Judging (attempt {attempt})...")
    logger.info(f"   CoT: {cot_answer} | Code: {code_result} | Error: {code_error}")

    # 코드 에러가 있으면 CoT 답을 신뢰
    if code_error:
        logger.info(f"   Code had errors. Trusting CoT answer.")
        return {
            "verified": True,
            "final_answer": cot_answer,
            "judge_reasoning": f"Code execution failed ({code_error}). Using CoT answer.",
        }

    # 최대 시도 횟수 초과 시 코드 결과 채택
    if attempt >= 3:
        logger.info(f"   Max attempts reached. Using code result.")
        final = code_result if code_result else cot_answer
        return {
            "verified": True,
            "final_answer": final,
            "judge_reasoning": "Max attempts reached. Adopting code result.",
        }

    # LLM에게 비교 판단 요청
    prompt = f"""You are a math judge. Compare two answers to a math problem and determine if they are equivalent.

## CoT Answer: {cot_answer}
## Code Result: {code_result}

## Instructions:
- Answers may be in different formats (e.g., "13/3" vs "4.333..." vs "4 1/3" vs "\\frac{{13}}{{3}}").
- Determine if they represent the SAME mathematical value.
- Respond with EXACTLY one of:
  - "MATCH" — if the answers are mathematically equivalent
  - "MISMATCH: <brief explanation of the difference>"
"""

    response = llm.invoke(prompt).content.strip()
    logger.info(f"   Judge verdict: {response}")

    if response.startswith("MATCH"):
        return {
            "verified": True,
            "final_answer": cot_answer,
            "judge_reasoning": "CoT and code agree.",
        }
    else:
        # 불일치 — 재추론 필요
        mismatch_reason = response.replace("MISMATCH:", "").strip()
        logger.info(f"   ❌ Mismatch detected. Will retry reasoning.")
        return {
            "verified": False,
            "judge_reasoning": f"CoT={cot_answer}, Code={code_result}. {mismatch_reason}",
        }
