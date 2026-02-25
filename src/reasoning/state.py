from typing import TypedDict, Optional


class ReasoningState(TypedDict):
    # --- Input ---
    problem: str                # 원본 문제

    # --- CoT Reasoning ---
    cot_reasoning: str          # LLM의 추론 과정 (풀 CoT)
    cot_answer: str             # CoT에서 도출한 답

    # --- Code Verification ---
    code: str                   # 검증용 Python 코드
    code_result: str            # 코드 실행 결과
    code_error: Optional[str]   # 코드 실행 에러 (있으면)

    # --- Judge ---
    verified: bool              # CoT 답과 코드 결과 일치 여부
    attempt: int                # 현재 시도 횟수
    judge_reasoning: str        # Judge의 판단 근거

    # --- Output ---
    final_answer: Optional[str] # 최종 답
