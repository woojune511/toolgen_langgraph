from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

class AgentState(TypedDict):
    # --- Input ---
    problem: str                # (Loader가 만든 Stuffed Prompt)
    work_dir: str               # (중요) 데이터 파일이 있는 실제 절대 경로
    
    # --- Internal Logic ---
    plan: Any                   # Planner가 만든 계획
    current_step_index: int     # 현재 계획 단계
    decision: str               # 라우팅 결정
    context_log: Annotated[List[str], operator.add]
    error: Optional[str]
    variable_inventory: Dict[str, str]
    tool_generated: List[Dict[str, str]]
    tool_retrieved: List[Dict[str, str]]
    feedback_history: List[Dict[str, Any]]
    
    # --- Output/Eval ---
    final_answer: Optional[str] # 최종 답변 (Analysis용)
    submission_path: Optional[str] # 최종 csv 경로 (Modeling용)