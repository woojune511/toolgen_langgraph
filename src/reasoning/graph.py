from langgraph.graph import StateGraph, END
from functools import partial
from src.reasoning.state import ReasoningState
from src.reasoning.nodes import cot_reasoner, code_verifier, judge
from src.utils.jupyter_sandbox import AgentSandbox


def build_reasoning_graph(sandbox: AgentSandbox):
    """
    Reason → Code Verify → Judge 파이프라인을 구축한다.

    Flow:
        cot_reasoner → code_verifier → judge
              ↑                          |
              └── (불일치 시 재시도) ←───┘
    """
    workflow = StateGraph(ReasoningState)

    # 노드 등록
    workflow.add_node("cot_reasoner", cot_reasoner)
    workflow.add_node("code_verifier", partial(code_verifier, sandbox=sandbox))
    workflow.add_node("judge", judge)

    # 엣지 연결
    workflow.set_entry_point("cot_reasoner")
    workflow.add_edge("cot_reasoner", "code_verifier")
    workflow.add_edge("code_verifier", "judge")

    # Judge 라우팅: 검증 통과 → 종료, 불일치 → 재추론
    def judge_router(state: ReasoningState):
        if state.get("verified", False):
            return "end"
        else:
            return "retry"

    workflow.add_conditional_edges(
        "judge",
        judge_router,
        {
            "end": END,
            "retry": "cot_reasoner",
        }
    )

    return workflow.compile()
