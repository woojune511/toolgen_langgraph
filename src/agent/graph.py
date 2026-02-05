from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    planner_node, tool_manager_node, tool_creator_node, 
    tool_tester_node, solver_node, final_answer_node
)
from functools import partial

def build_graph(sandbox):
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("planner", planner_node)
    workflow.add_node("manager", tool_manager_node)
    workflow.add_node("creator", tool_creator_node)
    workflow.add_node("tester", partial(tool_tester_node, sandbox=sandbox))
    workflow.add_node("solver", partial(solver_node, sandbox=sandbox))
    workflow.add_node("final_answer", partial(final_answer_node, sandbox=sandbox))

    # 엣지 연결
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "manager")

    # Manager 분기
    def manager_router(state):
        return "solver" if state["decision"] == "solve" else "creator"
    workflow.add_conditional_edges("manager", manager_router)

    # Creator -> Tester
    workflow.add_edge("creator", "tester")

    # Tester 분기
    def tester_router(state):
        return "solver" if state["decision"] == "solve" else "creator"
    workflow.add_conditional_edges("tester", tester_router)

    # # Solver 분기
    # def solver_router(state):
    #     d = state["decision"]
    #     if d == "end": return END
    #     elif d == "retry_create": return "creator"
    #     return "manager" # 다음 스텝 진행

    def route_after_solver(state: AgentState):
        # 1. Solver의 판정 결과를 먼저 확인
        decision = state.get("decision")
        
        # 🚨 CASE A: Solver 실패 (에러 발생)
        # -> 피드백을 들고 다시 도구를 고치러(Creator) 가야 함
        # -> 이때 step은 증가하지 않은 상태임
        if decision == "retry_create":
            return "tool_creator"

        # ✅ CASE B: Solver 성공
        elif decision == "continue":
            plan = state['plan']
            current_step = state['current_step_index']
            
            # B-1: 아직 수행할 계획(Step)이 남았음
            # -> 다음 Step을 위한 도구를 만들러(Creator) 이동
            if current_step < len(plan):
                return "manager"
                
            # B-2: 모든 계획 완료!
            # -> 최종 답변 작성(Final Answer)으로 이동
            else:
                return "final_answer"

        # 예외 상황 (혹시 decision이 없으면 안전하게 Creator로)
        return "tool_creator"

    workflow.add_conditional_edges(
        "solver",
        route_after_solver,
        {
            "manager": "manager",
            "final_answer": "final_answer",
            "tool_creator": "creator"
        }
    )

    workflow.add_edge("final_answer", END)

    return workflow.compile()