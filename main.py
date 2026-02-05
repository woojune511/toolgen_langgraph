import os
import json
from src.agent.graph import build_graph
from src.logger import get_logger
from dsbench_loader import DSBenchLoader
import langchain
from langfuse.langchain import CallbackHandler
from src.config import RESULT_DIR
from src.utils.jupyter_sandbox import AgentSandbox

logger = get_logger("MainExecutor")

langchain.debug = True

def main():
    # 1. 설정
    # 실행 모드 선택: "analysis" 또는 "modeling"
    MODE = "analysis" 
    DSBENCH_ROOT = "/c1/geonju/project/data/dataset/DSBench"
    
    logger.info(f"Starting DSBench Execution | Mode: {MODE}")

    langfuse_handler = CallbackHandler()
    
    # 2. 로더 초기화
    try:
        loader = DSBenchLoader(DSBENCH_ROOT, mode=MODE)
    except Exception as e:
        logger.error(f"Failed to initialize loader: {e}")
        return
    
    # 4. 전체 데이터셋 순회 (Question 단위 실행)
    total_tasks = len(loader)
    logger.info(f"Total tasks to process: {total_tasks}")


    # 중간결과가 존재하면 로드
    if os.path.exists(os.path.join(RESULT_DIR, "result.json")):
        with open(os.path.join(RESULT_DIR, "result.json"), 'r') as f:
            final_answer_dict = json.load(f)
    else:
        final_answer_dict = {}
        
    for i in range(total_tasks):
        # 4-1. 문제 가져오기 (Flattened Question)
        task_data = loader.get_problem(i)
        
        t_id = task_data['id']

        # 결과가 이미 존재하면 continue
        if t_id in final_answer_dict and task_data["question_id"] in final_answer_dict[t_id]:
            continue
        
        if not t_id in final_answer_dict:
            final_answer_dict[t_id] = {}

        prompt = task_data['prompt']
        target_dir = task_data['target_dir']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Processing Task [{i+1}/{total_tasks}] ID: {t_id}")
        logger.info(f"📂 Work Dir: {target_dir}")
        logger.info(f"{'='*60}")

        with AgentSandbox(work_dir=target_dir) as sandbox:        

            app = build_graph(sandbox)

            # 4-2. 초기 상태 설정 (State Injection)
            inputs = {
                "problem": prompt,          # Stuffed Prompt (Intro + Excel Path + Question)
                "work_dir": target_dir,     # (중요) 에이전트가 작업할 절대 경로
                "plan": [],
                "current_step_index": 0,
                "decision": "",
                "context_log": [],
                "variable_inventory": {},
                "tool_retrieved": [],
                "tool_generated": [],
                "feedback_history": [],
                "error": None
            }
            
            # 4-3. 그래프 실행
            try:
                # recursion_limit: 복잡한 문제일수록 높게 잡아야 함 (50~100)
                result = app.invoke(inputs, config={"recursion_limit": 100, "callbacks": [langfuse_handler]})

                final_answer = result.get("final_answer")
                if final_answer:
                    logger.info(f"✅ Task {t_id} Completed.")
                    final_answer_dict[t_id][task_data["question_id"]] = final_answer
                    
                
                # (선택) 결과 확인 로직
                # Analysis: result['context_log'] 마지막 내용 확인
                # Modeling: target_dir에 submission.csv 생겼는지 확인
                
            except Exception as e:
                logger.error(f"❌ Task {t_id} Failed: {e}")
                # 에러가 나도 다음 문제로 계속 진행 (Continue)

        with open(os.path.join(RESULT_DIR, "result.json"), 'w') as f:
            json.dump(final_answer_dict, f, indent=4)



def test_single():
    # 1. 설정
    # 실행 모드 선택: "analysis" 또는 "modeling"
    MODE = "analysis" 
    DSBENCH_ROOT = "/c1/geonju/project/data/dataset/DSBench"

    langchain.debug = True
    langfuse_handler = CallbackHandler()
    
    logger.info(f"Starting DSBench Execution | Mode: {MODE}")
    
    # 2. 로더 초기화
    try:
        loader = DSBenchLoader(DSBENCH_ROOT, mode=MODE)
    except Exception as e:
        logger.error(f"Failed to initialize loader: {e}")
        return

    # 3. LangGraph 앱 생성
    app = build_graph()
    
    # 4. 전체 데이터셋 순회 (Question 단위 실행)
    total_tasks = len(loader)
    logger.info(f"Total tasks to process: {total_tasks}")

    problem_id = 0

    # 4-1. 문제 가져오기 (Flattened Question)
    task_data = loader.get_problem(problem_id)
    
    t_id = task_data['id']
    prompt = task_data['prompt']
    target_dir = task_data['target_dir']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Processing Task [{problem_id+1}/{total_tasks}] ID: {t_id}")
    logger.info(f"📂 Work Dir: {target_dir}")
    logger.info(f"{'='*60}")
    
    # 4-2. 초기 상태 설정 (State Injection)
    inputs = {
        "problem": prompt,          # Stuffed Prompt (Intro + Excel Path + Question)
        "work_dir": target_dir,     # (중요) 에이전트가 작업할 절대 경로
        "plan": [],
        "current_step_index": 0,
        "decision": "",
        "context_log": [],
        "variable_inventory": {},
        "tool_retrieved": [],
        "tool_generated": [],
        "feedback_history": [],
        "error": None
    }
    
    # 4-3. 그래프 실행
    try:
        # recursion_limit: 복잡한 문제일수록 높게 잡아야 함 (50~100)
        for state in app.stream(
            inputs,
            config={
                "recursion_limit": 50,
                "callbacks": [langfuse_handler]},
            stream_mode="values"
        ):
            # logger.info(f"현재 상태: {state}")
            # print(f"현재 상태: {state}")

            if state.get("final_answer"):
                print(f"FINAL ANSWER: {state.get("final_answer")}")
            
        # result = app.stream(inputs, config={"recursion_limit": 50})
        
        logger.info(f"✅ Task {t_id} Completed.")
        
        # (선택) 결과 확인 로직
        # Analysis: result['context_log'] 마지막 내용 확인
        # Modeling: target_dir에 submission.csv 생겼는지 확인
        
    except Exception as e:
        logger.error(f"❌ Task {t_id} Failed: {e}")
        # 에러가 나도 다음 문제로 계속 진행 (Continue)



def run_test():
    from src.agent.nodes import tool_tester_node
    from src.utils.jupyter_sandbox import JupyterSandbox
    
    print("🧪 Starting Tool Tester Node Verification...\n")
    
    # 테스트용 작업 디렉토리 설정
    work_dir = "./test_workspace"
    os.makedirs(work_dir, exist_ok=True)

    # 샌드박스 컨텍스트 안에서 실행해야 함 (tester_node가 sandbox를 호출하므로)
    with JupyterSandbox(work_dir=work_dir) as sandbox:
        
        # ---------------------------------------------------------
        # CASE 1: 정상적인 도구 (테스트 통과 예상)
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("🟢 [CASE 1] Testing Valid Tool (Should PASS)")
        print("="*50)
        
        valid_tool_state = {
            "work_dir": work_dir,
            "plan": ["Calculate average"],
            "current_step_index": 0,
            "context_log": [],
            # 가짜로 생성된 도구 리스트 주입
            "tool_generated": [
                {
                    "name": "calculate_mean",
                    "docstring": "Calculates the mean of a list.",
                    # 정상 코드: 리스트의 평균을 구함
                    "code": """
def calculate_mean(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
"""
                }
            ]
        }
        
        # 노드 실행!
        result_1 = tool_tester_node(valid_tool_state)
        
        print(f"\n[Result]: {result_1['decision']}")
        if result_1['decision'] == "solve":
            print("✅ CASE 1 PASSED: Correctly accepted valid code.")
        else:
            print(f"❌ CASE 1 FAILED: Unexpectedly rejected valid code. Error: {result_1.get('error')}")


        # ---------------------------------------------------------
        # CASE 2: 고장난 도구 (테스트 실패 예상)
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("🔴 [CASE 2] Testing Buggy Tool (Should FAIL)")
        print("="*50)
        
        buggy_tool_state = {
            "work_dir": work_dir,
            "plan": ["Calculate average"],
            "current_step_index": 0,
            "context_log": [],
            "tool_generated": [
                {
                    "name": "calculate_mean_buggy",
                    "docstring": "Calculates the mean.",
                    # 버그 코드: 합계를 구하지 않고 그냥 길이로 나눔 (논리 오류)
                    # 혹은 문법 에러를 넣어봐도 됨
                    "code": """
def calculate_mean_buggy(numbers):
    '''
    Calculates the mean of a list of numbers.
    Input:
        - numbers: list of numbers
    Output:
        - mean of the numbers
    '''
    return len(numbers) 
"""
                }
            ]
        }
        
        # 노드 실행!
        result_2 = tool_tester_node(buggy_tool_state)
        
        print(f"\n[Result]: {result_2['decision']}")
        
        if result_2['decision'] == "retry_create":
            print("✅ CASE 2 PASSED: Correctly caught the bug.")
            print(f"   Error Log Captured: {result_2.get('error')}...") # 에러 내용 일부 출력
        else:
            print("❌ CASE 2 FAILED: Failed to catch the bug (returned 'solve').")


if __name__ == "__main__":
    main()
    # test_single()
    # run_test()