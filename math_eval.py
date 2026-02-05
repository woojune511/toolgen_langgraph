import os
import json
from src.agent.graph import build_graph
from src.logger import get_logger
import langchain
from langfuse.langchain import CallbackHandler
from src.config import RESULT_DIR
from src.utils.jupyter_sandbox import AgentSandbox

logger = get_logger("MainExecutor")

langchain.debug = True

def main():
    # 1. 설정
    dataset_path = "/c1/geonju/toolgen/datasets/math/math_100.json"
    final_answer_dict = {}
    
    with open(dataset_path, 'r') as f:
        _json = json.load(f)['test']
    
    dataset = []
    for domain in _json:
        dataset += _json[domain]
    
    logger.info(f"Starting MATH 100 Evaluation")

    langfuse_handler = CallbackHandler()
    
    # 4. 전체 데이터셋 순회 (Question 단위 실행)
    total_tasks = len(dataset)
    logger.info(f"Total tasks to process: {total_tasks}")
        
    for i in range(total_tasks):
        # 4-1. 문제 가져오기 (Flattened Question)
        task = dataset[i]
        question = task['question']
        answer = task['answer']
        domain = task['domain']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Processing Task [{i+1}/{total_tasks}]")
        logger.info(f"{'='*60}")

        with AgentSandbox() as sandbox:        

            app = build_graph(sandbox)

            # 4-2. 초기 상태 설정 (State Injection)
            inputs = {
                "problem": question,          # Stuffed Prompt (Intro + Excel Path + Question)
                "work_dir": "./",     # (중요) 에이전트가 작업할 절대 경로
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
                    logger.info(f"✅ Task {i} Completed.")
                    final_answer_dict[i] = {
                        "question": question,
                        "answer": answer,
                        "domain": domain,
                        "model_answer": final_answer
                    }
                    
                
                # (선택) 결과 확인 로직
                # Analysis: result['context_log'] 마지막 내용 확인
                # Modeling: target_dir에 submission.csv 생겼는지 확인
                
            except Exception as e:
                logger.error(f"❌ Task {i} Failed: {e}")
                # 에러가 나도 다음 문제로 계속 진행 (Continue)

        with open(os.path.join(RESULT_DIR, "math_100_result.json"), 'w') as f:
            json.dump(final_answer_dict, f, indent=4)


if __name__ == "__main__":
    main()
    # test_single()
    # run_test()