import os
import json
import time
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from src.reasoning.graph import build_reasoning_graph
from src.logger import get_logger
from src.utils.jupyter_sandbox import AgentSandbox

load_dotenv()

logger = get_logger("ReasoningPipeline")

from src.config import MODEL_NAME

# 모델명에서 파일명 안전한 부분만 추출
_model_tag = MODEL_NAME.split("/")[-1].replace(".", "_")
RESULT_FILE = f"reasoning_results_{_model_tag}.json"


def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        _json = json.load(f)['test']

    dataset = []
    for domain in _json:
        dataset += _json[domain]

    return dataset


def load_existing_results():
    """이미 완료된 결과를 로드하여 중단 후 재개를 지원."""
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def main():
    dataset_path = "/c1/geonju/toolgen/datasets/math/math_100.json"

    logger.info("Starting Reasoning Pipeline — FULL DATASET")

    langfuse_handler = CallbackHandler()

    # Load dataset
    dataset = load_dataset(dataset_path)
    total = len(dataset)
    logger.info(f"Loaded {total} tasks from {dataset_path}")

    # Load existing results (for resume)
    results = load_existing_results()
    already_done = len(results)
    if already_done > 0:
        logger.info(f"Resuming: {already_done} tasks already completed")

    start_time = time.time()
    correct_count = 0

    for i in range(total):
        idx_str = str(i)

        # Skip already completed
        if idx_str in results:
            if results[idx_str].get("final_answer"):
                correct_count += 1 if results[idx_str].get("is_correct", False) else 0
            continue

        task = dataset[i]
        question = task['question']
        answer = task['answer']
        domain = task['domain']

        logger.info(f"\n[{i+1}/{total}] 🚀 Task [{i}] Domain: {domain}")
        logger.info(f"  Q: {question[:80]}...")

        try:
            with AgentSandbox(work_dir="./") as sandbox:
                app = build_reasoning_graph(sandbox)

                inputs = {
                    "problem": question,
                    "cot_reasoning": "",
                    "cot_answer": "",
                    "code": "",
                    "code_result": "",
                    "code_error": None,
                    "verified": False,
                    "attempt": 0,
                    "judge_reasoning": "",
                    "final_answer": None,
                }

                result = app.invoke(
                    inputs,
                    config={"recursion_limit": 20, "callbacks": [langfuse_handler]}
                )

                final_answer = result.get("final_answer", "")
                attempts = result.get("attempt", 0)

                results[idx_str] = {
                    "domain": domain,
                    "ground_truth": answer,
                    "final_answer": final_answer,
                    "attempts": attempts,
                    "cot_answer": result.get("cot_answer", ""),
                    "code_result": result.get("code_result", ""),
                }

                logger.info(f"  ✅ Answer: {final_answer[:40]} | GT: {answer} | Attempts: {attempts}")

        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
            results[idx_str] = {
                "domain": domain,
                "ground_truth": answer,
                "error": str(e),
            }

        # 매 10문제마다 중간 저장
        if (i + 1) % 10 == 0:
            save_results(results)
            elapsed = time.time() - start_time
            done = sum(1 for k, v in results.items() if "final_answer" in v or "error" in v)
            logger.info(f"  💾 Saved ({done}/{total} done, {elapsed/60:.1f}min elapsed)")

    # 최종 저장
    save_results(results)

    # 결과 요약
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 FULL DATASET RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total: {total}")
    logger.info(f"Completed: {sum(1 for v in results.values() if 'final_answer' in v)}")
    logger.info(f"Errors: {sum(1 for v in results.values() if 'error' in v)}")
    logger.info(f"Time: {elapsed/60:.1f} min")
    logger.info(f"Results saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()
