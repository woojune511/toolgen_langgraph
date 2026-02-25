import os
import json
import time
from src.agent.graph import build_graph
from src.logger import get_logger
from dsbench_loader import DSBenchLoader
from langfuse.langchain import CallbackHandler
from src.utils.jupyter_sandbox import AgentSandbox

logger = get_logger("GroundingTest")

def main():
    DSBENCH_ROOT = "/c1/geonju/toolgen_langgraph/data/dataset/DSBench"
    MAX_TASKS = 10

    loader = DSBenchLoader(DSBENCH_ROOT, mode="analysis")
    langfuse_handler = CallbackHandler()
    
    total = min(MAX_TASKS, len(loader))
    logger.info(f"Testing {total} DSBench tasks with Grounding Node")

    results = {}
    total_start = time.time()

    for i in range(total):
        task_data = loader.get_problem(i)
        t_id = task_data['id']
        target_dir = task_data['target_dir']

        logger.info(f"\n[{i+1}/{total}] 🚀 Task: {t_id}")
        task_start = time.time()

        try:
            with AgentSandbox(work_dir=target_dir) as sandbox:
                app = build_graph(sandbox)
                inputs = {
                    "problem": task_data['prompt'],
                    "work_dir": target_dir,
                    "plan": [],
                    "current_step_index": 0,
                    "decision": "",
                    "context_log": [],
                    "grounding_context": "",
                    "variable_inventory": {},
                    "tool_retrieved": [],
                    "tool_generated": [],
                    "feedback_history": [],
                    "error": None
                }

                result = app.invoke(inputs, config={"recursion_limit": 100, "callbacks": [langfuse_handler]})

                elapsed = time.time() - task_start
                final = result.get("final_answer", "")
                grounding_len = len(result.get("grounding_context", ""))
                plan_len = len(result.get("plan", []))

                logger.info(f"  ✅ Done in {elapsed:.0f}s | Plan: {plan_len} steps | Grounding: {grounding_len} chars")
                logger.info(f"  Answer: {str(final)[:100]}...")

                results[t_id] = {
                    "time_s": round(elapsed, 1),
                    "plan_steps": plan_len,
                    "grounding_chars": grounding_len,
                    "answer_preview": str(final)[:200],
                    "question_id": task_data["question_id"],
                }

        except Exception as e:
            elapsed = time.time() - task_start
            logger.error(f"  ❌ Failed in {elapsed:.0f}s: {e}")
            results[t_id] = {"error": str(e), "time_s": round(elapsed, 1)}

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 SUMMARY ({total} tasks, {total_elapsed:.0f}s total)")
    logger.info(f"{'='*60}")
    
    times = [v["time_s"] for v in results.values() if "time_s" in v]
    errors = sum(1 for v in results.values() if "error" in v)
    logger.info(f"  Avg time: {sum(times)/len(times):.0f}s | Errors: {errors}/{total}")
    
    for tid, r in results.items():
        status = "❌" if "error" in r else "✅"
        logger.info(f"  {status} {tid}: {r['time_s']:.0f}s | {r.get('answer_preview','ERROR')[:60]}")

    with open("grounding_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
