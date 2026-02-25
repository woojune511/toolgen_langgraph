
import os
import json
import langchain
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from src.agent.graph import build_graph
from src.logger import get_logger
from src.utils.jupyter_sandbox import AgentSandbox
from src.config import RESULT_DIR

# Load environment variables
load_dotenv()

logger = get_logger("DebugExecutor")
langchain.debug = True

def main():
    # Configuration
    dataset_path = "/c1/geonju/toolgen/datasets/math/math_100.json"
    FAILURE_INDICES = [0, 4, 36, 38, 52]  # Indices of failed cases to reproduce

    logger.info(f"Starting Debug Execution for MATH dataset")
    logger.info(f"Reproducing failures for indices: {FAILURE_INDICES}")

    langfuse_handler = CallbackHandler()

    # Load Dataset
    try:
        with open(dataset_path, 'r') as f:
            _json = json.load(f)['test']
        
        dataset = []
        for domain in _json:
            dataset += _json[domain]
            
        logger.info(f"Loaded {len(dataset)} tasks from {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    for i in FAILURE_INDICES:
        try:
            if i >= len(dataset):
                logger.error(f"Index {i} out of bounds (max {len(dataset)-1})")
                continue

            # Get problem data
            task = dataset[i]
            question = task['question']
            answer = task['answer']
            domain = task['domain']

            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 Debugging Task [{i}] Domain: {domain}")
            logger.info(f"Question: {question[:100]}...")
            logger.info(f"{'='*60}")

            # Initialize Sandbox and build graph
            # Math tasks don't have a specific work_dir, so we use current dir or a temp dir
            with AgentSandbox(work_dir="./") as sandbox:
                app = build_graph(sandbox)

                # inputs
                inputs = {
                    "problem": question,
                    "work_dir": "./",
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

                # Run graph
                result = app.invoke(inputs, config={"recursion_limit": 100, "callbacks": [langfuse_handler]})
                
                final_answer = result.get("final_answer")
                if final_answer:
                    logger.info(f"✅ Debugging Task {i} Completed.")
                    logger.info(f"Final Answer: {final_answer}")
                    logger.info(f"Ground Truth: {answer}")
                else:
                    logger.warning(f"⚠️ Debugging Task {i} Completed but no Final Answer found.")

        except Exception as e:
            logger.error(f"❌ Debugging Task {i} Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
