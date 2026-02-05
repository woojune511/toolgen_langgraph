# ToolGen: LangGraph Agent Implementation

**ToolGen** is an autonomous agent framework built with **LangGraph** that dynamically generates, verifies, and utilizes tools to solve complex data science tasks. It leverages LLMs to decompose problems, generate Python code for tools, and iteratively refine them based on execution feedback.

This project specifically targets the **DSBench** benchmark, aiming to solve data science problems by creating and executing custom tools in a sandbox environment.

## 🚀 Key Features

*   **Autonomous Tool Generation**: Dynamically creates Python functions (tools) based on the current task and plan.
*   **Iterative Refinement**: Code is tested and refined through a feedback loop between the `Creator`, `Tester`, and `Solver` nodes.
*   **Sandboxed Execution**: Safe execution of generated code using `JupyterSandbox`.
*   **LangGraph Workflow**: Structured state machine managing planning, tool creation, testing, execution, and final answer generation.
*   **Performance Tracking**: Integrated with **LangFuse** for tracing and monitoring agent performance.

## 📂 Project Structure

```
toolgen_langgraph/
├── data/               # Datasets, logs, and databases (Ignored by Git)
├── src/
│   ├── agent/          # Core agent logic (nodes, graph, state)
│   ├── utils/          # Utilities (sandbox, code parser)
│   ├── config.py       # Configuration settings
│   └── logger.py       # Logging utility
├── main.py             # Entry point for running the agent
├── dsbench_loader.py   # Data loader for DSBench
├── requirements.txt    # Dependencies
└── .env                # Environment variables (API Keys)
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd toolgen_langgraph
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory:
    ```ini
    OPENAI_API_KEY=your_api_key_here
    OPENROUTER_API_KEY=your_openrouter_key
    # Add other keys as needed
    ```

## 📖 Usage

### Running the Agent on DSBench
The main script iterates through problems in the DSBench dataset.

```bash
python main.py
```

*   **Mode Selection**: Modify `MODE` in `main.py` (`"analysis"` or `"modeling"`).
*   **Dataset Path**: Ensure `DSBENCH_ROOT` in `main.py` points to your dataset directory.

### Testing Components
You can run individual test functions in `main.py` by uncommenting them at the bottom of the file:
*   `test_single()`: Run a single specific problem.
*   `run_test()`: Verify the `tool_tester_node` with sample valid/buggy code.

## 🤖 Agent Workflow

1.  **Planner**: Decomposes the user problem into a step-by-step plan.
2.  **Manager**: Decides whether to create a new tool or solve the current step.
3.  **Creator**: Generates Python code for a tool required by the plan.
4.  **Tester**: Statically analyzes and "unit tests" the generated tool.
5.  **Solver**: Executes the tool in the sandbox and updates the context.
    *   *Success*: Move to the next step.
    *   *Failure*: Return to **Creator** to fix the tool.
6.  **Final Answer**: Consolidates results to answer the original question.
