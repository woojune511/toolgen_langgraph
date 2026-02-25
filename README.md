# ToolGen: LangGraph Agent Implementation

**ToolGen** is an autonomous agent framework built with **LangGraph** that dynamically generates, verifies, and utilizes tools to solve complex data science tasks. It leverages LLMs to decompose problems, generate Python code for tools, and iteratively refine them based on execution feedback.

This project targets the **DSBench** benchmark for data science tasks and includes a **Reasoning Pipeline** for mathematical reasoning with code verification.

## 📊 Results

### MATH Benchmark (Llama 3.3 70B, 700 problems)

| Approach | Accuracy | Improvement |
|----------|----------|-------------|
| CoT Only (baseline) | 62.7% (439/700) | — |
| **Reasoning Pipeline (CoT + Code Verify)** | **77.9% (545/700)** | **+15.1pp** |

Code verification corrected **106 problems** that CoT alone got wrong.

| Domain | CoT Only | Pipeline | Δ |
|--------|----------|----------|---|
| Algebra | 90% | **98%** | +8 |
| Number Theory | 76% | **94%** | +18 |
| Prealgebra | 84% | **90%** | +6 |
| Counting & Probability | 65% | **82%** | +17 |
| Intermediate Algebra | 43% | **70%** | +27 |
| Precalculus | 35% | **58%** | +23 |
| Geometry | 46% | **53%** | +7 |

## 🚀 Key Features

*   **Autonomous Tool Generation**: Dynamically creates Python functions (tools) based on the current task and plan.
*   **Reasoning Pipeline**: CoT reasoning with code verification for math problem solving.
*   **Iterative Refinement**: Code is tested and refined through a feedback loop between the `Creator`, `Tester`, and `Solver` nodes.
*   **Sandboxed Execution**: Safe execution of generated code using `JupyterSandbox`.
*   **LangGraph Workflow**: Structured state machine managing planning, tool creation, testing, execution, and final answer generation.
*   **Performance Tracking**: Integrated with **LangFuse** for tracing and monitoring agent performance.

## 📂 Project Structure

```
toolgen_langgraph/
├── data/               # Datasets, logs, and databases
├── src/
│   ├── agent/          # ToolGen agent (nodes, graph, state)
│   ├── reasoning/      # Reasoning Pipeline (CoT + Code Verify)
│   │   ├── state.py    # ReasoningState definition
│   │   ├── nodes.py    # cot_reasoner, code_verifier, judge
│   │   └── graph.py    # LangGraph workflow
│   ├── utils/          # Utilities (sandbox, code parser)
│   ├── config.py       # Configuration settings
│   └── logger.py       # Logging utility
├── main.py             # Entry point for DSBench agent
├── reasoning_pipeline.py # Entry point for Reasoning Pipeline
├── grade_reasoning.py  # Grading script for MATH results
├── dsbench_loader.py   # Data loader for DSBench
├── requirements.txt    # Dependencies
└── .env                # Environment variables (API Keys)
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/woojune511/toolgen_langgraph.git
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
    ```

## 📖 Usage

### Reasoning Pipeline (MATH Benchmark)
```bash
python reasoning_pipeline.py
```
Runs the CoT → Code Verify → Judge pipeline on the MATH dataset. Results are saved to `reasoning_full_results.json` with resume support.

### ToolGen Agent (DSBench)
```bash
python main.py
```
*   **Mode Selection**: Modify `MODE` in `main.py` (`"analysis"` or `"modeling"`).
*   **Dataset Path**: Ensure `DSBENCH_ROOT` in `main.py` points to your dataset directory.

## 🤖 Architecture

### Reasoning Pipeline
```
Problem → CoT Reasoner → Code Verifier → Judge → Final Answer
                ↑                           |
                └──── (mismatch retry) ←────┘
```

### ToolGen Agent
1.  **Planner**: Decomposes the user problem into a step-by-step plan.
2.  **Manager**: Decides whether to create a new tool or solve the current step.
3.  **Creator**: Generates Python code for a tool required by the plan.
4.  **Tester**: Statically analyzes and "unit tests" the generated tool.
5.  **Solver**: Executes the tool in the sandbox and updates the context.
6.  **Final Answer**: Consolidates results to answer the original question.

