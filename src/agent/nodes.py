from langchain_openai import ChatOpenAI
import json
from src.config import BASE_URL, API_KEY, MODEL_NAME
from src.agent.state import AgentState
from src.memory.tool_memory import ToolMemory
from src.logger import get_logger
from src.utils.jupyter_sandbox import AgentSandbox
from src.utils.code_parser import parse_tools_from_code
import pprint
from textwrap import dedent
import re
import traceback

logger = get_logger(__name__)
memory = ToolMemory()

# TODO: final answer에서 정보가 부족하다고 판단하면? plan 수정?
# TODO: tool reusability 향상
# TODO: 이미 retrieve된 tool을 고치려 하는 경우가 있나? 고치는 게 맞나?

# vLLM 연동
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=1,
    max_retries=2,
    request_timeout=120,
)


def planner_node(state: AgentState):
    logger.info(f"Planning task...")


# data analysis task에서는 파일을 직접 읽어서 넣어주고,
# data modeling task에서는 폴더 경로만 주고 파일을 LLM이 직접 읽어야 함
# 3. **File I/O**:
    # - The first step of the plan has to be checking the file list in the working directory.
    # - You generally do NOT know the exact filenames yet. You must check them first.
    # - The second step of the plan has to be loading the relevant data file(s) identified in the first step.
    
    prompt = f"""Analyze the User Request and break it down into step-by-step Python coding tasks.
                Also, the subtasks should be general so that it can be applicable to other similar problems.
                You do **not** need to define specific Python function signatures at this stage.
                Instead, clearly define each subtask with sufficient abstraction and generality, enabling later selection or generation of reusable tools (Python functions).

                Follow these guidelines strictly:

                1. **Clearly define each subtask**:
                    - Each subtask must represent an independent, fundamental step solvable by a Python function.
                    - Clearly describe the input/output data required and intermediate values, without specifying exact function signatures.

                2. **Abstract and general**:
                    - Design subtasks broadly enough to encourage reuse across multiple similar queries.
                    - Avoid overly narrow or query-specific subtasks.
                    - Consider the meaning of each value in solving given problem and focus on underlying core principles and essence, not superficial functionality.
                    - Do not include trivial, meaningless step.
                    - Do not turn problem-specific logic into a subtask.

                ---

                ## Query:
                ===query===

                ---
                Respond **strictly** in the following **JSON format**:

                ```json
                [
                    {{
                        "name": "Name of subtask 1",
                        "description": "A brief, abstract description of subtask 1.",
                        "input": {{
                            "input_var_1": {{
                                "description": "Explain what this variable represents.",
                                "type": "data type (e.g., float, list[float])",
                                "shape": "scalar / list / expression / etc.",
                                "example": "example value (optional)
                            }}
                        }},
                        "output": {{
                            "output_var_1": {{
                                "description": "What this output represents.",
                                "type": "data type",
                                "shape": "scalar / list / expression / etc.",
                                "example": "example output (optional)"
                            }}
                        }}
                    }},
                    # add more subtasks as needed
                ]
                ```
            """


    prompt = f"""Analyze the User Request and break it down into subtasks.
If necessary, you can use Python coding tasks.

Follow these guidelines strictly:

1. **Clearly define each subtask**:
    - Each subtask must represent an independent, fundamental step solvable by a Python function.
    - Clearly describe the input/output data required and intermediate values, without specifying exact function signatures.

---

## Query:
===query===

---
Respond **strictly** in the following **JSON format**:

```json
[
    {{
        "name": "Name of subtask 1",
        "description": "A description of subtask 1."
    }},
    # add more subtasks as needed
]
```
"""
    # prompt = dedent(prompt)
    while True:
        try:
            plan = llm.invoke(prompt.replace('===query===', state['problem'])).content

            if '```json' in plan:
                plan = plan.split('```json')[1].split('```')[0]

            plan = json.loads(plan)
            break
        except:
            # try:
            #     plan = eval(plan)
            #     break
            # except:
            #     logger.error(f"{'='*20} [Plan Parse Error] {'='*20}\n")
            logger.error(f"{'='*20} [Plan Parse Error] {'='*20}\n")

    # print(f"\n{'='*20} [LLM RAW OUTPUT] {'='*20}")
    # pprint.pp(plan)
    # print(f"{'='*60}\n")
    
    return {"plan": plan, "current_step_index": 0, "context_log": []}



def tool_manager_node(state: AgentState):
    plan = state['plan']
    idx = state['current_step_index']
    current_task = plan[idx]
    logger.info(f"🔍 Checking tools for {idx+1}/{len(plan)} tasks...")

    # 1. 벡터 DB에서 검색
    candidates = memory.search_tools(current_task['description'], k=5)

    if not candidates:
        logger.info("❌ No candidates found in DB.")
        return {
            "tool_retrieved": [],
            "decision": "create"
        }

    candidates_info = "\n".join([
        f"[{i}] Name: {c['name']}\n    Description: {c['docstring']}\n    Code: {c['code']}"
        for i, c in enumerate(candidates)
    ])

    prompt = (
        f"You are a Tool Manager. Your goal is to decide whether tools are necessary or not, and if necessary, to reuse an existing tool or create a new one.\n\n"
        f"🎯 **Current Task**: {current_task['description']}\n"
        f"🔎 **Candidate Tools found in Memory**:\n"
        f"{candidates_info}\n\n"
        f"🛑 **Instruction**:\n"
        f"1. Analyze if any of the candidates PERFECTLY matches the Current Task.\n"
        f"2. Consider if the input variables required by the tool are available.\n"
        f"3. If a match is found, return the index number (e.g., '0', '1').\n"
        f"4. If NONE match or strictly require modification return 'CREATE'.\n\n"
        f"5. If tool is not necessary, return 'NO TOOL'.\n\n"
        f"Answer ONLY with the index number, 'CREATE' or 'NO TOOL'."
    )

    response = llm.invoke(prompt).content.strip()

    if response.upper() == "CREATE":
        logger.info("🤔 Candidates rejected. Creating new tool.")
        return {"decision": "create"}
    elif response.upper() == "NO TOOL":
        logger.info("🤔 Candidates rejected. No tool is needed.")
        return {
            "tool_retrieved": [],
            "tool_generated": [],
            "decision": "solve"
        }
    else:
        try:
            # 인덱스로 선택된 도구 가져오기
            selected_idx = int(response)
            selected_tool = candidates[selected_idx]
            
            logger.info(f"♻️ Reusing tool: {selected_tool['name']}")
            return {
                "tool_retrieved": [selected_tool],
                "tool_generated": [],
                "decision": "solve"
            }
        except ValueError:
            # LLM이 이상한 답을 하면 안전하게 생성으로 이동
            return {
                "tool_retrieved": [],
                "tool_generated": [],
                "decision": "create"
            }
    

def tool_creator_node(state: AgentState):
    plan = state['plan']
    idx = state['current_step_index']
    current_task = plan[idx]
    history = state.get("feedback_history", [])

    if history:
        # 🔄 수정 모드 (Fix Mode with History)
        logger.info(f"🔧 Fixing tool based on {len(history)} past failures...")
        
        # 히스토리를 텍스트로 예쁘게 포맷팅
        history_summary = ""
        for item in history:
            history_summary += f"=== ❌ ATTEMPT (Source: {item['source'].upper()}) ===\n"
            history_summary += f"[Tool Code Used]:\n{item.get('tool_code', 'N/A')}\n\n"
            history_summary += f"[Test/Exec Code]:\n{item.get('test_code') or item.get('execution_code')}\n\n"
            history_summary += f"[Error Log]:\n{item['error_log']}\n"
            history_summary += f"==========================================================\n"
    
        prompt = f"""You are a Senior Python Data Engineer & Debugging Expert.
            Your goal is to fix a broken tool based on the provided error log.

            ---
            ### 1. CONTEXT
            **Original Task:**
            {current_task}

            **Previous Attempts & Failures:**
            {history_summary}

            ---
            ### 2. 🕵️‍♂️ DIAGNOSTIC CHECKLIST (Review these BEFORE fixing)

            Run through this mental checklist to identify the root cause. Do NOT assume data quality is perfect.

            **A. Type Mismatch (Most Common)**
            - ❌ **Issue:** Treating `int`/`str` as `datetime` (e.g., `AttributeError: 'int' object has no attribute 'year'`).
            - ✅ **Fix:** Explicitly convert columns using `pd.to_datetime()` or `astype()` before processing.
            - ❌ **Issue:** Treating a scalar (float) as a list/iterable.
            - ✅ **Fix:** Wrap scalar values in a list `[value]` if iteration is needed.

            **B. Data Structure & Keys**
            - ❌ **Issue:** `KeyError` or `IndexError` (Assuming a column name or index exists).
            - ✅ **Fix:** Check column names (case-sensitivity, strip whitespace). Use `.get()` or check `if col in df.columns`.
            - ❌ **Issue:** Applying DataFrame methods to a Series (or vice versa).

            **C. Scope & Definitions**
            - ❌ **Issue:** `NameError` (Using variables/imports not defined inside the function).
            - ✅ **Fix:** Ensure all imports (e.g., `import pandas as pd`) and variables are defined INSIDE the function or passed as arguments.

            **D. Logic & Math**
            - ❌ **Issue:** Division by zero, or `NaN` propagation.
            - ✅ **Fix:** Handle edge cases (empty data, `NaN` values) using `.fillna()` or `if not data.empty`.

            ---
            ### 3. 📝 YOUR TASK (Chain of Thought)

            **Step 1: ANALYSIS**
            - Identify the specific line number from the Error Log.
            - Explain WHY the error occurred based on the "Diagnostic Checklist" above.
            - Explicitly state what assumption in the old code was wrong (e.g., "Code assumed 'date' column was datetime object, but it was likely an integer/string").

            **Step 2: REFACTORING**
            - Write the corrected Python code.
            - **CRITICAL:** Add defensive coding (e.g., explicit type conversion, check for empty df) to prevent this from happening again.
            - Include the necessary imports within the code.

            ---
            ### 4. OUTPUT FORMAT

            🔴 **OUTPUT FORMAT**:
                <analysis>your diagnosis here</analysis>
                <main_func>name of the function to call</main_func>
                <description>what this tool does</description>
                ```python
                # your code
                # DO NOT INCLUDE TEST CODE
                # ONLY INCLUDE ORIGINAL TOOL CODE
                ```
        """

        prompt = dedent(prompt)

    else:
        # ✨ 생성 모드 (Create Mode)
        logger.info("🚀 Creating new tool...")
    
        prompt = (
            f"You are a generic Python function generator.\n"
            f"Task to solve:\n"
            f"{json.dumps(current_task, indent=2)}\n\n"
            f"Requirements:\n"
            f"1. Create a Python function for the task.\n"
            f"2. The function must be independent and self-contained.\n"
            f"3. Return the result as a Python code snippet with tools (function definitions) in it.\n"
            f"4. Include the docstring for the function.\n"
            f"🔴 **OUTPUT FORMAT (Strict JSON)**:\n"
            f"<analysis>your analysis here</analysis>\n"
            f"<main_func>name of the function to call</main_func>\n"
            f"<description>what this tool does</description>\n"
            f"```python\n"
            f"# your code\n"
            f"```"
        )
    
    max_attempt = 3
    for attempt in range(max_attempt):
        try:
            response = llm.invoke(prompt).content

            analysis_match = re.search(r"<analysis>(.*?)</analysis>", response, re.DOTALL)
            desc_match = re.search(r"<description>(.*?)</description>", response, re.DOTALL)
            func_match = re.search(r"<main_func>(.*?)</main_func>", response, re.DOTALL)
            code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
            
            if not desc_match or not func_match or not code_match:
                raise ValueError("Failed to extract tool information")
            
            description = desc_match.group(1).strip()
            name = func_match.group(1).strip()
            code = code_match.group(1).strip()

        except Exception as e:
            if attempt == max_attempt - 1:
                logger.error(f"Tool generation failed: {e}")
                return {
                    # decision?
                    "tool_generated": [],
                    "error": "Parsing Failed"
                }
            else:
                continue

    logger.info(f"✅ Generated a new tool.")
    
    return {
        "tool_generated": [{
            "name": name,
            "code": code,
            "docstring": description
        }],
        "tool_retrieved": [],
        "error": None
    }



def tool_tester_node(state: AgentState, sandbox: AgentSandbox):
    tools = state['tool_generated']
    work_dir = state['work_dir']
    history = state.get("feedback_history", [])
    
    logger.info(f"🧪 Starting Sequential Testing for {len(tools)} tools...")
    
    # ---------------------------------------------------------
    # 1단계: 모든 함수 정의(Definition) 로드
    # (서로 의존성이 있을 수 있으므로 일단 다 메모리에 올립니다)
    # ---------------------------------------------------------
    all_defs = "\n\n".join([t['code'] for t in tools])
    
    # 정의 단계에서 에러나면 바로 Creator로 반려 (Syntax Error 등)
    def_result = sandbox.run_code(all_defs, mode="temporary")
    sandbox.cleanup_test_kernel()
    if def_result['stderr']:
        logger.error(f"❌ Syntax Error in Definitions: {def_result['stderr']}")
        return {
            "error": f"Syntax Error during function definition:\n{def_result['stderr']}",
            "decision": "retry_create"
        }

    # ---------------------------------------------------------
    # 2단계: 하나씩 순회하며 유닛 테스트 실행
    # ---------------------------------------------------------
    failed_reports = []
    max_attempt = 3

    if history:
        history_summary = ""
        for item in history:
            history_summary += f"=== ❌ ATTEMPT (Source: {item['source'].upper()}) ===\n"
            history_summary += f"[Tool Code Used]:\n{item.get('tool_code', 'N/A')}\n\n"
            history_summary += f"[Test/Exec Code]:\n{item.get('test_code') or item.get('execution_code')}\n\n"
            history_summary += f"[Error Log]:\n{item['error_log']}\n"
            history_summary += f"==========================================================\n"
    
    for tool in tools:
        logger.info(f"   👉 Testing individual tool: {tool['name']}")
        
        # (A) 테스트 코드 생성 (이 도구 하나에만 집중)
        prompt = f"""
            Write a Python Unit Test for the function: `{tool['name']}`.
            The unit tests should test whether the function is logically correct as intended, not only the syntax.
            Function Code:
            {tool['code']}


            REQUIREMENTS:
            1. Create minimal dummy data to verify logic.
            2. Use `assert` statements.
            3. Include necessary imports.
            4. Include the tool code as is. Just call the function and assert the result.
            5. You MUST run the test with: `unittest.main(argv=[''], exit=False)`
        """

        if history:
            prompt += f"\n\nPrevious Attempts & Failures:\n{history_summary}"

        prompt += """
        
            OUTPUT FORMAT:
            <thought>rationale on the test case</thought>
            ```python
            # your code
            ```
            """
        
        prompt = dedent(prompt)
        test_code = llm.invoke(prompt).content

        if '```python' in test_code:
            test_code = test_code.split('```python')[1].split('```')[0]
        elif '```' in test_code: # fallback
            test_code = test_code.split('```')[1].split('```')[0]

        # print(f"\n{'='*20} [LLM RAW OUTPUT] {'='*20}")
        # print(test_code)
        # print(f"{'='*60}\n")
        
        # (B) 실행 (이미 함수들은 1단계에서 로드되었으므로 테스트 코드만 실행)
        try:
            result = sandbox.run_code(test_code, mode="temporary")
            sandbox.cleanup_test_kernel()
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"FATAL ERROR: {error_msg}")
            return {
                "decision": "retry_create",
                "error": error_msg
            }
        
        # (C) 결과 기록
        if result['stderr']:
            if "OK" in result['stderr'] and "FAILED" not in result['stderr']:
                logger.info(f"      ✅ Passed: {tool['name']}")
            else:
                history = state.get("feedback_history", [])
                current_feedback = {
                    "source": "tester",
                    "tool_code": tool['code'],
                    "test_code": test_code,
                    "error_log": result['stderr']
                }

                new_history = history + [current_feedback]

                if len(new_history) > 3:
                    new_history = new_history[-3:]

                logger.warning(f"      ❌ Failed: {tool['name']}")

                logger.debug(
                    f"--- Tool: {tool['name']} ---\n"
                    f"Error: {result['stderr']}\n"
                    f"Test Code Used:\n{test_code}\n"
                )
                failed_reports.append(
                    f"--- Tool: {tool['name']} ---\n"
                    f"Error: {result['stderr']}\n"
                    f"Test Code Used:\n{test_code}\n"
                )

                return {
                    "decision": "retry_create",
                    "feedback_history": new_history,
                    "error": result['stderr']
                }
        else:
            logger.info(f"      ✅ Passed: {tool['name']}")

    # ---------------------------------------------------------
    # 3단계: 결과 취합 및 결정
    # ---------------------------------------------------------
    if failed_reports:
        # 하나라도 실패하면 반려
        error_summary = "\n".join(failed_reports)
        return {
            "error": f"Unit Tests Failed for the following tools:\n{error_summary}",
            "decision": "retry_create",
            # 실패 로그를 컨텍스트에 추가
            "context_log": state['context_log'] + [f"Test Failures:\n{error_summary}"]
        }
    else:
        # 모두 통과!
        return {
            "error": None,
            "decision": "solve"
        }


def solver_node(state: AgentState, sandbox: AgentSandbox):
    logger.info("Running solver...")
    """
    [Solver]
    실행 결과를 보고 다음 단계를 결정합니다.
    """
    plan = state['plan']
    current_idx = state['current_step_index']
    current_task = plan[current_idx]
    inventory = state.get("variable_inventory", {})
    final_context = sandbox.get_final_context()
    tools = state.get("tool_generated", []) + state.get("tool_retrieved", [])
    
    # 1. 정의 로드 (이미 Tester나 이전 단계에서 했겠지만 안전하게 다시)
    all_defs = "\n\n".join([t['code'] for t in tools])
    tool_desc = "\n".join([f"- {t['name']}: {t['docstring']}" for t in tools])

    max_retries = 3
    temp_history = []

    for attempt in range(max_retries):    
        # 2. 실전 실행 코드 생성
        prompt = (
            f"Task: {current_task}\nTools:\n{tool_desc}\nVariables currently in memory: {inventory}\n"
            f"Write code to solve the task using real data variables.\n"
            f"Save result to new variable."
            f"Include necessary imports."
        )

        # solver 내부 loop; 이전 실패 로그 추가
        if temp_history:
            prompt += "\n\n🚫 PREVIOUS FAILED ATTEMPTS (LEARN FROM MISTAKES):\n"
            for h in temp_history:
                prompt += f"- Code:\n{h['exec_code']}\n"
                prompt += f"- Error:\n{h['error']}\n\n"
            prompt += "🚨 ERROR ANALYSIS: The tool code is fixed. Focus on fixing YOUR calling arguments or logic."

        exec_code = llm.invoke(prompt).content

        if '```python' in exec_code:
            exec_code = exec_code.split('```python')[1].split('```')[0]
        elif '```' in exec_code: # fallback
            exec_code = exec_code.split('```')[1].split('```')[0]
        
        # 3. 실행
        full_code = f"{all_defs}\n\n# Execution\n{exec_code}"
        res = sandbox.run_code(full_code, mode="permanent")
    
        if res['stderr']:
            logger.error(f"❌ Solver Execution Failed: {res['stderr']}")
            logger.error(f"   Code used:\n{exec_code}")

            temp_history.append({
                "exec_code": exec_code,
                "error": res['stderr']
            })

            if attempt == max_retries - 1:
                logger.error("Solver max retries exceeded")

                history = state.get("feedback_history", [])
                current_feedback = {
                    "source": "solver",
                    "tool_code": all_defs,
                    "execution_code": exec_code,
                    "error_log": res['stderr']
                }

                new_history = history + [current_feedback]

                if len(new_history) > 3:
                    new_history = new_history[-3:]
                
                return {
                    "decision": "retry_create", 
                    "feedback_history": new_history,
                    "error": f"Runtime Error: {res['stderr']}"
                }
            
            continue
        
        # 4. 변수 업데이트 (Inspection)
        logger.info("✅Solver Execution Succeeded")
        inspect_code = "import json; print(json.dumps({k:type(v).__name__ for k,v in globals().items() if not k.startswith('_')}))"
        try:
            insp_res = sandbox.run_code(inspect_code, mode="permanent")
            new_inv = json.loads(insp_res['stdout'])
        except:
            new_inv = inventory

        # 5. 성공 시 도구 저장 (생성된 경우만)
        if state.get("tool_generated"):
            for t in state.get("tool_generated"):
                try:
                    memory.add_tool({
                        "name":t['name'],
                        "code":t['code'],
                        "docstring":t['docstring']
                    })
                except Exception as e:
                    logger.error(f"Failed to add tool to memory: {e}")

        # 6. 다음 스텝 판별
        next_idx = state['current_step_index'] + 1
        
        return {
            "decision": "continue", 
            "current_step_index": next_idx, 
            "variable_inventory": new_inv,
            "tool_generated": [],
            "tool_retrieved": [],
            "feedback_history": [],
            "error": None
        }


def final_answer_node(state: AgentState, sandbox: AgentSandbox):
    query = state['problem']
    inventory = state['variable_inventory'] # Solver들이 열심히 모은 결과값들
    final_context = sandbox.get_final_context()
    
    logger.info("🏁 Generating Final Answer...")

    # LLM에게 "자료 줄게, 답 써줘"라고 요청
    prompt = (
        f"You are a helpful Data Analyst Assistant.\n"
        f"Original Question: {query}\n\n"
        f"We have processed the data and executed the plan. "
        f"Here are the collected variables and their values:\n"
        f"{json.dumps(final_context, indent=2, default=str)}\n\n"
        f"MISSION:\n"
        f"1. Synthesize the information from the variables to answer the Original Question.\n"
        f"2. Be direct and concise.\n"
        f"3. If the answer is a specific number or list found in variables, provide it clearly.\n"
        f"4. Do NOT show python code or internal variable names in the final answer."
    )
    
    prompt = f"""You are a math expert. Your task is to answer the following question with your reasoning.
                The answer has to be in one of following formats: integer, float, complex number, (numeric) string, (LaTeX expression) string.
                You must put your answer in $\\boxed{{}}$

                Think step by step.
                ---
                ## Problem:
                {query}

                ## Here are some variables collected from intermediate steps. You can use these variables for your reasoning.
                {json.dumps(final_context, indent=2, default=str)}

                ## Reasoning:
                <Your step-by-step explanation>

                ## Answer:
                <Your answer>
            """
    
    prompt = dedent(prompt)
    
    response = llm.invoke(prompt).content

    print(response)
    
    return {"final_answer": response}