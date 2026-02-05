import os
import sys
import re
import json
import openai
from dotenv import load_dotenv
from sympy import parse_expr
from sympy.parsing.latex import parse_latex
from tenacity import retry, stop_after_attempt, wait_fixed

# It seems RESULT_DIR is not defined in this script. 
# I will assume it's in the current directory for now.
# If src.config exists, this should work, otherwise, it needs to be created.
try:
    from src.config import RESULT_DIR
except ImportError:
    RESULT_DIR = "."


def extract_answer(text: str) -> str:
    """
    Finds the last occurrence of \\boxed{...} and returns the content within it.
    If \\boxed{...} is not found, returns the original string.
    """
    # Find the last occurrence of \boxed{
    last_boxed_start = text.rfind("\\boxed{")

    if last_boxed_start == -1:
        # If \boxed{} is not found, return the original string
        return text

    # Find the start of the content inside \boxed{}
    content_start_index = last_boxed_start + len("\\boxed{")
    
    balance = 1
    current_content = ""
    
    # Scan from the content start to find the matching closing brace
    for i in range(content_start_index, len(text)):
        char = text[i]
        
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
        
        if balance == 0:
            return current_content
        
        current_content += char
        
    # If no matching brace is found, return the original text as a fallback.
    return text

def approx_equal(a, b, tol=1e-3):
    return abs(a - b) <= tol

def math_grading_function(ground_truth: str, answer: str):
    load_dotenv()

    # Initialize OpenRouter client
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    ground_truth_converted = None
    answer_converted = None
    try:
        try:
            answer_converted = int(answer)
        except (ValueError, TypeError):
            try:
                answer_converted = float(answer)
            except (ValueError, TypeError):
                try:
                    answer_converted = parse_latex(answer).evalf()
                except Exception:
                    raise Exception('Rule-based evluation failed: model answer')

        try:
            ground_truth_converted = int(ground_truth)
        except (ValueError, TypeError):
            try:
                ground_truth_converted = float(ground_truth)
            except (ValueError, TypeError):
                try:
                    ground_truth_converted = parse_latex(ground_truth).evalf()
                except Exception:
                    raise Exception('Rule-based evluation failed: ground truth')
        
        if ground_truth_converted is not None and answer_converted is not None:
            if approx_equal(ground_truth_converted, answer_converted):
                return True, answer_converted
    except Exception as e:
        print(f"Rule-based evaluation failed: {e}")

    llm_as_a_judge_prompt = """You are a grading assistant evaluating whether a LLM-generated answer is equivalent to a reference answer.
You are provided with:
- A reference answer (ground truth)
- A generated answer from a model

Your goal is to judge whether the generated answer is **acceptable and correct** in context, even if it's phrased differently from the reference.

### Guidelines:
- Be strict if the generated answer is incomplete or factually wrong.
- Provide a short justification for your decision.

---
## Reference Answer:
{REFERENCE_ANSWER}

## Generated Answer:
{GENERATED_ANSWER}

---

## Respond strictly in this JSON format:

```json
{{
  "correct": true or false,
  "justification": "Concise explanation of your reasoning (1-3 sentences)"
}}
```
Only output valid JSON. Do not include any other text."""

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    def get_llm_judge():
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "user", "content": llm_as_a_judge_prompt.format(REFERENCE_ANSWER=ground_truth, GENERATED_ANSWER=answer)}
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    try:
        response_text = get_llm_judge()
        # Extract JSON from the response
        json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        if json_match:
            llm_as_a_judge_result = json.loads(json_match.group(1))
            return llm_as_a_judge_result['correct'], answer
        else:
            # Fallback if no JSON block is found
            return False, answer
    except Exception as e:
        print(f"LLM-based evaluation failed after retries: {e}")
        return False, answer


if __name__ == "__main__":
    # Assuming math_100_result.json is in the RESULT_DIR
    input_filepath = os.path.join(RESULT_DIR, "math_100_result.json")
    output_filepath = os.path.join(RESULT_DIR, "math_100_graded.json")

    try:
        with open(input_filepath, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {input_filepath} was not found.")
        # Create a dummy file for demonstration if it doesn't exist
        print("Creating a dummy 'math_100_result.json' file for demonstration.")
        dummy_data = {
            "1": {
                "answer": "42",
                "domain": "Arithmetic",
                "model_answer": "The final answer is \\boxed{42}."
            },
            "2": {
                "answer": "x=5",
                "domain": "Algebra",
                "model_answer": "After solving the equation, we get \\boxed{x=5}"
            },
             "3": {
                "answer": "3.14",
                "domain": "Geometry",
                "model_answer": "The area is approximately 3.14."
            }
        }
        with open(input_filepath, 'w') as f:
            json.dump(dummy_data, f, indent=4)
        results = dummy_data

    graded_results = {}
    for i, data in results.items():
        ground_truth = data["answer"]
        model_output = data["model_answer"]
        
        # 1. Extract the answer from the model's output
        extracted_ans = extract_answer(model_output)
        
        # 2. Grade the extracted answer
        is_correct, justification = math_grading_function(ground_truth, extracted_ans)
        
        graded_results[i] = {
            "domain": data["domain"],
            "ground_truth": ground_truth,
            "model_output": model_output,
            "extracted_answer": extracted_ans,
            "is_correct": is_correct,
            "justification": justification
        }
        
        print(f"ID: {i}")
        print(f"  Domain: {data['domain']}")
        print(f"  Model Output: {model_output}")
        print(f"  Extracted Answer: {extracted_ans}")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Is Correct: {is_correct}")
        print("-" * 20)

    with open(output_filepath, 'w') as f:
        json.dump(graded_results, f, indent=4)
        
    print(f"Grading complete. Results saved to {output_filepath}")