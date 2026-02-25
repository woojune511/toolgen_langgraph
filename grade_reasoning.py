"""
Reasoning Pipeline 결과 채점 스크립트.
reasoning_full_results.json → reasoning_graded.json
"""
import json
import os
from grade_math import math_grading_function, extract_answer

INPUT_FILE = "reasoning_full_results.json"
OUTPUT_FILE = "reasoning_graded.json"


def main():
    with open(INPUT_FILE) as f:
        results = json.load(f)

    # Load existing graded results for resume
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            graded = json.load(f)
    else:
        graded = {}

    total = len(results)
    already = len(graded)
    print(f"Total: {total}, Already graded: {already}")

    from collections import Counter
    domain_correct = Counter()
    domain_total = Counter()

    for idx_str, data in sorted(results.items(), key=lambda x: int(x[0])):
        domain = data.get("domain", "Unknown")
        gt = data.get("ground_truth", "")
        final_answer = data.get("final_answer", "")
        
        if idx_str in graded:
            # Already graded
            domain_total[domain] += 1
            if graded[idx_str].get("is_correct"):
                domain_correct[domain] += 1
            continue

        # Extract answer from final_answer (may contain boxed)
        extracted = extract_answer(final_answer) if final_answer else ""

        # Grade
        is_correct, _ = math_grading_function(gt, extracted)

        graded[idx_str] = {
            "domain": domain,
            "ground_truth": gt,
            "final_answer": final_answer,
            "extracted_answer": extracted,
            "is_correct": is_correct,
            "attempts": data.get("attempts", 0),
            "cot_answer": data.get("cot_answer", ""),
            "code_result": data.get("code_result", ""),
        }

        domain_total[domain] += 1
        if is_correct:
            domain_correct[domain] += 1

        i = int(idx_str)
        if (i + 1) % 50 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(graded, f, indent=2, default=str)
            done = len(graded)
            correct_so_far = sum(1 for v in graded.values() if v.get("is_correct"))
            print(f"  [{done}/{total}] Correct so far: {correct_so_far} ({correct_so_far/done*100:.1f}%)")

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(graded, f, indent=2, default=str)

    # Print summary
    correct = sum(1 for v in graded.values() if v.get("is_correct"))
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total*100:.1f}%")
    print(f"\nPer Domain:")
    for d in sorted(domain_total.keys()):
        t = domain_total[d]
        c = domain_correct[d]
        print(f"  {d}: {c}/{t} = {c/t*100:.1f}%")


if __name__ == "__main__":
    main()
