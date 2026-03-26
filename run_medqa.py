import json
import re
import time
import os
import sys
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

INPUT_FILE = "src/data/medqa_sample_1000.json"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace(":", "_").replace(".", "_").replace("/", "_")


def call_ollama(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"].strip()


def build_prompt(item: dict) -> str:
    return f"""You are a careful medical AI assistant.
Please answer the following multiple-choice medical question.
Return only one letter: A, B, C, or D.

Question: {item['question']}
A. {item['options']['A']}
B. {item['options']['B']}
C. {item['options']['C']}
D. {item['options']['D']}
"""


def extract_choice(text: str) -> str:
    text = text.strip().upper()

    if text in ["A", "B", "C", "D"]:
        return text

    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)

    match = re.search(r"ANSWER\s*[:：]?\s*([ABCD])", text)
    if match:
        return match.group(1)

    match = re.search(r"THE ANSWER IS\s*([ABCD])", text)
    if match:
        return match.group(1)

    return "INVALID"


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python src/run_medqa.py <model_name>")

    model_name = sys.argv[1]
    model_tag = sanitize_model_name(model_name)
    output_file = f"outputs/medqa_results_{model_tag}_1000.csv"

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    os.makedirs("outputs", exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions from {INPUT_FILE}")
    print(f"Running model: {model_name}")
    print(f"Output file: {output_file}")

    rows = []

    for item in tqdm(data, desc=f"Running {model_name}"):
        prompt = build_prompt(item)

        start_time = time.time()
        raw_output = call_ollama(model_name, prompt)
        elapsed = time.time() - start_time

        pred = extract_choice(raw_output)
        gold = item["answer"]

        rows.append({
            "id": item["id"],
            "question": item["question"],
            "option_A": item["options"]["A"],
            "option_B": item["options"]["B"],
            "option_C": item["options"]["C"],
            "option_D": item["options"]["D"],
            "gold": gold,
            "pred": pred,
            "correct": int(pred == gold),
            "time_sec": round(elapsed, 4),
            "raw_output": raw_output,
            "subject_name": item.get("subject_name", ""),
            "topic_name": item.get("topic_name", "")
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    valid_df = df[df["pred"].isin(["A", "B", "C", "D"])]

    if len(valid_df) > 0:
        acc = accuracy_score(valid_df["gold"], valid_df["pred"])
        f1 = f1_score(valid_df["gold"], valid_df["pred"], average="macro")
    else:
        acc = 0.0
        f1 = 0.0

    invalid_rate = 1 - len(valid_df) / len(df)
    avg_time = df["time_sec"].mean()

    print("\n===== RESULT SUMMARY =====")
    print(f"Model: {model_name}")
    print(f"Total samples: {len(df)}")
    print(f"Valid predictions: {len(valid_df)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(f"Invalid rate: {invalid_rate:.4f}")
    print(f"Average time/question: {avg_time:.4f} seconds")
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()