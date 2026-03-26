import json
import random
import os
from datasets import load_dataset

random.seed(42)


def main():
    print("Loading MedMCQA dataset...")
    dataset = load_dataset("openlifescienceai/medmcqa")

    print(dataset)

    split_name = None
    for candidate in ["validation", "test", "train"]:
        if candidate in dataset:
            split_name = candidate
            break

    if split_name is None:
        raise ValueError("No usable split found.")

    data = dataset[split_name]
    print(f"Using split: {split_name}, size={len(data)}")

    processed = []
    for idx, ex in enumerate(data):
        question = str(ex.get("question", "")).strip()
        opa = str(ex.get("opa", "")).strip()
        opb = str(ex.get("opb", "")).strip()
        opc = str(ex.get("opc", "")).strip()
        opd = str(ex.get("opd", "")).strip()

        answer_map = {
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
            1: "A",
            2: "B",
            3: "C",
            4: "D"
        }
        raw_cop = ex.get("cop")
        answer_letter = answer_map.get(raw_cop, answer_map.get(str(raw_cop), ""))

        subject_name = str(ex.get("subject_name", "")).strip()
        topic_name = str(ex.get("topic_name", "")).strip()

        if question and opa and opb and opc and opd and answer_letter in {"A", "B", "C", "D"}:
            processed.append({
                "id": str(ex.get("id", idx)),
                "question": question,
                "options": {
                    "A": opa,
                    "B": opb,
                    "C": opc,
                    "D": opd
                },
                "answer": answer_letter,
                "subject_name": subject_name,
                "topic_name": topic_name
            })

    print(f"Usable examples: {len(processed)}")

    if len(processed) < 1000:
        raise ValueError(f"Not enough usable examples for 1000-question sampling. Found: {len(processed)}")

    sample = random.sample(processed, 1000)

    os.makedirs("data", exist_ok=True)

    output_path = "data/medqa_sample_1000.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()