from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "llama3.2:1b": OUTPUT_DIR / "medqa_results_llama3_2_1b_1000.csv",
    "llama3.2:3b": OUTPUT_DIR / "medqa_results_llama3_2_3b_1000.csv",
    "qwen2.5:1.5b": OUTPUT_DIR / "medqa_results_qwen2_5_1_5b_1000.csv",
    "qwen2.5:3b": OUTPUT_DIR / "medqa_results_qwen2_5_3b_1000.csv",
    "gemma3:1b": OUTPUT_DIR / "medqa_results_gemma3_1b_1000.csv",
    "gemma3:4b": OUTPUT_DIR / "medqa_results_gemma3_4b_1000.csv",
}

for model_name, file_path in files.items():
    if not file_path.exists():
        print(f"Skipping missing file: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # 取样本数至少10的学科，避免太稀疏
    subject_stats = df.groupby("subject_name").agg(
        accuracy=("correct", "mean"),
        count=("correct", "count")
    )

    subject_stats = subject_stats[subject_stats["count"] >= 10]
    subject_stats = subject_stats.sort_values(by="accuracy", ascending=False).head(10)

    safe_name = model_name.replace(":", "_").replace(".", "_")
    plt.figure(figsize=(10, 6))
    plt.bar(subject_stats.index, subject_stats["accuracy"])
    plt.ylabel("Accuracy")
    plt.title(f"Top-10 Subject-wise Accuracy ({model_name})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fig_subject_accuracy_{safe_name}_1000.png", dpi=300)
    plt.show()

    print(f"Saved: {OUTPUT_DIR / f'fig_subject_accuracy_{safe_name}_1000.png'}")