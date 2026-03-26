from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    valid_df = df[df["pred"].isin(["A", "B", "C", "D"])]

    cm = confusion_matrix(valid_df["gold"], valid_df["pred"], labels=["A", "B", "C", "D"])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A", "B", "C", "D"])
    disp.plot(ax=ax)
    plt.title(f"Confusion Matrix of {model_name}")
    plt.tight_layout()

    safe_name = model_name.replace(":", "_").replace(".", "_")
    save_path = OUTPUT_DIR / f"fig_confusion_{safe_name}_1000.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")