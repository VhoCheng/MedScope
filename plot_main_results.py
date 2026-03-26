from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_path = OUTPUT_DIR / "model_summary_1000.csv"
df = pd.read_csv(summary_path)

# 1. Accuracy
plt.figure(figsize=(10, 6))
plt.bar(df["model"], df["accuracy"])
plt.ylabel("Accuracy")
plt.title("Accuracy of Different Models on Medical MCQs")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_accuracy_1000.png", dpi=300)
plt.show()

# 2. Macro-F1
plt.figure(figsize=(10, 6))
plt.bar(df["model"], df["macro_f1"])
plt.ylabel("Macro-F1")
plt.title("Macro-F1 of Different Models on Medical MCQs")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_macro_f1_1000.png", dpi=300)
plt.show()

# 3. Average Time
plt.figure(figsize=(10, 6))
plt.bar(df["model"], df["avg_time_sec"])
plt.ylabel("Average Time per Question (s)")
plt.title("Inference Efficiency of Different Models")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_time_1000.png", dpi=300)
plt.show()

print("Saved figures:")
print(OUTPUT_DIR / "fig_accuracy_1000.png")
print(OUTPUT_DIR / "fig_macro_f1_1000.png")
print(OUTPUT_DIR / "fig_time_1000.png")