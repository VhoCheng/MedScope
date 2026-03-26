from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

# ===== 路径 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ===== 你的6个模型 =====
files = {
    "llama3.2:1b": OUTPUT_DIR / "medqa_results_llama3_2_1b_1000.csv",
    "llama3.2:3b": OUTPUT_DIR / "medqa_results_llama3_2_3b_1000.csv",
    "qwen2.5:1.5b": OUTPUT_DIR / "medqa_results_qwen2_5_1_5b_1000.csv",
    "qwen2.5:3b": OUTPUT_DIR / "medqa_results_qwen2_5_3b_1000.csv",
    "gemma3:1b": OUTPUT_DIR / "medqa_results_gemma3_1b_1000.csv",
    "gemma3:4b": OUTPUT_DIR / "medqa_results_gemma3_4b_1000.csv",
}

# ===== 读取数据 =====
dfs = {}
for name, path in files.items():
    dfs[name] = pd.read_csv(path)

# =====================================================
# 🔥 1. 顶刊级热力图（Model × Subject）
# =====================================================

subject_matrix = {}

for name, df in dfs.items():
    subject_acc = df.groupby("subject_name")["correct"].mean()
    subject_matrix[name] = subject_acc

heat_df = pd.DataFrame(subject_matrix).fillna(0)

plt.figure(figsize=(12, 8))
plt.imshow(heat_df.T, aspect='auto')
plt.colorbar(label="Accuracy")

plt.yticks(range(len(heat_df.columns)), heat_df.columns)
plt.xticks(range(len(heat_df.index)), heat_df.index, rotation=45, ha="right")

plt.title("Model vs Subject Accuracy Heatmap", fontsize=14)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "fig_heatmap_model_subject.png", dpi=300)
plt.show()


# =====================================================
# 🔥 2. 顶刊气泡图（Accuracy × Time × Model）
# =====================================================

accs = []
times = []
names = []

for name, df in dfs.items():
    valid_df = df[df["pred"].isin(["A","B","C","D"])]
    acc = accuracy_score(valid_df["gold"], valid_df["pred"])
    time = df["time_sec"].mean()

    accs.append(acc)
    times.append(time)
    names.append(name)

# 气泡大小（用accuracy放大）
sizes = [a * 2000 for a in accs]

plt.figure(figsize=(10, 6))
scatter = plt.scatter(times, accs, s=sizes, alpha=0.6)

for i, txt in enumerate(names):
    plt.annotate(txt, (times[i], accs[i]))

plt.xlabel("Avg Time per Question (s)")
plt.ylabel("Accuracy")
plt.title("Efficiency vs Performance Trade-off")

plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_bubble_tradeoff.png", dpi=300)
plt.show()


# =====================================================
# 🔥 3. 箱线图（稳定性分析）
# =====================================================

data = []
labels = []

for name, df in dfs.items():
    data.append(df["correct"])
    labels.append(name)

plt.figure(figsize=(12, 6))
plt.boxplot(data, labels=labels)

plt.ylabel("Correctness Distribution (0/1)")
plt.title("Model Stability Comparison (Boxplot)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_boxplot_stability.png", dpi=300)
plt.show()


# =====================================================
# 🔥 4. 伪3D图（Accuracy × Time × Invalid Rate）
# =====================================================

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for name, df in dfs.items():
    valid_df = df[df["pred"].isin(["A","B","C","D"])]
    acc = accuracy_score(valid_df["gold"], valid_df["pred"])
    time = df["time_sec"].mean()
    invalid_rate = 1 - len(valid_df) / len(df)

    ax.scatter(time, acc, invalid_rate, label=name)

ax.set_xlabel("Time")
ax.set_ylabel("Accuracy")
ax.set_zlabel("Invalid Rate")

ax.set_title("3D Trade-off: Performance vs Efficiency vs Reliability")

ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_3d_tradeoff.png", dpi=300)
plt.show()


print("\n🔥 ALL ADVANCED FIGURES GENERATED!")