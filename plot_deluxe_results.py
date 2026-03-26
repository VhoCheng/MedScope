from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 基础设置
# =========================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titleweight"] = "bold"

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

dfs = {}
for model_name, file_path in files.items():
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    dfs[model_name] = pd.read_csv(file_path)

# =========================
# 计算汇总指标
# =========================
summary_rows = []
subject_matrix = {}

for model_name, df in dfs.items():
    valid_df = df[df["pred"].isin(["A", "B", "C", "D"])].copy()

    acc = accuracy_score(valid_df["gold"], valid_df["pred"])
    f1 = f1_score(valid_df["gold"], valid_df["pred"], average="macro")
    invalid_rate = 1 - len(valid_df) / len(df)
    avg_time = df["time_sec"].mean()

    summary_rows.append({
        "model": model_name,
        "accuracy": acc,
        "macro_f1": f1,
        "invalid_rate": invalid_rate,
        "avg_time_sec": avg_time,
    })

    subj = df.groupby("subject_name")["correct"].mean()
    subject_matrix[model_name] = subj

summary_df = pd.DataFrame(summary_rows)
subject_df = pd.DataFrame(subject_matrix).fillna(0)

# 只保留样本量稍微多一点的学科
subject_counts = next(iter(dfs.values())).groupby("subject_name").size()
valid_subjects = subject_counts[subject_counts >= 10].index
subject_df = subject_df.loc[subject_df.index.intersection(valid_subjects)]

# =========================
# 图1：聚类热力图（高级感最强）
# =========================
# 对学科和模型都做层次聚类
row_linkage = linkage(subject_df.values, method="average", metric="euclidean")
col_linkage = linkage(subject_df.T.values, method="average", metric="euclidean")

cg = sns.clustermap(
    subject_df,
    row_linkage=row_linkage,
    col_linkage=col_linkage,
    cmap="viridis",
    linewidths=0.2,
    figsize=(12, 12),
    cbar_kws={"label": "Accuracy"}
)
cg.fig.suptitle("Clustered Heatmap of Subject-wise Accuracy", y=1.02, fontsize=18, fontweight="bold")
cg.savefig(OUTPUT_DIR / "fig_deluxe_clustermap_subject_accuracy.png", dpi=300, bbox_inches="tight")
plt.close("all")

# =========================
# 图2：雷达图（多指标综合）
# =========================
# 指标方向统一：时间越小越好，所以做反向归一化
radar_df = summary_df.copy()

def minmax(x):
    if x.max() == x.min():
        return np.ones_like(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min())

radar_df["accuracy_n"] = minmax(radar_df["accuracy"])
radar_df["macro_f1_n"] = minmax(radar_df["macro_f1"])
radar_df["speed_n"] = 1 - minmax(radar_df["avg_time_sec"])  # 越快越高
radar_df["reliability_n"] = 1 - minmax(radar_df["invalid_rate"] + 1e-12)

categories = ["Accuracy", "Macro-F1", "Speed", "Reliability"]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

for _, row in radar_df.iterrows():
    values = [
        row["accuracy_n"],
        row["macro_f1_n"],
        row["speed_n"],
        row["reliability_n"]
    ]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=row["model"])
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title("Multi-metric Radar Comparison", pad=25, fontsize=18, fontweight="bold")
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_deluxe_radar_multimetric.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 图3：Pareto 气泡图（性能-效率-稳定性）
# =========================
plot_df = summary_df.copy()
plot_df["bubble_size"] = (plot_df["macro_f1"] + 0.05) * 4000

fig, ax = plt.subplots(figsize=(11, 8))
sns.scatterplot(
    data=plot_df,
    x="avg_time_sec",
    y="accuracy",
    size="bubble_size",
    sizes=(200, 2000),
    hue="macro_f1",
    palette="viridis",
    alpha=0.75,
    legend=False,
    ax=ax
)

for _, row in plot_df.iterrows():
    ax.annotate(
        row["model"],
        (row["avg_time_sec"], row["accuracy"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10
    )

# 简单 Pareto frontier（时间越小越好，准确率越高越好）
frontier = plot_df.sort_values("avg_time_sec")
pareto_x, pareto_y = [], []
best_acc = -1
for _, row in frontier.iterrows():
    if row["accuracy"] > best_acc:
        pareto_x.append(row["avg_time_sec"])
        pareto_y.append(row["accuracy"])
        best_acc = row["accuracy"]

ax.plot(pareto_x, pareto_y, linestyle="--", linewidth=2)
ax.set_xlabel("Average Time per Question (s)")
ax.set_ylabel("Accuracy")
ax.set_title("Pareto-style Trade-off: Efficiency vs Performance", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_deluxe_pareto_bubble.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 图4：小提琴图（正确性分布）
# =========================
violin_rows = []
for model_name, df in dfs.items():
    for value in df["correct"].tolist():
        violin_rows.append({"model": model_name, "correct": value})

violin_df = pd.DataFrame(violin_rows)

fig, ax = plt.subplots(figsize=(12, 7))
sns.violinplot(data=violin_df, x="model", y="correct", inner="quartile", cut=0, ax=ax)
ax.set_title("Distribution of Prediction Correctness Across Models", fontsize=18, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Correctness (0/1)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_deluxe_violin_correctness.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 图5：模型间一致性相关矩阵
# =========================
pred_matrix = {}
for model_name, df in dfs.items():
    pred_matrix[model_name] = df["pred"].astype(str)

pred_df = pd.DataFrame(pred_matrix)

# 将“预测是否相同”转成 pairwise agreement
models = list(pred_df.columns)
agreement = pd.DataFrame(index=models, columns=models, dtype=float)

for m1 in models:
    for m2 in models:
        agreement.loc[m1, m2] = (pred_df[m1] == pred_df[m2]).mean()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(agreement.astype(float), annot=True, fmt=".2f", cmap="mako", square=True, ax=ax)
ax.set_title("Pairwise Prediction Agreement Between Models", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_deluxe_model_agreement_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 图6：多面板总图（论文主图感）
# =========================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

# Panel A: Accuracy
ax1 = fig.add_subplot(gs[0, 0])
order_df = summary_df.sort_values("accuracy", ascending=False)
sns.barplot(data=order_df, x="model", y="accuracy", ax=ax1)
ax1.set_title("A. Accuracy Ranking", fontweight="bold")
ax1.set_xlabel("")
ax1.set_ylabel("Accuracy")
ax1.tick_params(axis="x", rotation=25)

# Panel B: Macro-F1
ax2 = fig.add_subplot(gs[0, 1])
order_df2 = summary_df.sort_values("macro_f1", ascending=False)
sns.barplot(data=order_df2, x="model", y="macro_f1", ax=ax2)
ax2.set_title("B. Macro-F1 Ranking", fontweight="bold")
ax2.set_xlabel("")
ax2.set_ylabel("Macro-F1")
ax2.tick_params(axis="x", rotation=25)

# Panel C: Time
ax3 = fig.add_subplot(gs[1, 0])
order_df3 = summary_df.sort_values("avg_time_sec", ascending=True)
sns.barplot(data=order_df3, x="model", y="avg_time_sec", ax=ax3)
ax3.set_title("C. Inference Efficiency", fontweight="bold")
ax3.set_xlabel("")
ax3.set_ylabel("Avg Time (s)")
ax3.tick_params(axis="x", rotation=25)

# Panel D: Bubble
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(plot_df["avg_time_sec"], plot_df["accuracy"], s=plot_df["bubble_size"], alpha=0.65)
for _, row in plot_df.iterrows():
    ax4.annotate(row["model"], (row["avg_time_sec"], row["accuracy"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
ax4.set_title("D. Trade-off Landscape", fontweight="bold")
ax4.set_xlabel("Avg Time (s)")
ax4.set_ylabel("Accuracy")

fig.suptitle("Comprehensive Benchmarking of Lightweight Medical LLMs", fontsize=20, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_deluxe_multipanel_main.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGenerated deluxe figures:")
for p in [
    "fig_deluxe_clustermap_subject_accuracy.png",
    "fig_deluxe_radar_multimetric.png",
    "fig_deluxe_pareto_bubble.png",
    "fig_deluxe_violin_correctness.png",
    "fig_deluxe_model_agreement_heatmap.png",
    "fig_deluxe_multipanel_main.png",
]:
    print(OUTPUT_DIR / p)