from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# 基础设置
# =========================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

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
# 汇总指标
# =========================
summary_rows = []
subject_matrix = {}
pred_matrix = {}

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

    subject_acc = df.groupby("subject_name")["correct"].mean()
    subject_matrix[model_name] = subject_acc
    pred_matrix[model_name] = df["pred"].astype(str).values

summary_df = pd.DataFrame(summary_rows)
subject_df = pd.DataFrame(subject_matrix).fillna(0)

# 过滤低频subject
subject_counts = next(iter(dfs.values())).groupby("subject_name").size()
valid_subjects = subject_counts[subject_counts >= 10].index
subject_df = subject_df.loc[subject_df.index.intersection(valid_subjects)]

# =====================================================
# 1. 弦图风格网络图（模型一致性网络）
# =====================================================
models = list(pred_matrix.keys())
agreement = pd.DataFrame(index=models, columns=models, dtype=float)

pred_df = pd.DataFrame(pred_matrix)

for m1 in models:
    for m2 in models:
        agreement.loc[m1, m2] = (pred_df[m1] == pred_df[m2]).mean()

G = nx.Graph()
for m in models:
    G.add_node(m)

for i, m1 in enumerate(models):
    for j, m2 in enumerate(models):
        if j > i:
            w = agreement.loc[m1, m2]
            if w >= 0.32:  # 阈值稍微放低一点，图更复杂
                G.add_edge(m1, m2, weight=w)

pos = nx.circular_layout(G)
plt.figure(figsize=(10, 10))

edges = G.edges(data=True)
weights = [d["weight"] * 8 for (_, _, d) in edges]

nx.draw_networkx_nodes(G, pos, node_size=3000, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=11)
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)

edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

plt.title("Agreement Network Across Models", fontsize=18, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_ultra_agreement_network.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# 2. Sankey图（gold -> pred，选最强模型）
# =====================================================
best_model = summary_df.sort_values("accuracy", ascending=False).iloc[0]["model"]
best_df = dfs[best_model].copy()

labels = ["Gold A", "Gold B", "Gold C", "Gold D", "Pred A", "Pred B", "Pred C", "Pred D"]
label_to_idx = {lab: i for i, lab in enumerate(labels)}

flow = []
for g in ["A", "B", "C", "D"]:
    for p in ["A", "B", "C", "D"]:
        count = ((best_df["gold"] == g) & (best_df["pred"] == p)).sum()
        flow.append({
            "source": label_to_idx[f"Gold {g}"],
            "target": label_to_idx[f"Pred {p}"],
            "value": int(count)
        })

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=20,
        thickness=20,
        label=labels,
    ),
    link=dict(
        source=[x["source"] for x in flow],
        target=[x["target"] for x in flow],
        value=[x["value"] for x in flow],
    )
)])
fig.update_layout(
    title_text=f"Sankey Flow of Gold-to-Prediction ({best_model})",
    font_size=12,
    width=1100,
    height=700
)
fig.write_image(str(OUTPUT_DIR / "fig_ultra_sankey_gold_pred.png"))

# =====================================================
# 3. 3D气泡图（Accuracy × F1 × Time）
# =====================================================
fig = go.Figure()

for _, row in summary_df.iterrows():
    fig.add_trace(go.Scatter3d(
        x=[row["avg_time_sec"]],
        y=[row["accuracy"]],
        z=[row["macro_f1"]],
        mode="markers+text",
        text=[row["model"]],
        textposition="top center",
        marker=dict(
            size=18 + row["accuracy"] * 50,
            opacity=0.8,
            color=row["macro_f1"],
            colorscale="Viridis",
            showscale=False
        ),
        name=row["model"]
    ))

fig.update_layout(
    title="3D Trade-off Landscape of Lightweight Medical LLMs",
    scene=dict(
        xaxis_title="Avg Time (s)",
        yaxis_title="Accuracy",
        zaxis_title="Macro-F1",
    ),
    width=1100,
    height=800,
    showlegend=False
)
fig.write_image(str(OUTPUT_DIR / "fig_ultra_3d_bubble.png"))

# =====================================================
# 4. 手工版 Ridgeline（subject-wise accuracy distribution）
# =====================================================
ridge_rows = []
for model_name, df in dfs.items():
    subj_acc = df.groupby("subject_name")["correct"].mean().reset_index()
    subj_acc["model"] = model_name
    ridge_rows.append(subj_acc)

ridge_df = pd.concat(ridge_rows, ignore_index=True)

models_order = list(dfs.keys())
fig, axes = plt.subplots(len(models_order), 1, figsize=(12, 10), sharex=True)

if len(models_order) == 1:
    axes = [axes]

for i, model_name in enumerate(models_order):
    ax = axes[i]
    sub = ridge_df[ridge_df["model"] == model_name]

    # KDE 曲线
    sns.kdeplot(
        data=sub,
        x="correct",
        fill=True,
        alpha=0.8,
        linewidth=1.5,
        ax=ax
    )

    ax.text(
        0.01, 0.55,
        model_name,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylabel("")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if i != len(models_order) - 1:
        ax.set_xlabel("")
        ax.set_xticklabels([])

fig.suptitle("Ridgeline-style Distribution of Subject-wise Accuracy", fontsize=18, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_ultra_ridgeline_subject_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# 5. 极坐标玫瑰图（top subjects profile）
# =====================================================
# 取整体平均最有区分度的前8个subject
subj_var = subject_df.var(axis=1).sort_values(ascending=False).head(8).index
rose_df = subject_df.loc[subj_var]

angles = np.linspace(0, 2 * np.pi, len(rose_df.index), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(11, 11))
ax = plt.subplot(111, polar=True)

for model_name in rose_df.columns:
    values = rose_df[model_name].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=model_name)
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(rose_df.index)
ax.set_title("Polar Subject Profile of Models", fontsize=18, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_ultra_polar_subject_profile.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# 6. 重新优化后的 6-panel 主图（防重叠版）
# =====================================================

import textwrap

# ------- 统一短标签，避免模型名太长 -------
short_name_map = {
    "llama3.2:1b": "LLaMA-1B",
    "llama3.2:3b": "LLaMA-3B",
    "qwen2.5:1.5b": "Qwen-1.5B",
    "qwen2.5:3b": "Qwen-3B",
    "gemma3:1b": "Gemma-1B",
    "gemma3:4b": "Gemma-4B",
}

summary_plot_df = summary_df.copy()
summary_plot_df["model_short"] = summary_plot_df["model"].map(short_name_map)

agreement_plot = agreement.copy()
agreement_plot.index = [short_name_map.get(x, x) for x in agreement_plot.index]
agreement_plot.columns = [short_name_map.get(x, x) for x in agreement_plot.columns]

mini_df = subject_df.loc[subject_df.var(axis=1).sort_values(ascending=False).head(8).index].copy()
mini_df.columns = [short_name_map.get(x, x) for x in mini_df.columns]

# 全局稍微小一点
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig = plt.figure(figsize=(16, 12), constrained_layout=False)
gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.30)

# A Accuracy
ax1 = fig.add_subplot(gs[0, 0])
tmp = summary_plot_df.sort_values("accuracy", ascending=False)
sns.barplot(data=tmp, x="model_short", y="accuracy", ax=ax1, color="#4c72b0")
ax1.set_title("A. Accuracy Ranking", fontweight="bold", pad=8)
ax1.set_xlabel("")
ax1.set_ylabel("Accuracy")
ax1.tick_params(axis="x", rotation=20)
ax1.set_ylim(0, max(tmp["accuracy"]) * 1.15)

# B Macro-F1
ax2 = fig.add_subplot(gs[0, 1])
tmp = summary_plot_df.sort_values("macro_f1", ascending=False)
sns.barplot(data=tmp, x="model_short", y="macro_f1", ax=ax2, color="#55a868")
ax2.set_title("B. Macro-F1 Ranking", fontweight="bold", pad=8)
ax2.set_xlabel("")
ax2.set_ylabel("Macro-F1")
ax2.tick_params(axis="x", rotation=20)
ax2.set_ylim(0, max(tmp["macro_f1"]) * 1.15)

# C Time
ax3 = fig.add_subplot(gs[1, 0])
tmp = summary_plot_df.sort_values("avg_time_sec", ascending=True)
sns.barplot(data=tmp, x="model_short", y="avg_time_sec", ax=ax3, color="#c44e52")
ax3.set_title("C. Inference Efficiency", fontweight="bold", pad=8)
ax3.set_xlabel("")
ax3.set_ylabel("Avg Time (s)")
ax3.tick_params(axis="x", rotation=20)
ax3.set_ylim(0, max(tmp["avg_time_sec"]) * 1.15)

# D Agreement heatmap
ax4 = fig.add_subplot(gs[1, 1])
sns.heatmap(
    agreement_plot.astype(float),
    annot=True,
    fmt=".2f",
    cmap="mako",
    square=True,
    ax=ax4,
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.85}
)
ax4.set_title("D. Model Agreement Matrix", fontweight="bold", pad=8)
ax4.tick_params(axis="x", rotation=30)
ax4.tick_params(axis="y", rotation=0)

# E Pareto bubble
ax5 = fig.add_subplot(gs[2, 0])
pareto_df = summary_plot_df.copy()
sizes = (pareto_df["macro_f1"] + 0.05) * 3500

ax5.scatter(
    pareto_df["avg_time_sec"],
    pareto_df["accuracy"],
    s=sizes,
    alpha=0.65,
    color="#4c72b0",
    edgecolor="black",
    linewidth=0.6
)

# 给注释加偏移，减少重叠
offsets = {
    "LLaMA-1B": (6, 6),
    "LLaMA-3B": (6, 6),
    "Qwen-1.5B": (6, 6),
    "Qwen-3B": (6, -10),
    "Gemma-1B": (6, 6),
    "Gemma-4B": (6, 6),
}

for _, row in pareto_df.iterrows():
    dx, dy = offsets.get(row["model_short"], (5, 5))
    ax5.annotate(
        row["model_short"],
        (row["avg_time_sec"], row["accuracy"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8
    )

ax5.set_xlabel("Avg Time (s)")
ax5.set_ylabel("Accuracy")
ax5.set_title("E. Pareto Trade-off", fontweight="bold", pad=8)
ax5.grid(True, alpha=0.3)

# F Subject heatmap mini
ax6 = fig.add_subplot(gs[2, 1])
sns.heatmap(
    mini_df,
    cmap="viridis",
    ax=ax6,
    annot=False,
    cbar_kws={"shrink": 0.85}
)
ax6.set_title("F. Top-variable Subject Heatmap", fontweight="bold", pad=8)
ax6.tick_params(axis="x", rotation=25)
ax6.tick_params(axis="y", rotation=0, labelsize=8)

fig.suptitle(
    "Ultra-Deluxe Benchmark Dashboard for Lightweight Medical LLMs",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

# 手动调整边距，防止标题和坐标轴挤压
plt.subplots_adjust(
    top=0.90,
    bottom=0.07,
    left=0.08,
    right=0.97,
    hspace=0.60,
    wspace=0.30
)

plt.savefig(OUTPUT_DIR / "fig_ultra_dashboard_6panel_v2.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", OUTPUT_DIR / "fig_ultra_dashboard_6panel_v2.png")