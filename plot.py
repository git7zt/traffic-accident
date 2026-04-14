# -*- coding: utf-8 -*-
"""
plot.py
Visualization ONLY (final merged version).
No model training, no SHAP recomputation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb
import joblib


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

plt.rcParams["font.family"] = "Arial"

# ======================================================
# 0. Paths
# ======================================================
DATA_DIR = r"E:\jiedan\交通事故\data\output"
RESULT_DIR = r"E:\jiedan\交通事故\results\two_stage_soft"
MID_DIR = os.path.join(RESULT_DIR, "MID_result")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ======================================================
# 1. Load data
# ======================================================
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

with open(os.path.join(DATA_DIR, "feature_names.json"), "r") as f:
    feature_names = json.load(f)

# probabilities & predictions
proba_3 = np.load(os.path.join(MID_DIR, "proba_3.npy"))
y_pred = np.load(os.path.join(MID_DIR, "y_pred.npy"))
p_severe = np.load(os.path.join(MID_DIR, "p_severe.npy"))

# SHAP final attribution
shap_fatal = np.load(os.path.join(MID_DIR, "shap_fatal_final.npy"))
shap_serious = np.load(os.path.join(MID_DIR, "shap_serious_final.npy"))
shap_slight = np.load(os.path.join(MID_DIR, "shap_slight_final.npy"))
shap_severe = shap_fatal + shap_serious

# ======================================================
# Feature name cleaning (remove 'remainder' prefix)
# ======================================================
clean_feature_names = []
for name in feature_names:
    if name.startswith("remainder__"):
        clean_feature_names.append(name.replace("remainder__", ""))
    else:
        clean_feature_names.append(name)

clean_feature_names = np.array(clean_feature_names)

# ======================================================
# 2. Confusion Matrix
# ======================================================
cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlOrRd",
    xticklabels=["Fatal", "Serious", "Slight"],
    yticklabels=["Fatal", "Serious", "Slight"]
)
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig_Confusion_Matrix.png"), dpi=300)
plt.close()


# ======================================================
# 3. ROC Curves
# ======================================================
y_test_bin = label_binarize(y_test, classes=[1, 2, 3])

fig, ax = plt.subplots(figsize=(7, 6))

for i, label in enumerate(["Fatal", "Serious", "Slight"]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba_3[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.4)

# 关键：固定范围 + 去掉轴内边距（避免左下/右上留白）
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.margins(x=0, y=0)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", frameon=True)

# 关键：尽量压缩画布留白（但不影响坐标轴文字）
fig.tight_layout(pad=0.2)

# 关键：保存时裁剪外圈空白
fig.savefig(
    os.path.join(FIG_DIR, "Fig_ROC_Curve.png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.0
)
plt.close(fig)


# ======================================================
# 4. Feature Correlation Matrix (Top-Variance Features)
# ======================================================
def plot_feature_correlation_matrix(X, feature_names, save_path, top_n=30):
    """
    Plot feature correlation matrix using top-N features ranked by variance.
    """
    df = pd.DataFrame(X, columns=feature_names)

    # 1. Select top-N features by variance
    feature_var = df.var().sort_values(ascending=False)
    top_features = feature_var.head(top_n).index.tolist()

    # 2. Correlation matrix (subset)
    corr_subset = df[top_features].corr()

    # 3. Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_subset,
        annot=True,  # ✅ 显示数字
        fmt='.2f',  # ✅ 格式化为小数点后2位
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 8}  # ✅ 调整数字字体大小（可选）
    )

    #plt.title(
        #f"Feature Correlation Matrix (Top {top_n} Variance Features)",
        #fontsize=14,
        #pad=20
    #)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---- Use FULL dataset for correlation (recommended) ----
X_all = np.vstack([
    np.load(os.path.join(DATA_DIR, "X_train.npy")),
    np.load(os.path.join(DATA_DIR, "X_val.npy")),
    X_test
])

plot_feature_correlation_matrix(
    X_all,
    clean_feature_names,
    save_path=os.path.join(FIG_DIR, "Fig_Feature_Correlation_Matrix.png"),
    top_n=30
)


# ======================================================
# 4. PR Curves
# ======================================================
for i, label in enumerate(["Fatal", "Serious", "Slight"]):
    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i], proba_3[:, i]
    )
    ap = average_precision_score(y_test_bin[:, i], proba_3[:, i])

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve – {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"Fig_PR_Curve_{label}.png"), dpi=300)
    plt.close()


# ======================================================
# 5. SHAP Summary Plots
# ======================================================

# 5.1 标准 SHAP Summary（四个类别）
def plot_shap_summary(shap_values, title, filename):
    n_samples = shap_values.shape[0]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=clean_feature_names,
        max_display=20,
        show=False,
        plot_size=None,
        alpha=0.5
    )
    #plt.title(f"{title}\n(n = {n_samples:,} samples)", fontsize=12)
    plt.xlabel("SHAP value (impact on model output)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ {filename} saved with {n_samples:,} samples")


# 绘制四个基础 SHAP Summary
plot_shap_summary(shap_fatal, "SHAP Summary – Fatal", "Fig_SHAP_Summary_Fatal.png")
plot_shap_summary(shap_serious, "SHAP Summary – Serious", "Fig_SHAP_Summary_Serious.png")
plot_shap_summary(shap_slight, "SHAP Summary – Slight", "Fig_SHAP_Summary_Slight.png")

# ✅ 关键修正:这里不应该用 shap_severe（会抵消），而应该用 shap_stage1_raw
# shap_severe = shap_fatal + shap_serious  # ❌ 错误:会抵消

# ✅ 正确:加载原始 Stage-1 SHAP
shap_stage1_raw = np.load(os.path.join(MID_DIR, "shap_stage1_raw.npy"))

# 绘制 Severe 的 SHAP Summary（基于 Stage-1）
plot_shap_summary(
    shap_stage1_raw,  # ← 用 Stage-1 的原始 SHAP
    "SHAP Summary – Severe (Fatal + Serious)",
    "Fig_SHAP_Summary_Severe.png"
)

# 5.2 SHAP Summary for Severe（按 LightGBM gain 排序）- 最终版
print("\n📊 Generating SHAP Summary for Severe (sorted by LightGBM gain)...")

# 加载 Stage-1 模型，获取 gain 排序
stage1 = joblib.load(os.path.join(MID_DIR, "stage1.pkl"))
importance_gain = stage1.booster_.feature_importance(importance_type='gain')

# 按 gain 降序排序
sorted_idx_gain = np.argsort(importance_gain)[::-1]

# 只选择 gain > 0 的特征
valid_gain_mask = importance_gain[sorted_idx_gain] > 0
sorted_idx_gain = sorted_idx_gain[valid_gain_mask]

# 根据实际可用特征数调整 top_n
top_n_requested = 20
top_n = min(top_n_requested, len(sorted_idx_gain))
top_idx_gain = sorted_idx_gain[:top_n]

print(f"   Requested top {top_n_requested} features, using top {top_n} (with gain > 0)")

# 使用原始 Stage-1 SHAP
shap_severe_top = shap_stage1_raw[:, top_idx_gain]
X_test_top = X_test[:, top_idx_gain]
feature_names_top = clean_feature_names[top_idx_gain]

print(f"   shap_severe_top shape: {shap_severe_top.shape}")

# ✅ 关键:反转数据顺序，因为 SHAP 从下往上画
shap_severe_reversed = shap_severe_top[:, ::-1]
X_test_reversed = X_test_top[:, ::-1]
feature_names_reversed = feature_names_top[::-1]

n_samples = shap_severe_top.shape[0]

# ✅ 使用 SHAP 库的 summary_plot（保留 violin 效果）
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_severe_reversed,
    X_test_reversed,
    feature_names=feature_names_reversed,
    max_display=top_n,
    show=False,
    plot_size=None,  # 使用全部样本
    alpha=0.5,  # 透明度
)

# 添加标题
#plt.title(
    #f"SHAP Summary – Severe (Fatal + Serious)\n(Top-{top_n} features by LightGBM gain, n = {n_samples:,} samples)",
    #fontsize=13, fontweight='bold', pad=15
#)

plt.xlabel(f"SHAP value (impact on Severe prediction, n = {n_samples:,} samples)", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig_SHAP_Summary_Severe_Sorted.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ SHAP Summary for Severe (sorted by gain) saved")
print(f"   Top-{top_n} features: {', '.join(feature_names_top[:min(5, top_n)])}\n")


# ======================================================
# Stage-1 Evaluation: Severe vs Slight (for paper reporting)
# ======================================================
# Ground truth: Severe = 1 (Fatal/Serious), Slight = 0
y_test_stage1 = np.isin(y_test, [1, 2]).astype(int)

# Prediction with a default threshold (0.5). You can tune this later if needed.
y_pred_stage1 = (p_severe >= 0.5).astype(int)

acc = accuracy_score(y_test_stage1, y_pred_stage1)
pre = precision_score(y_test_stage1, y_pred_stage1, zero_division=0)
rec = recall_score(y_test_stage1, y_pred_stage1, zero_division=0)
f1  = f1_score(y_test_stage1, y_pred_stage1, zero_division=0)
auc_sev = roc_auc_score(y_test_stage1, p_severe)   # IMPORTANT: use probability for AUC

print("\n===== Stage-1 (Severe vs Slight) Metrics =====")
print(f"Accuracy : {acc:.2f}")
print(f"Precision: {pre:.2f}")
print(f"Recall   : {rec:.2f}")
print(f"F1-score : {f1:.2f}")
print(f"AUC      : {auc_sev:.2f}")

# ======================================================
# 6. SHAP Global Importance (Stacked, Class-balanced)
# ======================================================
def plot_shap_stacked_emphasize_severe(
        shap_fatal,
        shap_serious,
        shap_slight,
        clean_feature_names,
        save_path,
        top_n=20
):
    """
    三分类堆叠图，但强调Severe（给Fatal/Serious更高权重）
    """

    n_fatal = shap_fatal.shape[0]
    n_serious = shap_serious.shape[0]
    n_slight = shap_slight.shape[0]

    mean_fatal = np.abs(shap_fatal).mean(axis=0)
    mean_serious = np.abs(shap_serious).mean(axis=0)
    mean_slight = np.abs(shap_slight).mean(axis=0)

    # 关键:只对Fatal和Serious加倍权重，Slight保持原值
    severe_boost = 3.0  # 可调整这个倍数
    mean_fatal_weighted = mean_fatal * severe_boost
    mean_serious_weighted = mean_serious * severe_boost
    mean_slight_weighted = mean_slight  # 不加权

    mean_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Fatal": mean_fatal_weighted,
        "Serious": mean_serious_weighted,
        "Slight": mean_slight_weighted
    })

    mean_df["Total"] = mean_df[["Fatal", "Serious", "Slight"]].sum(axis=1)
    mean_df = mean_df.sort_values("Total", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = np.zeros(len(mean_df))

    colors = {"Fatal": "#d62728", "Serious": "#ff7f0e", "Slight": "#2ca02c"}

    for cls in ["Fatal", "Serious", "Slight"]:
        values = mean_df[cls].values
        ax.barh(mean_df["Feature"], values, left=bottom, label=cls, color=colors[cls])
        bottom += values

    #ax.set_xlabel("Weighted mean |SHAP value| (Severe emphasized)", fontsize=12)
    #ax.set_title("Global SHAP Feature Importance", fontsize=14)
    ax.invert_yaxis()
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Severe-emphasized SHAP stacked importance saved")


# ✅ 调用修正后的函数名
plot_shap_stacked_emphasize_severe(
    shap_fatal,
    shap_serious,
    shap_slight,
    clean_feature_names,
    save_path=os.path.join(
        RESULT_DIR, "Fig_SHAP_Global_Importance_Stacked.png"
    ),
    top_n=20
)

# ======================================================
# 6.5 Feature Importance for Severe (基于 SHAP Mean |Value|)
# ======================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("\n📊 Generating Feature Importance for Severe (based on SHAP values)...")

# ✅ 加载原始 Stage-1 SHAP
shap_stage1_raw = np.load(os.path.join(MID_DIR, "shap_stage1_raw.npy"))

# 1. 计算每个特征的平均绝对 SHAP 值
mean_abs_shap = np.abs(shap_stage1_raw).mean(axis=0)

# 2. 按降序排序
sorted_idx = np.argsort(mean_abs_shap)[::-1]

# 3. 只选择 Top-30（或 Top-20）
top_n = 30
top_idx = sorted_idx[:top_n]
top_features = clean_feature_names[top_idx]
top_shap_values = mean_abs_shap[top_idx]

# 4. 创建 DataFrame（方便绘图）
df_importance = pd.DataFrame({
    'Feature': top_features,
    'SHAP_Importance': top_shap_values
}).sort_values('SHAP_Importance', ascending=True)  # ✅ 升序，因为 barh 从下往上画

# 5. 绘制横向条形图（模仿 LightGBM 风格）
plt.figure(figsize=(10, 8))
bars = plt.barh(
    df_importance['Feature'],
    df_importance['SHAP_Importance'],
    color='steelblue',
    edgecolor='black',
    linewidth=0.5
)

# 6. 添加数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(
        width +0.005, bar.get_y() + bar.get_height() / 2,
        f'{width:.2f}',
        ha='left', va='center', fontsize=10, color='black'
    )

# 7. 美化
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
#plt.title(
    #'Feature Importance for Severe Accidents (Severe vs Slight)',
    #fontsize=14, fontweight='bold', pad=20
#)
#plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()

# 8. 保存
plt.savefig(
    os.path.join(FIG_DIR, "Fig_Feature_Importance_Severe_SHAP.png"),
    dpi=300, bbox_inches='tight'
)
plt.close()

print(f"✅ Feature Importance (SHAP-based) saved")
print(f"   Top-5 features: {', '.join(top_features[:5])}")




# ======================================================
# 7. Global SHAP Heatmap with f(x) - 展示所有样本
# ======================================================

# ✅ 加载原始 Stage-1 SHAP
shap_stage1_raw = np.load(os.path.join(MID_DIR, "shap_stage1_raw.npy"))
base_values_stage1 = np.load(os.path.join(MID_DIR, "base_values_stage1.npy"))

# 计算所有样本的 f(x)
model_output_all = base_values_stage1 + shap_stage1_raw.sum(axis=1)

print(f"📊 Data check:")
print(f"   Total samples: {len(model_output_all)}")
print(f"   shap_stage1_raw range: [{shap_stage1_raw.min():.3f}, {shap_stage1_raw.max():.3f}]")
print(f"   f(x) range: [{model_output_all.min():.3f}, {model_output_all.max():.3f}]")
print(f"   Samples with f(x) > 0 (Severe): {(model_output_all > 0).sum()}")
print(f"   Samples with f(x) < 0 (Slight): {(model_output_all < 0).sum()}")


def plot_global_shap_heatmap_with_fx(
        shap_values,
        model_output,
        feature_names,
        save_path,
        top_n=20,
        sample_ratio=0.1  # 采样比例（如果样本太多）
):
    """
    Global SHAP heatmap - 展示所有测试样本
    参数 sample_ratio: 如果样本数 > 1000，按此比例采样（0.1 = 10%）
    """

    n_total = len(model_output)

    # 1. Select top-N features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    shap_top = shap_values[:, top_idx]
    feature_names_top = [feature_names[i] for i in top_idx]

    # 2. Sort by f(x) descending (Severe → Slight)
    order = np.argsort(model_output)[::-1]
    shap_sorted = shap_top[order]
    fx_sorted = model_output[order]

    # 3. 智能采样（保持分布特征）
    if n_total > 1000:
        # 分层采样:分别从高、中、低 f(x) 区域采样
        n_sample = max(500, int(n_total * sample_ratio))
        n_sample = min(n_sample, n_total)  # 不超过总数

        # 三分位采样
        n_high = n_sample // 3
        n_mid = n_sample // 3
        n_low = n_sample - n_high - n_mid

        idx_high = np.linspace(0, n_total // 3 - 1, n_high, dtype=int)
        idx_mid = np.linspace(n_total // 3, 2 * n_total // 3 - 1, n_mid, dtype=int)
        idx_low = np.linspace(2 * n_total // 3, n_total - 1, n_low, dtype=int)

        sample_idx = np.concatenate([idx_high, idx_mid, idx_low])
        sample_idx.sort()

        shap_sorted = shap_sorted[sample_idx]
        fx_sorted = fx_sorted[sample_idx]

        print(f"\n📊 Sampling:")
        print(f"   Original: {n_total} samples → Sampled: {len(fx_sorted)} samples")
    else:
        print(f"\n📊 Using all {n_total} samples (no sampling needed)")

    print(f"   f(x) range after sorting: [{fx_sorted.min():.3f}, {fx_sorted.max():.3f}]")
    print(f"   Severe samples: {(fx_sorted > 0).sum()}")
    print(f"   Slight samples: {(fx_sorted < 0).sum()}")

    # 4. Plot
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.05)

    # ---- f(x) curve ----
    ax_fx = fig.add_subplot(gs[0])
    ax_fx.plot(fx_sorted, color="black", linewidth=1.2)
    ax_fx.set_ylabel("Model output\nf(x)", fontsize=11, fontweight='bold')
    ax_fx.set_xticks([])
    ax_fx.spines['top'].set_visible(False)
    ax_fx.spines['right'].set_visible(False)
    ax_fx.spines['bottom'].set_visible(False)

    # Decision boundary
    ax_fx.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Labels
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
    ax_fx.text(0.02, 0.95, "Slight", transform=ax_fx.transAxes,
               fontsize=10, color='#2ca02c', va='top', fontweight='bold', bbox=bbox_props)
    ax_fx.text(0.98, 0.95, "Severe", transform=ax_fx.transAxes,
               fontsize=10, color='#d62728', va='top', ha='right', fontweight='bold', bbox=bbox_props)

    # Set reasonable y-axis limits
    y_range = fx_sorted.max() - fx_sorted.min()
    ax_fx.set_ylim(fx_sorted.min() - 0.1 * y_range, fx_sorted.max() + 0.1 * y_range)

    # ---- SHAP heatmap ----
    ax_hm = fig.add_subplot(gs[1])

    # Color scale: use 95th percentile to avoid extreme outliers
    vmax = np.percentile(np.abs(shap_sorted), 95)
    vmax = max(vmax, 0.01)  # 防止全0

    print(f"\n🎨 Heatmap:")
    print(f"   SHAP value range: [{shap_sorted.min():.4f}, {shap_sorted.max():.4f}]")
    print(f"   Color scale (95th %ile): ±{vmax:.4f}")

    sns.heatmap(
        shap_sorted.T,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=False,
        yticklabels=feature_names_top,
        cbar_kws={"label": "SHAP value", "shrink": 0.8},
        ax=ax_hm,
        linewidths=0,
        rasterized=True
    )

    ax_hm.set_xlabel("Test instances",
                     fontsize=12, fontweight='bold')
    ax_hm.set_ylabel("Feature", fontsize=12, fontweight='bold')

    # Decision boundary line
    split_idx = np.searchsorted(-fx_sorted, 0)

    if 0 < split_idx < len(fx_sorted):
        ax_hm.axvline(split_idx, color='red', linestyle='--', linewidth=2.5, alpha=0.8)

        # Add region labels at the bottom
        ax_hm.text(split_idx * 0.5, -1, "Slight\n(Predicted)",
                   ha='center', va='top', fontsize=11, color='#2ca02c',
                   fontweight='bold', transform=ax_hm.transData)
        ax_hm.text((split_idx + len(fx_sorted)) * 0.5, -1, "Severe\n(Predicted)",
                   ha='center', va='top', fontsize=11, color='#d62728',
                   fontweight='bold', transform=ax_hm.transData)

        print(f"   Decision boundary at index: {split_idx}/{len(fx_sorted)} ({split_idx / len(fx_sorted) * 100:.1f}%)")
    else:
        print(f"   ⚠️ No decision boundary visible (all samples on one side)")

    #plt.suptitle(f"Global SHAP Heatmap for Accident Severity Prediction\n(n = {len(fx_sorted):,} samples)",
                 #fontsize=16, fontweight='bold', y=0.98)

    plt.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.06)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Heatmap saved: {save_path}\n")


# ✅ 调用:使用所有样本
plot_global_shap_heatmap_with_fx(
    shap_values=shap_stage1_raw,
    model_output=model_output_all,
    feature_names=clean_feature_names,
    save_path=os.path.join(FIG_DIR, "Fig_SHAP_Global_Heatmap_All_Samples.png"),
    top_n=20,
    sample_ratio=0.05  # 如果样本 > 1000，采样5%（约500-1000个样本用于可视化）
)

# ======================================================
# 8. SHAP Dependence Plots – Severe (Top 6) - 改进版
# ======================================================
print("\n📊 Generating SHAP Dependence Plots for Severe (enhanced with categorical labels)...")

# ✅ 使用原始 Stage-1 SHAP
shap_stage1_raw = np.load(os.path.join(MID_DIR, "shap_stage1_raw.npy"))

# 1. 计算每个特征的平均绝对 SHAP 值
mean_abs_shap_severe = np.abs(shap_stage1_raw).mean(axis=0)

# 2. 按降序排序,选 Top-6
top6_severe_idx = np.argsort(mean_abs_shap_severe)[::-1][:6]

# ======================================================
# 🔧 关键改进:定义分类特征的映射字典
# ======================================================
# 根据你的数据定义类别映射
categorical_mappings = {
    'Junction_Control': {
        0: 'Not at junction or within 20 metres', 1: 'Authorised person', 2: 'Auto traffic signal', 3: 'Stop sign',
        4: 'Give way or uncontrolled'},
    'Urban_or_Rural': {
        1: 'Urban', 2: 'Rural'},
    'Road_Type': {
        1: 'Roundabout',
        2: 'One way street',
        3: 'Dual carriageway',
        6: 'Single carriageway',
        7: 'Slip road',
        9: 'Unknown',
        12: 'One way street/Slip road'
    },
    'Time_Period': {
        1: 'Late Night',
        2: 'Morning',
        3: 'Noon',
        4: 'Afternoon',
        5: 'Evening'
    },
    'Month': {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    },
    'Number_of_Casualties': {i: str(i) for i in range(1, 69)
                             },
    'Number_of_Vehicles': {i: str(i) for i in range(1, 32)
                             },
    'Speed_limit': {
        30: '30 mph', 40: '40 mph',
        50: '50 mph', 60: '60 mph', 70: '70 mph'
    }
}

"""添加其他分类特征..."""

def customize_categorical_colorbar(ax, feature_name, feature_values, cmap='viridis'):
    """
    为分类特征自定义colorbar,显示类别名称

    Parameters:
    - ax: matplotlib axes对象
    - feature_name: 特征名称
    - feature_values: 该特征在X_test中的值
    - cmap: 颜色映射
    """
    if feature_name not in categorical_mappings:
        return  # 如果不是分类特征,保持默认

    mapping = categorical_mappings[feature_name]
    unique_vals = sorted(set(feature_values) & set(mapping.keys()))

    if len(unique_vals) == 0:
        return

    # 获取colorbar对象
    cbar = ax.collections[0].colorbar
    if cbar is None:
        return

    # 设置刻度位置和标签
    cbar.set_ticks(unique_vals)
    cbar.set_ticklabels([mapping[v] for v in unique_vals])
    cbar.ax.tick_params(labelsize=9)

    # 如果类别太多,旋转标签
    if len(unique_vals) > 5:
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=0, ha='left')


# ======================================================
# 8.1 单独绘制每个特征的 Dependence Plot(改进版)
# ======================================================
print("\n📊 Step 1: Generating individual Dependence Plots with categorical labels...")

for idx in top6_severe_idx:
    feature_name = clean_feature_names[idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制dependence plot
    shap.dependence_plot(
        idx,
        shap_stage1_raw,
        X_test,
        feature_names=clean_feature_names,
        interaction_index='auto',  # 自动选择交互特征
        show=False,
        alpha=0.6,
        ax=ax
    )

    # 🔧 关键:自定义交互特征的colorbar
    # SHAP会自动选择最强交互特征,我们需要获取它的名称
    # 检查是否有colorbar(表示有交互特征)
    if len(ax.collections) > 0 and hasattr(ax.collections[0], 'colorbar'):
        # 获取交互特征的索引
        interaction_idx = shap.approximate_interactions(
            idx, shap_stage1_raw, X_test
        )[0]
        interaction_feature_name = clean_feature_names[interaction_idx]

        # 自定义colorbar
        customize_categorical_colorbar(
            ax,
            interaction_feature_name,
            X_test[:, interaction_idx]
        )

        # 更新colorbar标签
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.set_label(
                interaction_feature_name,
                fontsize=11,
                fontweight='bold'
            )

    # 添加标题和标签
    #plt.title(
        #f"SHAP Dependence Plot: {feature_name}\n"
        #f"(Mean |SHAP|: {mean_abs_shap_severe[idx]:.4f})",
        #fontsize=14,
        #fontweight='bold',
        #pad=15
    #)
    ax.set_xlabel(f"{feature_name}", fontsize=12, fontweight='bold')
    ax.set_ylabel("SHAP value for Severe", fontsize=12, fontweight='bold')
    #ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.grid(False)

    plt.tight_layout()

    safe_name = feature_name.replace('/', '_').replace(' ', '_')
    plt.savefig(
        os.path.join(FIG_DIR, f"Fig_SHAP_Dependence_Severe_{safe_name}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print(f"   ✅ Saved: Fig_SHAP_Dependence_Severe_{safe_name}.png")

# ======================================================
# 8.2 绘制 3x2 复合图(改进版)
# ======================================================
print("\n📊 Step 2: Generating 3x2 composite Dependence Plot with categorical labels...")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

for ax, feat_idx in zip(axes, top6_severe_idx):
    feature_name = clean_feature_names[feat_idx]

    # 绘制dependence plot
    shap.dependence_plot(
        feat_idx,
        shap_stage1_raw,
        X_test,
        feature_names=clean_feature_names,
        interaction_index='auto',
        ax=ax,
        show=False,
        alpha=0.6
    )

    # 🔧 自定义交互特征colorbar
    if len(ax.collections) > 0 and hasattr(ax.collections[0], 'colorbar'):
        interaction_idx = shap.approximate_interactions(
            feat_idx, shap_stage1_raw, X_test
        )[0]
        interaction_feature_name = clean_feature_names[interaction_idx]

        customize_categorical_colorbar(
            ax,
            interaction_feature_name,
            X_test[:, interaction_idx]
        )

        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.set_label(
                interaction_feature_name,
                fontsize=10,
                fontweight='bold'
            )
            cbar.ax.tick_params(labelsize=8)

    # 设置子图标题
    rank = list(top6_severe_idx).index(feat_idx) + 1
    ax.set_title(
        f"({chr(96 + rank)}) {feature_name}\n"
        f"(Mean |SHAP|: {mean_abs_shap_severe[feat_idx]:.4f})",
        fontsize=11,
        fontweight='bold',
        pad=8
    )

    ax.set_xlabel(f"{feature_name}", fontsize=10, fontweight='bold')
    ax.set_ylabel("SHAP value", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

#fig.suptitle(
    #"SHAP Dependence Plots for Top-6 Severe-Related Features\n"
    #"(Based on Stage-1: Severe vs Slight)",
    #fontsize=16,
    #fontweight='bold',
    #y=0.995
#)

plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig(
    os.path.join(FIG_DIR, "Fig_SHAP_Dependence_Top6_Severe_2x2_Enhanced.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.close()

print(f"✅ Enhanced SHAP Dependence plots saved!\n")

# ======================================================
# 10. Spatial Distribution of Accident Severity
# ======================================================
def plot_spatial_distribution_full_dataset(save_path):
    """
    使用原始完整数据集绘制空间分布统计图
    不使用 X_test，而是直接从源数据加载
    """
    import pandas as pd

    print("=" * 60)
    print("📊 Loading FULL original dataset for spatial distribution...")
    print("=" * 60)

    # 加载原始完整数据
    source_csv = r"E:\jiedan\交通事故\data\Accidents0514_copy.csv"

    try:
        # 只读取需要的列以节省内存
        df = pd.read_csv(source_csv, usecols=['Longitude', 'Latitude', 'Accident_Severity'])

        print(f"✅ Loaded {len(df):,} accidents from full dataset")
        print(f"   Unique severities: {df['Accident_Severity'].unique()}")

        # 提取经纬度和严重程度
        lons = df['Longitude'].values
        lats = df['Latitude'].values
        severities = df['Accident_Severity'].values

        # 统计各类别数量
        severity_counts = df['Accident_Severity'].value_counts().sort_index()
        print(f"\n📈 Severity distribution:")
        for sev, count in severity_counts.items():
            print(f"   {sev}: {count:,} ({count / len(df) * 100:.2f}%)")

    except FileNotFoundError:
        print(f"❌ Error: Source file not found at {source_csv}")
        return
    except KeyError as e:
        print(f"❌ Error: Column {e} not found in CSV")
        print(f"Available columns: {df.columns.tolist()}")
        return
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # 移除缺失值
    valid_mask = ~(pd.isna(lons) | pd.isna(lats) | pd.isna(severities))
    lons = lons[valid_mask]
    lats = lats[valid_mask]
    severities = severities[valid_mask]

    print(f"✅ Valid coordinates: {len(lons):,} (removed {(~valid_mask).sum():,} NaN values)")
    print(f"📍 Data range: Lon [{lons.min():.4f}, {lons.max():.4f}], Lat [{lats.min():.4f}, {lats.max():.4f}]")

    # 三分类配置（根据UK数据标准，通常是1=Fatal, 2=Serious, 3=Slight）
    severity_config = {
        1: {'label': 'Fatal', 'color': '#d62728', 'alpha': 0.8, 'size': 20},  # 红色，最大最明显
        2: {'label': 'Serious', 'color': '#ff7f0e', 'alpha': 0.6, 'size': 10},  # 橙色，中等
        3: {'label': 'Slight', 'color': '#2ca02c', 'alpha': 0.4, 'size': 2}  # 绿色，最小
    }

    # 创建 1x3 子图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, severity_level in enumerate([1, 2, 3]):
        ax = axes[idx]
        mask = (severities == severity_level)

        if not np.any(mask):
            print(f"⚠️ No data for severity level {severity_level}")
            ax.text(0.5, 0.5, f'No data for {severity_config[severity_level]["label"]}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue

        lats_sub = lats[mask]
        lons_sub = lons[mask]
        config = severity_config[severity_level]
        n_points = len(lons_sub)

        print(f"\n🎨 Plotting {config['label']}: {n_points:,} points...")

        # 智能采样策略（保持空间分布特征）
        if n_points > 100000:
            # 对于超大数据集，分层采样
            sample_size = 100000
            print(f"   ⚡ Sampling to {sample_size:,} points for rendering speed")

            # 使用网格采样保持空间分布
            from sklearn.cluster import KMeans
            coords = np.column_stack([lons_sub, lats_sub])

            # 简单随机采样（更快）
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            lons_sub = lons_sub[sample_indices]
            lats_sub = lats_sub[sample_indices]

        elif n_points > 50000:
            sample_size = 50000
            print(f"   ⚡ Sampling to {sample_size:,} points")
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            lons_sub = lons_sub[sample_indices]
            lats_sub = lats_sub[sample_indices]

        # 绘制散点
        scatter = ax.scatter(
            lons_sub, lats_sub,
            c=config['color'],
            s=config['size'],
            alpha=config['alpha'],
            edgecolors='none',
            rasterized=True  # 关键:加速大数据量PDF渲染
        )

        # 标题和标签
        ax.set_title(f"({chr(97 + idx)}) {config['label']}",
                     fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

        # 统一坐标范围（所有子图使用相同范围以便比较）
        ax.set_xlim(lons.min() - 0.05, lons.max() + 0.05)
        ax.set_ylim(lats.min() - 0.05, lats.max() + 0.05)

        # 网格和外观
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=10)

        # 数据量标注（带百分比）
        percentage = (n_points / len(severities)) * 100
        ax.text(0.02, 0.98,
                f'n={n_points:,}\n({percentage:.1f}%)',
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # 总标题
    #plt.suptitle(f'Spatial Distribution of Accident Severities (Total: {len(severities):,} accidents)',
                 #fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Spatial distribution plot saved: {save_path}")
    print("=" * 60)


# 调用函数（不需要传入 X_test 和 y_test）
plot_spatial_distribution_full_dataset(
    save_path=os.path.join(RESULT_DIR, "Fig_Spatial_Distribution_Full.png")
)


# ======================================================
# 11. Temporal Distribution of Accident Severity
# (Time Period / Day / Month) – 3-class
# ======================================================

print("\n📊 Generating temporal distribution plots (3-class)...")

# ---------- Load FULL original dataset ----------
source_csv = r"E:\jiedan\交通事故\data\Accidents0514_copy.csv"
df_time = pd.read_csv(source_csv)

# ---------- Time feature engineering ----------
df_time["Date"] = pd.to_datetime(df_time["Date"], errors="coerce")

# Month
df_time["Month"] = df_time["Date"].dt.month

# Day of week (Mon=0 ... Sun=6)
day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df_time["Day"] = df_time["Date"].dt.dayofweek.map(day_map)

# Time period (consistent with your preprocessing)
def time_period(t):
    if pd.isna(t):
        return np.nan
    try:
        hour = int(str(t).split(":")[0])
    except:
        return np.nan
    if 0 <= hour < 6:
        return "Late Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 14:
        return "Noon"
    elif 14 <= hour < 18:
        return "Afternoon"
    else:
        return "Evening"

df_time["Time_Period"] = df_time["Time"].apply(time_period)

# ---------- Severity mapping (3-class) ----------
severity_map = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}
df_time["Severity"] = df_time["Accident_Severity"].map(severity_map)

# Keep valid rows only
df_time = df_time.dropna(subset=["Severity"])

# ---------- Plot helper ----------
def plot_temporal_bar(df, time_col, order, save_name, xlabel=None, title=None):
    pivot = (
        df.groupby([time_col, "Severity"])
          .size()
          .unstack(fill_value=0)
          .reindex(order)
    )

    pivot = pivot[["Fatal", "Serious", "Slight"]]  # 固定顺序

    x = np.arange(len(pivot.index))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, pivot["Fatal"], width, label="Fatal", color="#E10F1F")
    plt.bar(x,         pivot["Serious"], width, label="Serious", color="#FF7F0E")
    plt.bar(x + width, pivot["Slight"], width, label="Slight", color="#2CA02C")

    plt.xticks(x, pivot.index, rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel("Number of accidents")
    plt.title(title)
    plt.legend(
        loc="upper center",  #图例改到正下方
        bbox_to_anchor=(0.5, -0.08),  #legend放高一点
        ncol=3,
        frameon=False
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    plt.savefig(os.path.join(RESULT_DIR, save_name), dpi=300)
    plt.close()

    print(f"   ✅ Saved {save_name}")

# ---------- (a) Time Period ----------
time_period_order = ["Late Night", "Morning", "Noon", "Afternoon", "Evening"]

plot_temporal_bar(
    df_time,
    time_col="Time_Period",
    order=time_period_order,
    #xlabel="Time period",
    #title="Accident severity distribution by time period",
    save_name="Fig_Temporal_Time_Period.png"
)

# ---------- (b) Day ----------
day_order = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]

plot_temporal_bar(
    df_time,
    time_col="Day",
    order=day_order,
    #xlabel="Day of week",
    #title="Accident severity distribution by day of week",
    save_name="Fig_Temporal_Day.png"
)

# ---------- (c) Month ----------
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

df_time["Month_Name"] = df_time["Month"].map({
    1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June",
    7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"
})

plot_temporal_bar(
    df_time,
    time_col="Month_Name",
    order=month_order,
    #xlabel="Month",
    #title="Accident severity distribution by month",
    save_name="Fig_Temporal_Month.png"
)

print("✅ Temporal distribution plots finished.")


# ======================================================
# 12. SHAP Subcategory Analysis for Top-6 Severe Features
# ======================================================
print("\n📊 Generating SHAP Subcategory Analysis for Top-6 Severe Features...")


def plot_shap_subcategory_for_top_features(
        shap_values,
        X_test,
        feature_names,
        top_n=6,
        save_dir=FIG_DIR
):
    """
    为 Top-N 重要特征绘制子类别 SHAP 分析图
    所有条形向右延伸（使用绝对值），通过颜色区分正负
    """

    # 1. 计算特征重要性，选择 Top-N
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    print(f"   Selected top-{top_n} features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank}. {feature_names[idx]} (Mean |SHAP|: {mean_abs_shap[idx]:.4f})")

    for rank, feat_idx in enumerate(top_indices, 1):
        feature_name = feature_names[feat_idx]

        if feature_name not in categorical_mappings:
            print(f"   ⚠️ Skipping {feature_name} (not in categorical_mappings)")
            continue

        print(f"\n   [{rank}/{top_n}] Processing: {feature_name}")

        mapping = categorical_mappings[feature_name]
        feature_values = X_test[:, feat_idx]
        feature_shap = shap_values[:, feat_idx]

        # 3. 计算每个子类别的平均 SHAP 值
        subcategory_stats = {}

        for cat_value, cat_label in mapping.items():
            mask = (feature_values == cat_value)

            if not np.any(mask):
                continue

            mean_shap = feature_shap[mask].mean()
            count = mask.sum()

            subcategory_stats[cat_label] = {
                'mean_shap': mean_shap,
                'count': count
            }

        if len(subcategory_stats) == 0:
            print(f"      ⚠️ No valid subcategories found")
            continue

        # 过滤逻辑
        if len(subcategory_stats) > 20:
            original_count = len(subcategory_stats)
            subcategory_stats = {
                label: stats for label, stats in subcategory_stats.items()
                if stats['count'] > 100
            }
            print(f"      ⚠️ Too many subcategories ({original_count}), "
                  f"filtered to {len(subcategory_stats)} (count > 100)")

            if len(subcategory_stats) > 20:
                sorted_stats = sorted(
                    subcategory_stats.items(),
                    key=lambda x: abs(x[1]['mean_shap']),
                    reverse=True
                )[:20]
                subcategory_stats = dict(sorted_stats)
                print(f"      ⚠️ Further filtered to top-20 by |SHAP|")

        # 4. 准备绘图数据
        df_plot = pd.DataFrame([
            {
                'Subcategory': label,
                'Mean_SHAP': stats['mean_shap'],
                'Abs_SHAP': abs(stats['mean_shap']),  # ✅ 绘图用绝对值
                'Count': stats['count'],
                'Direction': 'Positive' if stats['mean_shap'] > 0 else 'Negative'  # ✅ 记录方向
            }
            for label, stats in subcategory_stats.items()
        ])

        # 按绝对值排序（升序，因为 barh 从下往上画）
        df_plot = df_plot.sort_values('Abs_SHAP', ascending=True)

        print(f"      Found {len(df_plot)} subcategories")

        # 5. 绘制横向条形图
        fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.5)))

        # ✅ 颜色映射：根据原始值的正负，但都画向右
        colors = ['#d62728' if direction == 'Positive' else '#1f77b4'
                  for direction in df_plot['Direction']]

        bars = ax.barh(
            df_plot['Subcategory'],
            df_plot['Abs_SHAP'],  # ✅ 使用绝对值，所有条形向右
            color=colors,
            edgecolor='black',
            linewidth=0.8,
            alpha=0.8
        )

        # 6. 添加 SHAP 值标签
        for bar, (_, row) in zip(bars, df_plot.iterrows()):
            width = bar.get_width()
            y_pos = bar.get_y() + bar.get_height() / 2

            # SHAP 值标签
            label_x = width + 0.002
            shap_text = f"{row['Mean_SHAP']:+.2f}" if pd.notna(row['Mean_SHAP']) else "+0.0000"

            ax.text(
                label_x, y_pos, shap_text,
                ha='left', va='center',
                fontsize=9, fontweight='bold', color='black'
            )

        # 7. 美化图表
        ax.set_xlabel(f'{feature_name}', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        #ax.set_title(
            #f'SHAP Subcategory Analysis: {feature_name}\n'
            #f'(Rank #{rank}, Mean |SHAP|: {mean_abs_shap[feat_idx]:.4f})',
            #fontsize=13, fontweight='bold', pad=15
        #)

        # ✅ 添加图例说明颜色含义
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', edgecolor='black', label='Positive (Increase Severe Risk)'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='Negative (Decrease Severe Risk)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

        # 网格
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # ✅ 设置 x 轴从 0 开始
        ax.set_xlim(left=0)

        # 调整布局
        plt.tight_layout()

        # 8. 保存
        safe_name = (feature_name
                     .replace('/', '_')
                     .replace(' ', '_')
                     .replace('(', '')
                     .replace(')', '')
                     .replace('-', '_')
                     .replace('&', 'and'))

        output_path = os.path.join(save_dir, f"Fig_SHAP_Subcategory_{safe_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"      ✅ Saved: {os.path.basename(output_path)}")

    print(f"\n✅ SHAP Subcategory Analysis completed!\n")


# ✅ 调用函数
plot_shap_subcategory_for_top_features(
    shap_values=shap_stage1_raw,
    X_test=X_test,
    feature_names=clean_feature_names,
    top_n=6,
    save_dir=FIG_DIR
)
