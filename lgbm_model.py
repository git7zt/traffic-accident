# -*- coding: utf-8 -*-
"""
models_twostage_soft.py
Two-stage LightGBM severity model (training + inference).
All intermediate results are saved for paper plotting.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from lightgbm import LGBMClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ======================================================
# 0. Path configuration
# ======================================================
DATA_DIR = r"E:\jiedan\交通事故\data\output"
RESULT_DIR = r"E:\jiedan\交通事故\results\two_stage_soft"
MID_DIR = os.path.join(RESULT_DIR, "MID_result")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MID_DIR, exist_ok=True)

# ======================================================
# 1. Load data (ground truth only from DATA_DIR)
# ======================================================
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

with open(os.path.join(DATA_DIR, "feature_names.json"), "r") as f:
    feature_names = json.load(f)

print("Train label distribution:")
print(pd.Series(y_train).value_counts())

# ======================================================
# Feature name cleaning (remove 'remainder' prefix)
# ======================================================
clean_feature_names = [
    name.replace("remainder__", "") if name.startswith("remainder__") else name
    for name in feature_names
]


# ======================================================
# 2. Stage-1: Severe vs Slight
# ======================================================
y_train_stage1 = np.isin(y_train, [1, 2]).astype(int)

stage1 = LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight={0: 1, 1: 4},
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

stage1.fit(X_train, y_train_stage1, feature_name=clean_feature_names)
joblib.dump(stage1, os.path.join(MID_DIR, "stage1.pkl"))

# 方案1: 增加图表高度，特征名自动换行
plt.figure(figsize=(14, 25))  # 增加高度
ax = plot_importance(
    stage1,
    max_num_features=30,
    height=0.7,
    importance_type='gain'  # ← 关键修改：使用 gain
)
ax.grid(False)

# 获取当前的y轴标签并进行换行处理
labels = ax.get_yticklabels()
new_labels = []
for label in labels:
    text = label.get_text()
    if len(text) > 24 and text.count('_') >= 2:
        # 找到第2个下划线的位置
        first_underscore = text.find('_')
        second_underscore = text.find('_', first_underscore + 1)
        # 在第2个下划线处插入换行
        text = text[:second_underscore] + '_\n' + text[second_underscore+1:]
    new_labels.append(text)

ax.set_yticklabels(new_labels, fontsize=9, linespacing=1) # 增加行间距

plt.title("Stage-1 Feature Importance (Severe vs Slight)", fontsize=14, pad=20)
plt.xlabel("Feature Importance", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "feature_importance_stage1.png"), dpi=300, bbox_inches='tight')
plt.close()




# ======================================================
# 3. Stage-2: Fatal vs Serious (within Severe)
# ======================================================
mask_severe_train = np.isin(y_train, [1, 2])
X_train_severe = X_train[mask_severe_train]

# Fatal = 1, Serious = 0
y_train_stage2 = (y_train[mask_severe_train] == 1).astype(int)

stage2 = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight={0: 1, 1: 2},
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

stage2.fit(X_train_severe, y_train_stage2, feature_name=clean_feature_names)
joblib.dump(stage2, os.path.join(MID_DIR, "stage2.pkl"))

# ======================================================
# 4. Soft fusion inference (3-class probabilities)
# ======================================================
# Stage-1: Severe probability
p_severe = stage1.predict_proba(X_test)[:, 1]

# Stage-2: Fatal | Serious (only inside Severe)
p_fatal = np.zeros(len(X_test))
mask_severe_test = p_severe > 0

p_fatal[mask_severe_test] = stage2.predict_proba(
    X_test[mask_severe_test]
)[:, 1]

# Final 3-class probability
proba_3 = np.zeros((len(X_test), 3))

# Fatal
proba_3[:, 0] = p_severe * p_fatal

# Serious
proba_3[:, 1] = p_severe * (1 - p_fatal)

# Slight
proba_3[:, 2] = 1 - p_severe

y_pred = np.argmax(proba_3, axis=1) + 1  # {1,2,3}

# ======================================================
# 5. Save intermediate results (ONLY once)
# ======================================================
np.save(os.path.join(MID_DIR, "p_severe.npy"), p_severe)
np.save(os.path.join(MID_DIR, "p_fatal.npy"), p_fatal)
np.save(os.path.join(MID_DIR, "proba_3.npy"), proba_3)
np.save(os.path.join(MID_DIR, "y_pred.npy"), y_pred)

# ======================================================
# 5.5 Final-output SHAP attribution (for 3-class interpretation)
# ======================================================
import shap

# ======================================================
# 5.5.1 Stage-1 SHAP: Severe vs Slight
# ======================================================
explainer_stage1 = shap.TreeExplainer(stage1)
shap_values_stage1_raw = explainer_stage1(X_test).values

# ✅ 保存原始 Stage-1 SHAP（用于计算 f(x)）
if shap_values_stage1_raw.ndim == 3:
    # 多分类格式：取正类（Severe = class 1）
    shap_stage1_raw = shap_values_stage1_raw[:, :, 1]
else:
    # 二分类格式
    shap_stage1_raw = shap_values_stage1_raw

np.save(os.path.join(MID_DIR, "shap_stage1_raw.npy"), shap_stage1_raw)

# ✅ 【关键修改】保存 Stage-1 的 SHAP 值（用于后续计算 shap_severe）
shap_stage1 = shap_stage1_raw  # 确保后续代码能用到这个变量


# ------------------------------
# 5.5.2 Stage-2 SHAP: Fatal vs Serious | Severe
# ------------------------------
mask_severe = p_severe > 0.5
X_test_severe = X_test[mask_severe]

explainer_stage2 = shap.TreeExplainer(stage2)
shap_values_stage2 = explainer_stage2(X_test_severe).values

# ✅ 保存 Stage-2 的 base_values（可选，如果需要分析 Fatal vs Serious）
if isinstance(explainer_stage2.expected_value, np.ndarray):
    base_values_stage2 = explainer_stage2.expected_value[1]
else:
    base_values_stage2 = explainer_stage2.expected_value

np.save(os.path.join(MID_DIR, "base_values_stage2.npy"), base_values_stage2)
print(f"✅ Saved base_values_stage2: {base_values_stage2}")

if shap_values_stage2.ndim == 3:
    shap_stage2 = shap_values_stage2[:, :, 1]
else:
    shap_stage2 = shap_values_stage2


# ------------------------------
# 5.5.3 Final SHAP attribution for three classes
# ------------------------------

# Slight：完全由 Stage-1 决定（对称）
shap_slight_final = -shap_stage1

# Fatal / Serious：仅在 Severe 子空间内有定义
shap_fatal_final = np.zeros_like(shap_stage1)
shap_serious_final = np.zeros_like(shap_stage1)

shap_fatal_final[mask_severe] = shap_stage2
shap_serious_final[mask_severe] = -shap_stage2

# ------------------------------
# 5.5.4 Save SHAP results for plotting
# ------------------------------
np.save(os.path.join(MID_DIR, "shap_stage1_severe.npy"), shap_stage1)
np.save(os.path.join(MID_DIR, "shap_stage2_fatal.npy"), shap_stage2)

np.save(os.path.join(MID_DIR, "shap_fatal_final.npy"), shap_fatal_final)
np.save(os.path.join(MID_DIR, "shap_serious_final.npy"), shap_serious_final)
np.save(os.path.join(MID_DIR, "shap_slight_final.npy"), shap_slight_final)


# ======================================================
# 6. Evaluation (for checking, not re-used elsewhere)
# ======================================================
print("\n===== Final 3-Class Evaluation (Soft Fusion) =====")
print(classification_report(
    y_test, y_pred,
    target_names=["Fatal", "Serious", "Slight"],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlOrRd",
    xticklabels=["Fatal", "Serious", "Slight"],
    yticklabels=["Fatal", "Serious", "Slight"]
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# ======================================================
# 7. ROC curve (supplementary)
# ======================================================
y_test_bin = label_binarize(y_test, classes=[1, 2, 3])

plt.figure(figsize=(8, 6))
for i, label in enumerate(["Fatal", "Serious", "Slight"]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba_3[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"), dpi=300)
plt.close()

print("\n✅ Two-stage model training & inference finished.")
print("📁 Intermediate results:", MID_DIR)
print("📁 Paper figures:", RESULT_DIR)
