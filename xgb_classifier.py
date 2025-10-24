import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as calc_auc

# Load SNP Embeddings
X = np.load(config.ZSCORE_VECTOR_NPY)  # or config.SNP_IMAGE_NPY for PCA
print(f"Loaded embeddings: {X.shape}")

# Load label CSV
label_csv = config.CASE_CONTROL_LABELS  # should point to matched_subject_ids.csv
df_labels = pd.read_csv(label_csv)

# Filter out unmatched labels (group == -1)
df_valid = df_labels[df_labels["group"] > 0].copy()
print(f"Found {len(df_valid)} matched labels.")

# Remap: 1 → 0 (control), 2 → 1 (case) for XGBoost
df_valid["group"] = df_valid["group"].map({1: 0, 2: 1})

# Align samples
indices = df_valid["subject_index"].astype(int).values
y = df_valid["group"].astype(int).values
X = X[indices]
assert X.shape[0] == y.shape[0], "Mismatch between X and y after filtering."

# Setup Stratified 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, aucs = [], []

print("\nRunning 5-fold CV on XGBoost classifier...\n")
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    accs.append(acc)
    aucs.append(auc)

    print(f"Fold {fold} — Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# Summary
print("\nCross-Validation Summary")
print(f"Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Mean AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

## Plots
PLOT_DIR = config.FINAL_IMPUTED_DIR / "embedding_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Figure: Accuracy and AUC per fold
folds = np.arange(1, 6)

plt.figure(figsize=(8, 4))
plt.bar(folds - 0.2, accs, width=0.4, label="Accuracy", color="#1f77b4")
plt.bar(folds + 0.2, aucs, width=0.4, label="AUC", color="#ff7f0e")
plt.xticks(folds)
plt.xlabel("Fold")
plt.ylabel("Score")
plt.ylim(0.5, 0.8)
plt.title("Accuracy and AUC per Fold (XGBoost)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "figure_fold_scores.png", dpi=300)
plt.close()
print(f"Saved Figure Fold Scores to: {PLOT_DIR / 'figure_fold_scores.png'}")

# Figure: ROC curves per fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
plt.figure(figsize=(6, 5))

for i, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = calc_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Fold {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves per Fold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "figure_roc_curves.png", dpi=300)
plt.close()
print(f"Saved Figure ROC curves to: {PLOT_DIR / 'figure_roc_curves.png'}")
