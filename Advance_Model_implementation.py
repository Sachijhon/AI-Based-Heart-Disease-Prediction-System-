
# ============================================================
# AI-Based Heart Disease Prediction System
# Random Forest Training, Hyperparameter Tuning & Evaluation
# ============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. Load dataset
DATA_PATH = Path("heart_cleveland_upload.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.rename(columns={"condition": "target"}, inplace=True)

# 2. Validate dataset
if "target" not in df.columns:
    raise ValueError("Target column not found in dataset.")

if df["target"].isnull().any():
    raise ValueError("Target column contains missing values.")

if not set(df["target"].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["target"])
y = df["target"].astype(int)

# Ensure model inputs are numeric
X = X.apply(pd.to_numeric, errors="raise")

if X.isnull().any().any():
    raise ValueError("Features contain missing values. Handle them before training.")

FEATURE_NAMES = X.columns.tolist()

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print("Dataset shape:", df.shape)
print("Features:", FEATURE_NAMES)
print("No disease (0):", (y == 0).sum())
print("Heart disease (1):", (y == 1).sum())

# 3. Split data before hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.18,
    random_state=11,
    stratify=y
)

# 4. Define Random Forest and hyperparameter grid
rf = RandomForestClassifier(
    random_state=123,
    n_jobs=-1,
    class_weight=None
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"]
}

# 5. Stratified 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Recall for class 1 is the tuning objective.
# Precision and F1 will also be evaluated afterward.
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="recall",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)

print("\nStarting hyperparameter tuning...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n" + "=" * 60)
print("BEST RANDOM FOREST MODEL")
print("=" * 60)
print("Best parameters:")
print(grid_search.best_params_)
print(f"Best CV recall: {grid_search.best_score_:.4f}")

# 6. Evaluate on untouched test data
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Disease", "Heart Disease"],
    zero_division=0
))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

# 7. Visualize confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Heart Disease"]
).plot(cmap="Blues", values_format="d")

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=300)
plt.show()

# 8. ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Random Forest - ROC Curve")
plt.tight_layout()
plt.savefig(MODEL_DIR / "roc_curve.png", dpi=300)
plt.show()

# 9. Feature importance
importance = pd.Series(
    best_model.feature_importances_,
    index=FEATURE_NAMES
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance.round(4))

importance.sort_values().plot(
    kind="barh",
    figsize=(9, 6),
    title="Random Forest Feature Importance"
)
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(MODEL_DIR / "feature_importance.png", dpi=300)
plt.show()

# 10. Save model and metadata
model_path = MODEL_DIR / "rf_heart_model.pkl"

joblib.dump(
    {
        "model": best_model,
        "feature_names": FEATURE_NAMES,
        "target_name": "target",
        "best_params": grid_search.best_params_,
        "cv_recall": float(grid_search.best_score_),
        "test_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }
    },
    model_path
)

print("\nModel saved successfully:", model_path.resolve())
print("Training and evaluation completed.")
