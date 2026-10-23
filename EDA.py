#1 : Install / Import Libraries
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn', 'pandas',
                'numpy', 'matplotlib', 'seaborn', 'joblib', 'ipywidgets', '-q'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              ConfusionMatrixDisplay, RocCurveDisplay)

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

print('All libraries loaded successfully!')

# 2 Load & Explore Data 
df = pd.read_csv('heart_cleveland_upload.csv')
df = df.rename(columns={'condition': 'target'})

print('═'*55)
print('  DATASET OVERVIEW')
print('═'*55)
print(f'  Shape            : {df.shape[0]} rows × {df.shape[1]} columns')
print(f'  Missing values   : {df.isnull().sum().sum()}')
print(f'  Duplicate rows   : {df.duplicated().sum()}')
print(f'  No Disease  (0)  : {(df.target==0).sum()} patients')
print(f'  Heart Disease (1): {(df.target==1).sum()} patients')
print('═'*55)
display(df.head())
print()
display(df.describe().round(2))

# EDA: correlations, distributions, and outliers
focus_cols = [c for c in ["age", "trestbps", "chol"] if c in df.columns]

# Compare the selected health measures for patients with and without disease
print("Summary by heart disease outcome (0 = no disease, 1 = disease)")
display(df.groupby("target")[focus_cols].agg(["mean", "median", "std"]).round(2))

# Correlations with the outcome and among the selected measures
corr_cols = focus_cols + ["target"]
print("Pearson correlations")
display(df[corr_cols].corr(method="pearson").round(2))

print("Spearman correlations")
display(df[corr_cols].corr(method="spearman").round(2))

# Distributions by outcome
fig, axes = plt.subplots(1, len(focus_cols), figsize=(5 * len(focus_cols), 4))
if len(focus_cols) == 1:
    axes = [axes]

for col, ax in zip(focus_cols, axes):
    sns.histplot(
        data=df, x=col, hue="target", kde=True,
        element="step", stat="density", common_norm=False, ax=ax
    )
    ax.set_title(f"{col} distribution by outcome")
    ax.set_xlabel(col)

plt.tight_layout()
plt.show()

# Compare spread and potential outliers by outcome
fig, axes = plt.subplots(1, len(focus_cols), figsize=(5 * len(focus_cols), 4))
if len(focus_cols) == 1:
    axes = [axes]

for col, ax in zip(focus_cols, axes):
    sns.boxplot(data=df, x="target", y=col, ax=ax)
    ax.set_title(f"{col} by outcome")
    ax.set_xlabel("Heart disease outcome (0 = no, 1 = yes)")

plt.tight_layout()
plt.show()

# Count potential outliers using the 1.5 × IQR rule
outlier_counts = {}
for col in focus_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outlier_counts[col] = ((df[col] < q1 - 1.5 * iqr) |
                          (df[col] > q3 + 1.5 * iqr)).sum()

print("Potential outliers (1.5 × IQR rule)")
display(pd.Series(outlier_counts, name="outlier_count").to_frame())
