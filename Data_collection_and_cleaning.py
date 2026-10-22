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