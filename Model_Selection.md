# Sprint 004 Task 002 --- Final Hyperparameter Tuning and Model Selection

## 1. Objective

The objective of this task was to tune the Random Forest classifier
using GridSearchCV and select the final model configuration for the
AI-Based Heart Disease Prediction System.

## 2. Dataset

-   Dataset shape: 297 rows × 14 columns
-   Input features: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`,
    `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
-   Target: `target`
-   Class distribution:
    -   No Disease (0): 160 records
    -   Heart Disease (1): 137 records

## 3. Hyperparameter Tuning

GridSearchCV evaluated 144 parameter combinations using StratifiedKFold
cross-validation with 5 folds, for a total of 720 fits. The scoring
metric was recall.

### Best parameters

``` python
{
    'criterion': 'gini',
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 100
}
```

-   Best cross-validation recall: **0.7688 (76.88%)**

## 4. Held-Out Test Results

  Metric         Score
  ----------- --------
  Accuracy      88.89%
  Precision     85.19%
  Recall        92.00%
  F1-score      88.46%
  ROC-AUC       0.9566

### Classification report

  Class                Precision   Recall   F1-score   Support
  ------------------ ----------- -------- ---------- ---------
  No Disease                0.93     0.86       0.89        29
  Heart Disease             0.85     0.92       0.88        25
  Overall accuracy           ---      ---       0.89        54

## 5. Confusion Matrix

The confusion matrix was:

``` text
[[25, 4],
 [ 2, 23]]
```

Interpretation: - True negatives: 25 - False positives: 4 - False
negatives: 2 - True positives: 23

The model correctly identified 23 of the 25 test cases belonging to the
Heart Disease class. It missed 2 positive cases and produced 4
false-positive predictions.

## 6. Feature Importance

  Feature      Importance
  ---------- ------------
  cp               0.1411
  thal             0.1214
  thalach          0.1195
  age              0.1071
  oldpeak          0.0972
  ca               0.0942
  chol             0.0867
  trestbps         0.0793
  exang            0.0484
  slope            0.0405
  sex              0.0331
  restecg          0.0208
  fbs              0.0108

## 7. Model Selection Summary

The tuned Random Forest model achieved 88.89% accuracy, 92.00% recall,
and an ROC-AUC of 0.9566 on the held-out test set. Recall is especially
relevant for a heart-disease screening support tool because
false-negative predictions are important to examine. The model and its
results may support clinical decision-making research, but they are not
a substitute for a clinician's assessment or a validated medical device.

## 8. Limitations and Notes

-   The dataset is small (297 records), so the results may not
    generalize to other patient populations.
-   The held-out test set contains 54 cases; estimates can vary with a
    different split.
-   The cross-validation recall (76.88%) is lower than the test recall
    (92.00%). This difference alone does not prove overfitting; use
    train-versus-validation results and check the full workflow for data
    leakage before drawing that conclusion.
-   Random Forest has been tuned here, but a claim that it is superior
    to every other algorithm would require fair comparative evaluation
    using the same split and evaluation process.
-   These figures were transcribed from the run output provided for this
    task and were not independently rerun during report creation.

## 9. Conclusion

The selected Random Forest configuration uses 100 trees, Gini criterion,
unlimited tree depth, square-root feature sampling, minimum leaf size 1,
and minimum split size 2. It produced promising test-set metrics for
this project. Further validation on larger, independent, representative
clinical data is required before any real-world clinical use.
