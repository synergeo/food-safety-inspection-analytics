"""
Reusable machine-learning utilities for the BVL Food Safety Inspection
Analytics project.

The project models a highly imbalanced binary target:
Target_Adverse = 1 for adverse food samples and 0 otherwise.

The workflow uses:
- an 80/20 stratified train-test split,
- one-hot encoding for categorical predictors,
- 5-fold stratified cross-validation,
- class weighting for the main models,
- PR-AUC / Average Precision as the main tuning metric,
- out-of-fold probabilities for threshold selection,
- and a final untouched test-set evaluation.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "PnMgT",
    "MatrixGrT",
    "MatrixT",
    "BetriebsartT",
    "Country_Feature",
    "Processing_Feature",
    "Packaging_Feature",
    "Sampling_Month",
]

SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
}


def prepare_xy(df_model):
    """
    Separate predictor matrix X and binary target y.

    ProbenNrPseud and Sampling_Date are intentionally excluded from the
    predictive feature matrix.
    """
    X = df_model[FEATURE_COLUMNS].copy()
    y = df_model["Target_Adverse"].copy()

    return X, y


def stratified_train_test_split(
    X,
    y,
    test_size=0.20,
):
    """
    Create the project's stratified 80/20 train-test split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor():
    """
    Create the categorical preprocessing pipeline.

    All eight predictors used in the final project are treated as
    categorical and one-hot encoded. Unknown categories encountered later
    are ignored safely.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                FEATURE_COLUMNS,
            )
        ]
    )


def build_cross_validator():
    """
    Create the project's 5-fold stratified cross-validation strategy.
    """
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def build_model_pipelines(y_train):
    """
    Build the four classifiers evaluated in the project.

    Logistic Regression, Decision Tree and Random Forest use balanced
    class weights. XGBoost uses scale_pos_weight calculated from the
    training data.
    """
    preprocessor = build_preprocessor()

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = (
        negative_count / positive_count
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    decision_tree_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=3,
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic_pipeline,
        "Decision Tree": decision_tree_pipeline,
        "Random Forest": random_forest_pipeline,
        "XGBoost": xgb_pipeline,
    }


def cross_validate_model(
    pipeline,
    X_train,
    y_train,
):
    """
    Evaluate a model using the project's 5-fold stratified CV strategy.
    """
    cv = build_cross_validator()

    return cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
    )


def summarize_cv_results(results):
    """
    Return mean cross-validation performance metrics.
    """
    return {
        "Precision":
            results["test_precision"].mean(),
        "Recall":
            results["test_recall"].mean(),
        "F1":
            results["test_f1"].mean(),
        "PR_AUC":
            results["test_average_precision"].mean(),
        "ROC_AUC":
            results["test_roc_auc"].mean(),
    }


def tune_random_forest(
    random_forest_pipeline,
    X_train,
    y_train,
):
    """
    Tune the Random Forest using the parameter grid from the final notebook.

    Average Precision (PR-AUC) is used as the GridSearchCV optimization
    metric because the adverse class is extremely rare.
    """
    rf_param_grid = {
        "classifier__n_estimators":
            [100, 200, 300],
        "classifier__max_depth":
            [None, 10, 20],
        "classifier__min_samples_leaf":
            [1, 2, 5],
        "classifier__max_features":
            ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid=rf_param_grid,
        scoring="average_precision",
        cv=build_cross_validator(),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    return grid_search


def generate_oof_probabilities(
    model,
    X_train,
    y_train,
):
    """
    Generate out-of-fold positive-class probabilities.

    These probabilities allow classification thresholds to be evaluated
    using training data without using the held-out test set.
    """
    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=build_cross_validator(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return probabilities


def evaluate_thresholds(
    y_true,
    probabilities,
    thresholds=None,
):
    """
    Evaluate precision, recall, F1 and screening burden across thresholds.

    The final notebook examined thresholds from 0.05 through 0.80 in
    increments of 0.05.
    """
    if thresholds is None:
        thresholds = np.arange(
            0.05,
            0.81,
            0.05,
        )

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "Samples_Flagged":
                    int(predictions.sum()),
                "Flagged_Percentage":
                    predictions.mean() * 100,
            }
        )

    return pd.DataFrame(results)


def evaluate_classifier(
    model,
    X,
    y,
    threshold=0.40,
):
    """
    Evaluate a fitted probabilistic classifier at a specified threshold.

    The final project locked the Random Forest classification threshold
    at 0.40 before evaluating the untouched test set.
    """
    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y,
        predictions,
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "Threshold": threshold,
        "Precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "PR_AUC": average_precision_score(
            y,
            probabilities,
        ),
        "ROC_AUC": roc_auc_score(
            y,
            probabilities,
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Samples_Flagged":
            int(predictions.sum()),
        "Flagged_Percentage":
            predictions.mean() * 100,
    }
