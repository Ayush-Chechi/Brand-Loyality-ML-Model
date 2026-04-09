"""Training entrypoint for final project pipeline.

Models:
- Random Forest (deployment model)
- Logistic Regression (comparison only)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import CFG
from .data import load_dataset
from .preprocessing import FeatureEngineering, build_preprocessor


def evaluate(model, X_train, X_test, y_train, y_test, X_all, y_all) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.random_state)
    cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring="accuracy")
    return {
        "accuracy": float(acc),
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "ROC-AUC": float(auc),
        "CV Score": float(cv_scores.mean()),
        "CV Std": float(cv_scores.std(ddof=0)),
        "cm": cm,
    }


def _save_confusion(cm, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Loyal", "Loyal"], yticklabels=["Not Loyal", "Loyal"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_bar(x, y, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=x, y=y, color="#4f46e5")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def train(dataset_path: str = CFG.dataset_path, output_dir: str = "artifacts") -> None:
    out = Path(output_dir)
    models_dir = out / "models"
    plots_dir = out / "plots"
    reports_dir = out / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _, X, y_multiclass = load_dataset(dataset_path)
    y = (y_multiclass >= 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=CFG.random_state)

    preprocessor = build_preprocessor()

    lr = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineering()),
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=CFG.logreg_max_iter, class_weight="balanced", random_state=CFG.random_state)),
        ]
    )

    rf_base = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineering()),
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    class_weight="balanced",
                    random_state=CFG.random_state,
                ),
            ),
        ]
    )
    rf_search = RandomizedSearchCV(
        rf_base,
        param_distributions={
            "model__max_depth": [5, 6, 7],
            "model__min_samples_split": [10],
            "model__min_samples_leaf": [5],
            "model__max_features": ["sqrt", "log2", None],
        },
        n_iter=3,
        cv=5,
        scoring="accuracy",
        random_state=CFG.random_state,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_

    lr_metrics = evaluate(lr, X_train, X_test, y_train, y_test, X, y)
    rf_metrics = evaluate(rf, X_train, X_test, y_train, y_test, X, y)

    # Save models
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    joblib.dump(lr, models_dir / "logistic_regression.joblib")
    joblib.dump(rf, models_dir / "random_forest_tuned.joblib")

    # Plots
    _save_confusion(lr_metrics["cm"], "Confusion Matrix - Logistic Regression", plots_dir / "logistic_regression_cm.png")
    _save_confusion(rf_metrics["cm"], "Confusion Matrix - Random Forest (Tuned)", plots_dir / "random_forest_tuned_cm.png")

    # Class distribution
    counts = y.value_counts().sort_index()
    _save_bar(
        x=["Not Loyal", "Loyal"],
        y=[int(counts.get(0, 0)), int(counts.get(1, 0))],
        title="Class Distribution",
        xlabel="Class",
        ylabel="Count",
        path=plots_dir / "class_distribution.png",
    )

    # Model comparison chart
    _save_bar(
        x=["Logistic Regression", "Random Forest (Tuned)"],
        y=[lr_metrics["accuracy"], rf_metrics["accuracy"]],
        title="Model Comparison (Accuracy)",
        xlabel="Model",
        ylabel="Accuracy",
        path=plots_dir / "model_comparison_accuracy.png",
    )

    # RF feature importance
    feature_names = list(rf.named_steps["preprocess"].get_feature_names_out())
    importances = rf.named_steps["model"].feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    top_names = [feature_names[i] for i in idx]
    top_vals = [float(importances[i]) for i in idx]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_vals, y=top_names, color="#0ea5e9")
    plt.title("Random Forest Feature Importance (Top 10)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plots_dir / "random_forest_feature_importance.png", dpi=180)
    plt.close()

    # Reports
    table = pd.DataFrame(
        [
            {"Model": "Random Forest (Tuned)", **{k: rf_metrics[k] for k in ["accuracy", "F1", "Precision", "Recall", "ROC-AUC", "CV Score"]}},
            {"Model": "Logistic Regression", **{k: lr_metrics[k] for k in ["accuracy", "F1", "Precision", "Recall", "ROC-AUC", "CV Score"]}},
        ]
    ).rename(columns={"accuracy": "Accuracy"})
    table.to_csv(reports_dir / "model_performance_table.csv", index=False)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path,
        "leakage_free": True,
        "excluded_feature": "Next Purchase Decision",
        "target": "Loyal (1) vs Not Loyal (0)",
        "best_model": "Random Forest (Tuned)",
        "best_accuracy": rf_metrics["accuracy"],
        "models": table.to_dict(orient="records"),
        "rf_best_params": rf_search.best_params_,
        "rf_top_features": [{"feature": n, "importance": v} for n, v in zip(top_names, top_vals)],
        "overfitting_check": {
            "train_accuracy_rf": float(accuracy_score(y_train, rf.predict(X_train))),
            "test_accuracy_rf": float(rf_metrics["accuracy"]),
        },
    }
    (reports_dir / "project_experiment_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Training complete. Best accuracy: %.4f", rf_metrics["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final brand loyalty binary classifier")
    parser.add_argument("--dataset", type=str, default=CFG.dataset_path)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()

