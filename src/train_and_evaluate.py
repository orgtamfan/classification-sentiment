"""
Model training and evaluation module for the research project:
"Reddit Comment Sentiment Classification in Counter-Strike 2 and VALORANT
Communities Using Naive Bayes and SVM Algorithms"

Features:
  1. Train Naive Bayes (MultinomialNB and ComplementNB) with alpha tuning.
  2. Train Support Vector Machine (LinearSVC) with C tuning.
  3. Run stratified 5-fold cross-validation.
  4. Evaluate on a held-out test set:
     - Accuracy, precision, recall, macro F1, and weighted F1.
     - Confusion matrices saved as PNG heatmaps.
  5. Compare results across the CS2 and VALORANT communities.
  6. Save models and visualizations to models/ and reports/.
  7. Provide end-to-end inference for new text.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import load_npz, spmatrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
TFIDF_DIR = DATA_DIR / "processed" / "tfidf"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["negative", "neutral", "positive"]
DISPLAY_CLASS_LABELS = ["Negative", "Neutral", "Positive"]


# ---------------------------------------------------------------------------
# 1. Load TF-IDF data and metadata
# ---------------------------------------------------------------------------
def load_data(
    tfidf_dir: Path = TFIDF_DIR,
) -> tuple[spmatrix, spmatrix, pd.DataFrame, pd.DataFrame, Any]:
    """Load TF-IDF feature matrices, train/test metadata, and the vectorizer."""
    logger.info("Loading data from: %s", tfidf_dir)
    x_train_path = tfidf_dir / "X_train.npz"
    x_test_path = tfidf_dir / "X_test.npz"
    train_meta_path = tfidf_dir / "train_metadata.csv"
    test_meta_path = tfidf_dir / "test_metadata.csv"
    vec_path = tfidf_dir / "tfidf_vectorizer.joblib"

    for path in (x_train_path, x_test_path, train_meta_path, test_meta_path, vec_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    x_train = load_npz(x_train_path)
    x_test = load_npz(x_test_path)
    train_meta = pd.read_csv(train_meta_path)
    test_meta = pd.read_csv(test_meta_path)
    vectorizer = joblib.load(vec_path)

    logger.info("X_train shape: %s | X_test shape: %s", x_train.shape, x_test.shape)
    logger.info("Train label distribution: %s", dict(train_meta["sentiment"].value_counts()))
    logger.info("Test label distribution: %s", dict(test_meta["sentiment"].value_counts()))

    return x_train, x_test, train_meta, test_meta, vectorizer


# ---------------------------------------------------------------------------
# 2. Train and tune models (GridSearchCV)
# ---------------------------------------------------------------------------
def train_and_tune_models(
    x_train: spmatrix,
    y_train: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Train Naive Bayes and SVM models using GridSearchCV with stratified
    5-fold cross-validation.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    trained_models: dict[str, dict[str, Any]] = {}

    # Define models and parameter grids
    model_configs = {
        "Multinomial Naive Bayes": {
            "estimator": MultinomialNB(),
            "param_grid": {"alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]},
            "filename": "multinomial_nb.joblib",
        },
        "Complement Naive Bayes": {
            "estimator": ComplementNB(),
            "param_grid": {"alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]},
            "filename": "complement_nb.joblib",
        },
        "Linear SVM": {
            "estimator": LinearSVC(random_state=random_state, max_iter=3000, dual="auto"),
            "param_grid": {"C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
            "filename": "linear_svm.joblib",
        },
    }

    for name, cfg in model_configs.items():
        logger.info("Tuning hyperparameters for: %s ...", name)
        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(x_train, y_train)

        best_estimator = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        logger.info(
            "[%s] Best parameters: %s | CV macro F1: %.4f",
            name, best_params, best_cv_score
        )

        # Save the model
        saved_model = best_estimator
        save_path = MODELS_DIR / cfg["filename"]
        joblib.dump(saved_model, save_path)
        logger.info("Model saved to: %s", save_path)

        trained_models[name] = {
            "model": saved_model,
            "raw_best_estimator": best_estimator,
            "best_params": best_params,
            "best_cv_macro_f1": best_cv_score,
            "save_path": save_path,
        }

    return trained_models


# ---------------------------------------------------------------------------
# 3. Evaluate on the test set
# ---------------------------------------------------------------------------
def evaluate_models(
    trained_models: dict[str, dict[str, Any]],
    x_test: spmatrix,
    test_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate all models on the test set and create a metrics table."""
    y_test = test_meta["sentiment"].to_numpy()
    metrics_list = []
    predictions: dict[str, np.ndarray] = {}
    conf_matrices: dict[str, np.ndarray] = {}

    for name, info in trained_models.items():
        model = info["model"]
        y_pred = model.predict(x_test)
        predictions[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
        conf_matrices[name] = cm

        metrics_list.append({
            "Algorithm": name,
            "Best Parameters": str(info["best_params"]),
            "CV Macro F1": round(info["best_cv_macro_f1"], 4),
            "Test Accuracy": round(acc, 4),
            "Test Precision (Macro)": round(prec_macro, 4),
            "Test Recall (Macro)": round(rec_macro, 4),
            "Test F1 (Macro)": round(f1_macro, 4),
            "Test F1 (Weighted)": round(f1_weighted, 4),
        })

        # Print the per-class classification report
        logger.info("\n--- Classification report: %s ---", name)
        print(classification_report(
            y_test,
            y_pred,
            labels=CLASS_LABELS,
            target_names=["Negative", "Neutral", "Positive"],
            digits=4,
            zero_division=0,
        ))

    df_metrics = pd.DataFrame(metrics_list)
    csv_report_path = REPORTS_DIR / "model_comparison_metrics.csv"
    df_metrics.to_csv(csv_report_path, index=False, encoding="utf-8-sig")
    logger.info("Comparison table saved to: %s", csv_report_path)

    return df_metrics, predictions, conf_matrices


# ---------------------------------------------------------------------------
# 4. Plot confusion matrices
# ---------------------------------------------------------------------------
def plot_confusion_matrices(
    conf_matrices: dict[str, np.ndarray],
    save_path: Path = FIGURES_DIR / "confusion_matrices.png",
) -> None:
    """Save confusion matrix plots for all models in a single image."""
    n_models = len(conf_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, conf_matrices.items()):
        # Annotate with raw counts and row percentages
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=DISPLAY_CLASS_LABELS,
            yticklabels=DISPLAY_CLASS_LABELS,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Confusion matrix plot saved to: %s", save_path)


# ---------------------------------------------------------------------------
# 5. Compare sentiment by community: CS2 and VALORANT
# ---------------------------------------------------------------------------
def analyze_community_sentiment(
    test_meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    save_path: Path = FIGURES_DIR / "sentiment_comparison_cs2_vs_valorant.png",
) -> pd.DataFrame:
    """
    Compare sentiment and model accuracy between the CS2 and VALORANT communities.
    """
    df_eval = test_meta.copy()
    for name, pred in predictions.items():
        df_eval[f"pred_{name}"] = pred

    # Use Linear SVM as the primary model for distribution analysis
    primary_model = "Linear SVM" if "Linear SVM" in predictions else list(predictions.keys())[0]

    # Calculate metrics for each game
    community_metrics = []
    for game in ["CS2", "VALORANT"]:
        mask = df_eval["game"] == game
        y_true = df_eval.loc[mask, "sentiment"].to_numpy()
        y_pred = df_eval.loc[mask, f"pred_{primary_model}"].to_numpy()

        acc = accuracy_score(y_true, y_pred)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        dist = pd.Series(y_pred).value_counts(normalize=True)

        community_metrics.append({
            "Community": game,
            "Test Samples": int(mask.sum()),
            "Accuracy": round(acc, 4),
            "Macro F1": round(f1_m, 4),
            "% Predicted Positive": round(dist.get("positive", 0.0) * 100, 2),
            "% Predicted Neutral": round(dist.get("neutral", 0.0) * 100, 2),
            "% Predicted Negative": round(dist.get("negative", 0.0) * 100, 2),
        })

    df_comm = pd.DataFrame(community_metrics)
    comm_csv_path = REPORTS_DIR / "community_comparison.csv"
    df_comm.to_csv(comm_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Community comparison saved to: %s", comm_csv_path)

    # Plot sentiment distributions for CS2 and VALORANT
    plot_df = (
        df_eval.groupby(["game", f"pred_{primary_model}"])
        .size()
        .unstack(fill_value=0)
    )
    # Convert values to percentages
    plot_df_pct = plot_df.div(plot_df.sum(axis=1), axis=0) * 100
    # Ensure columns are in the expected order
    plot_df_pct = plot_df_pct.reindex(columns=CLASS_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]  # red: negative, gray: neutral, green: positive
    plot_df_pct.plot(kind="bar", ax=ax, color=colors, edgecolor="black", width=0.6)

    ax.set_title(
        f"Comment Sentiment Distribution: CS2 and VALORANT\n({primary_model} Predictions)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Game Community", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.legend(title="Sentiment", labels=["Negative", "Neutral", "Positive"])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add value labels above each bar
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h + 1),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Community sentiment plot saved to: %s", save_path)

    return df_comm


# ---------------------------------------------------------------------------
# 6. Predict sentiment for a new comment
# ---------------------------------------------------------------------------
def predict_sentiment(
    text: str,
    model_name: str = "Linear SVM",
    vectorizer_path: Path = TFIDF_DIR / "tfidf_vectorizer.joblib",
) -> dict[str, Any]:
    """
    Predict sentiment for a new comment end to end.
    """
    from src.preprocessing.text_preprocessing import preprocess_text

    clean = preprocess_text(text)
    vectorizer = joblib.load(vectorizer_path)
    feat = vectorizer.transform([clean])

    model_file = "linear_svm.joblib" if model_name == "Linear SVM" else "multinomial_nb.joblib"
    model = joblib.load(MODELS_DIR / model_file)

    pred = model.predict(feat)[0]
    result: dict[str, Any] = {
        "original_text": text,
        "clean_text": clean,
        "model": model_name,
        "predicted_sentiment": pred,
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        classes = model.classes_
        result["probabilities"] = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}

    return result


# ---------------------------------------------------------------------------
# 7. Main execution pipeline
# ---------------------------------------------------------------------------
def run_training_pipeline() -> None:
    """Run model training, hyperparameter tuning, evaluation, and reporting."""
    print("=" * 70)
    print(" NAIVE BAYES & SVM TRAINING AND EVALUATION PIPELINE")
    print(" Reddit Sentiment Analysis: CS2 vs VALORANT")
    print("=" * 70)

    # 1. Load data
    x_train, x_test, train_meta, test_meta, vectorizer = load_data()
    y_train = train_meta["sentiment"].to_numpy()

    # 2. Train models and find the best parameters
    trained_models = train_and_tune_models(x_train, y_train, cv_splits=5)

    # 3. Evaluate on the test set
    df_metrics, predictions, conf_matrices = evaluate_models(trained_models, x_test, test_meta)

    # 4. Plot confusion matrices
    plot_confusion_matrices(conf_matrices)

    # 5. Compare CS2 and VALORANT communities
    df_comm = analyze_community_sentiment(test_meta, predictions)

    # 6. Print a summary of the results
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(df_metrics.to_string(index=False))

    print("\n" + "=" * 70)
    print(" COMMUNITY SENTIMENT ANALYSIS (CS2 vs VALORANT)")
    print("=" * 70)
    print(df_comm.to_string(index=False))

    print("\n" + "=" * 70)
    print(" SAMPLE PREDICTIONS FOR NEW COMMENTS")
    print("=" * 70)
    sample_tests = [
        "Subtick in CS2 is so much better now, smooth 200 fps and hitreg feels crisp!",
        "Valorant ranked matchmaking is pure garbage, smurfs everywhere and toxic players.",
        "Just a normal update note about patch 9.04 weapon balance changes.",
    ]
    for sample in sample_tests:
        res = predict_sentiment(sample, model_name="Linear SVM")
        sentiment_names = {
            "negative": "NEGATIVE",
            "neutral": "NEUTRAL",
            "positive": "POSITIVE",
        }
        print(f"\n[Comment] : {res['original_text']}")
        print(
            f"[Prediction] : "
            f"{sentiment_names.get(res['predicted_sentiment'], res['predicted_sentiment'])} "
            f"(Probabilities: {res.get('probabilities')})"
        )


if __name__ == "__main__":
    run_training_pipeline()
