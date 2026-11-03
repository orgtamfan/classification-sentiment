"""
Modul pelatihan dan evaluasi model untuk penelitian:
"Klasifikasi Sentimen Komentar Reddit pada Komunitas Counter Strike 2 dan VALORANT
Menggunakan Algoritma Naive Bayes dan SVM"

Fitur:
  1. Pelatihan Naive Bayes (MultinomialNB dan ComplementNB) dengan pencarian
     hyperparameter alpha.
  2. Pelatihan Support Vector Machine (LinearSVC) dengan pencarian
     hyperparameter C.
  3. Validasi silang terstratifikasi 5-fold.
  4. Evaluasi menyeluruh pada data uji:
     - Akurasi, presisi, recall, serta F1-score makro dan berbobot.
     - Matriks kebingungan (heatmap disimpan sebagai PNG).
  5. Analisis perbandingan komunitas game CS2 dan VALORANT.
  6. Penyimpanan model dan visualisasi ke folder models/ dan reports/.
  7. Fungsi inferensi/prediksi untuk teks baru secara menyeluruh.
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

# Atur pencatatan log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Folder proyek
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
DISPLAY_CLASS_LABELS = ["Negatif", "Netral", "Positif"]


# ---------------------------------------------------------------------------
# 1. Memuat data TF-IDF dan metadata
# ---------------------------------------------------------------------------
def load_data(
    tfidf_dir: Path = TFIDF_DIR,
) -> tuple[spmatrix, spmatrix, pd.DataFrame, pd.DataFrame, Any]:
    """Muat matriks fitur TF-IDF, metadata latih/uji, dan objek vectorizer."""
    logger.info("Memuat data dari: %s", tfidf_dir)
    x_train_path = tfidf_dir / "X_train.npz"
    x_test_path = tfidf_dir / "X_test.npz"
    train_meta_path = tfidf_dir / "train_metadata.csv"
    test_meta_path = tfidf_dir / "test_metadata.csv"
    vec_path = tfidf_dir / "tfidf_vectorizer.joblib"

    for path in (x_train_path, x_test_path, train_meta_path, test_meta_path, vec_path):
        if not path.is_file():
            raise FileNotFoundError(f"File penting tidak ditemukan: {path}")

    x_train = load_npz(x_train_path)
    x_test = load_npz(x_test_path)
    train_meta = pd.read_csv(train_meta_path)
    test_meta = pd.read_csv(test_meta_path)
    vectorizer = joblib.load(vec_path)

    logger.info("X_train shape: %s | X_test shape: %s", x_train.shape, x_test.shape)
    logger.info("Distribusi label Train: %s", dict(train_meta["sentiment"].value_counts()))
    logger.info("Distribusi label Test : %s", dict(test_meta["sentiment"].value_counts()))

    return x_train, x_test, train_meta, test_meta, vectorizer


# ---------------------------------------------------------------------------
# 2. Pelatihan dan pencarian parameter model (GridSearchCV)
# ---------------------------------------------------------------------------
def train_and_tune_models(
    x_train: spmatrix,
    y_train: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Latih model Naive Bayes dan SVM menggunakan GridSearchCV dengan validasi
    silang terstratifikasi 5-fold.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    trained_models: dict[str, dict[str, Any]] = {}

    # Definisi model dan kisi parameter
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
        logger.info("Memulai pencarian hyperparameter untuk: %s ...", name)
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
            "[%s] Parameter terbaik: %s | F1 makro validasi silang: %.4f",
            name, best_params, best_cv_score
        )

        # Simpan model ke penyimpanan
        saved_model = best_estimator
        save_path = MODELS_DIR / cfg["filename"]
        joblib.dump(saved_model, save_path)
        logger.info("Model tersimpan di: %s", save_path)

        trained_models[name] = {
            "model": saved_model,
            "raw_best_estimator": best_estimator,
            "best_params": best_params,
            "best_cv_macro_f1": best_cv_score,
            "save_path": save_path,
        }

    return trained_models


# ---------------------------------------------------------------------------
# 3. Evaluasi menyeluruh pada data uji
# ---------------------------------------------------------------------------
def evaluate_models(
    trained_models: dict[str, dict[str, Any]],
    x_test: spmatrix,
    test_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluasi kinerja semua model pada data uji dan buat tabel metrik."""
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
            "Algoritma": name,
            "Parameter Terbaik": str(info["best_params"]),
            "F1 Makro Validasi Silang": round(info["best_cv_macro_f1"], 4),
            "Akurasi Data Uji": round(acc, 4),
            "Presisi Makro Data Uji": round(prec_macro, 4),
            "Recall Makro Data Uji": round(rec_macro, 4),
            "F1 Makro Data Uji": round(f1_macro, 4),
            "F1 Berbobot Data Uji": round(f1_weighted, 4),
        })

        # Tampilkan laporan klasifikasi untuk setiap kelas
        logger.info("\n--- Laporan klasifikasi: %s ---", name)
        print(classification_report(
            y_test,
            y_pred,
            labels=CLASS_LABELS,
            target_names=DISPLAY_CLASS_LABELS,
            digits=4,
            zero_division=0,
        ))

    df_metrics = pd.DataFrame(metrics_list)
    csv_report_path = REPORTS_DIR / "model_comparison_metrics.csv"
    df_metrics.to_csv(csv_report_path, index=False, encoding="utf-8-sig")
    logger.info("Tabel perbandingan tersimpan di: %s", csv_report_path)

    return df_metrics, predictions, conf_matrices


# ---------------------------------------------------------------------------
# 4. Visualisasi matriks kebingungan
# ---------------------------------------------------------------------------
def plot_confusion_matrices(
    conf_matrices: dict[str, np.ndarray],
    save_path: Path = FIGURES_DIR / "confusion_matrices.png",
) -> None:
    """Simpan grafik matriks kebingungan semua model dalam satu gambar."""
    n_models = len(conf_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, conf_matrices.items()):
        # Buat label yang menampilkan nilai dan persentase per baris
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
        ax.set_title(f"Matriks Kebingungan: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Label Prediksi", fontsize=10)
        ax.set_ylabel("Label Sebenarnya", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Grafik Confusion Matrix tersimpan di: %s", save_path)


# ---------------------------------------------------------------------------
# 5. Analisis perbandingan tiap komunitas: CS2 dan VALORANT
# ---------------------------------------------------------------------------
def analyze_community_sentiment(
    test_meta: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    save_path: Path = FIGURES_DIR / "sentiment_comparison_cs2_vs_valorant.png",
) -> pd.DataFrame:
    """
    Analisis perbedaan sentimen dan akurasi model antara komunitas CS2 dan VALORANT.
    """
    df_eval = test_meta.copy()
    for name, pred in predictions.items():
        df_eval[f"pred_{name}"] = pred

    # Gunakan Linear SVM sebagai model utama untuk analisis distribusi
    primary_model = "Linear SVM" if "Linear SVM" in predictions else list(predictions.keys())[0]

    # Hitung kinerja tiap game
    community_metrics = []
    for game in ["CS2", "VALORANT"]:
        mask = df_eval["game"] == game
        y_true = df_eval.loc[mask, "sentiment"].to_numpy()
        y_pred = df_eval.loc[mask, f"pred_{primary_model}"].to_numpy()

        acc = accuracy_score(y_true, y_pred)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        dist = pd.Series(y_pred).value_counts(normalize=True)

        community_metrics.append({
            "Komunitas": game,
            "Total Data Uji": int(mask.sum()),
            "Akurasi": round(acc, 4),
            "F1 Makro": round(f1_m, 4),
            "% Prediksi Positif": round(dist.get("positive", 0.0) * 100, 2),
            "% Prediksi Netral": round(dist.get("neutral", 0.0) * 100, 2),
            "% Prediksi Negatif": round(dist.get("negative", 0.0) * 100, 2),
        })

    df_comm = pd.DataFrame(community_metrics)
    comm_csv_path = REPORTS_DIR / "community_comparison.csv"
    df_comm.to_csv(comm_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Analisis komparatif komunitas tersimpan di: %s", comm_csv_path)

    # Buat grafik perbandingan sentimen CS2 dan VALORANT
    plot_df = (
        df_eval.groupby(["game", f"pred_{primary_model}"])
        .size()
        .unstack(fill_value=0)
    )
    # Ubah nilai menjadi persentase
    plot_df_pct = plot_df.div(plot_df.sum(axis=1), axis=0) * 100
    # Pastikan urutan kolom sesuai
    plot_df_pct = plot_df_pct.reindex(columns=CLASS_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]  # merah: negatif, abu-abu: netral, hijau: positif
    plot_df_pct.plot(kind="bar", ax=ax, color=colors, edgecolor="black", width=0.6)

    ax.set_title(
        f"Distribusi Sentimen Komentar: CS2 dan VALORANT\n(Prediksi Model {primary_model})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Komunitas Game", fontsize=11)
    ax.set_ylabel("Persentase (%)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.legend(title="Sentimen", labels=["Negatif", "Netral", "Positif"])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Tampilkan nilai di atas setiap batang
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
    logger.info("Grafik distribusi sentimen komunitas tersimpan di: %s", save_path)

    return df_comm


# ---------------------------------------------------------------------------
# 6. Fungsi inferensi sentimen komentar baru
# ---------------------------------------------------------------------------
def predict_sentiment(
    text: str,
    model_name: str = "Linear SVM",
    vectorizer_path: Path = TFIDF_DIR / "tfidf_vectorizer.joblib",
) -> dict[str, Any]:
    """
    Prediksi sentimen untuk satu teks komentar baru secara menyeluruh.
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
# 7. Alur eksekusi utama
# ---------------------------------------------------------------------------
def run_training_pipeline() -> None:
    """Jalankan seluruh proses pelatihan, pencarian parameter, evaluasi, dan pembuatan laporan."""
    print("=" * 70)
    print(" ALUR PELATIHAN DAN EVALUASI NAIVE BAYES & SVM")
    print(" Penelitian Sentimen Reddit: CS2 vs VALORANT")
    print("=" * 70)

    # 1. Muat data
    x_train, x_test, train_meta, test_meta, vectorizer = load_data()
    y_train = train_meta["sentiment"].to_numpy()

    # 2. Latih dan cari parameter terbaik untuk model
    trained_models = train_and_tune_models(x_train, y_train, cv_splits=5)

    # 3. Evaluasi pada data uji
    df_metrics, predictions, conf_matrices = evaluate_models(trained_models, x_test, test_meta)

    # 4. Buat grafik matriks kebingungan
    plot_confusion_matrices(conf_matrices)

    # 5. Analisis komparatif CS2 vs VALORANT
    df_comm = analyze_community_sentiment(test_meta, predictions)

    # 6. Tampilkan ringkasan hasil
    print("\n" + "=" * 70)
    print(" RINGKASAN HASIL PERBANDINGAN MODEL")
    print("=" * 70)
    print(df_metrics.to_string(index=False))

    print("\n" + "=" * 70)
    print(" ANALISIS SENTIMEN KOMUNITAS (CS2 vs VALORANT)")
    print("=" * 70)
    print(df_comm.to_string(index=False))

    print("\n" + "=" * 70)
    print(" CONTOH PREDIKSI KOMENTAR BARU")
    print("=" * 70)
    sample_tests = [
        "Subtick in CS2 is so much better now, smooth 200 fps and hitreg feels crisp!",
        "Valorant ranked matchmaking is pure garbage, smurfs everywhere and toxic players.",
        "Just a normal update note about patch 9.04 weapon balance changes.",
    ]
    for sample in sample_tests:
        res = predict_sentiment(sample, model_name="Linear SVM")
        sentiment_names = {
            "negative": "NEGATIF",
            "neutral": "NETRAL",
            "positive": "POSITIF",
        }
        print(f"\n[Komentar] : {res['original_text']}")
        print(
            f"[Prediksi] : "
            f"{sentiment_names.get(res['predicted_sentiment'], res['predicted_sentiment'])} "
            f"(Probabilitas: {res.get('probabilities')})"
        )


if __name__ == "__main__":
    run_training_pipeline()
