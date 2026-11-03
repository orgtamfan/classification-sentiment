"""Ekstrak fitur TF-IDF dari kumpulan komentar Reddit yang telah diberi label."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FIELDS = {"clean_text", "sentiment"}
LABELS = {"positive", "neutral", "negative"}
METADATA_FIELDS = ("row_index", "game", "sentiment", "comment_id")


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def extract_tfidf_features(
    input_path: Path,
    output_dir: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int | None = 10_000,
) -> dict[str, Any]:
    """Bagi komentar berlabel, latih TF-IDF pada teks pelatihan, lalu simpan artefak."""
    if not 0 < test_size < 1:
        raise ValueError("test_size harus lebih besar dari 0 dan kurang dari 1.")
    if max_features is not None and max_features < 1:
        raise ValueError("max_features harus berupa bilangan bulat positif.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Berkas CSV masukan tidak ditemukan: {input_path}")

    with input_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        missing_fields = INPUT_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Kolom wajib tidak ditemukan pada CSV masukan: {missing}.")
        input_rows = list(reader)

    records: list[dict[str, str]] = []
    skipped_empty_text = 0
    skipped_unlabeled = 0
    for row_number, row in enumerate(input_rows, start=2):
        text = (row.get("clean_text") or "").strip()
        sentiment = (row.get("sentiment") or "").strip().lower()
        if not text:
            skipped_empty_text += 1
            continue
        if not sentiment:
            skipped_unlabeled += 1
            continue
        if sentiment not in LABELS:
            raise ValueError(
                f"Label sentimen pada baris {row_number} tidak didukung: "
                f"{sentiment!r}."
            )
        records.append(
            {
                "clean_text": text,
                "game": (row.get("game") or "").strip(),
                "sentiment": sentiment,
                "comment_id": (row.get("comment_id") or "").strip(),
            }
        )

    if not records:
        raise ValueError("Tidak ditemukan baris dengan clean_text dan sentiment yang terisi.")

    labels = [record["sentiment"] for record in records]
    label_counts = {
        label: labels.count(label)
        for label in sorted(set(labels))
    }
    if any(count < 2 for count in label_counts.values()):
        raise ValueError(
            "Setiap kelas sentimen harus memiliki sedikitnya dua baris untuk "
            "pembagian train/test secara stratifikasi."
        )

    row_indices = list(range(len(records)))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    train_texts = [records[index]["clean_text"] for index in train_indices]
    test_texts = [records[index]["clean_text"] for index in test_indices]
    train_labels = [labels[index] for index in train_indices]
    test_labels = [labels[index] for index in test_indices]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "X_train.npz", x_train)
    save_npz(output_dir / "X_test.npz", x_test)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")

    train_metadata = [
        {
            "row_index": matrix_index,
            "game": records[source_index]["game"],
            "sentiment": records[source_index]["sentiment"],
            "comment_id": records[source_index]["comment_id"],
        }
        for matrix_index, source_index in enumerate(train_indices)
    ]
    test_metadata = [
        {
            "row_index": matrix_index,
            "game": records[source_index]["game"],
            "sentiment": records[source_index]["sentiment"],
            "comment_id": records[source_index]["comment_id"],
        }
        for matrix_index, source_index in enumerate(test_indices)
    ]
    _write_metadata(output_dir / "train_metadata.csv", train_metadata)
    _write_metadata(output_dir / "test_metadata.csv", test_metadata)

    with (output_dir / "tfidf_features.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as feature_file:
        writer = csv.writer(feature_file)
        writer.writerow(("feature_index", "term"))
        writer.writerows(enumerate(vectorizer.get_feature_names_out()))

    return {
        "rows": len(records),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "features": x_train.shape[1],
        "skipped_empty_text": skipped_empty_text,
        "skipped_unlabeled": skipped_unlabeled,
        "label_counts": label_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ekstrak fitur TF-IDF train/test dari komentar berlabel."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "data/processed/sentiment_labels_vader.csv",
        help=(
            "Berkas CSV komentar berlabel (bawaan: "
            "data/processed/sentiment_labels_vader.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data/processed/tfidf",
        help="Folder untuk matriks TF-IDF, label, dan vectorizer.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi baris berlabel untuk data uji (bawaan: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed acak agar pembagian train/test dapat direproduksi.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10_000,
        help="Ukuran kosakata maksimum; gunakan 0 untuk tanpa batas.",
    )
    args = parser.parse_args()

    result = extract_tfidf_features(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features or None,
    )
    print(
        f"Ekstraksi TF-IDF disimpan ke {args.output_dir}: "
        f"{result['train_rows']} baris latih, {result['test_rows']} baris uji, "
        f"{result['features']} fitur."
    )
    print(
        f"{result['skipped_empty_text']} baris dengan teks kosong dan "
        f"{result['skipped_unlabeled']} baris tanpa label dilewati. "
        f"Jumlah label: {result['label_counts']}."
    )


if __name__ == "__main__":
    main()
