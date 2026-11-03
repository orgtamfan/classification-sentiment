"""Split labeled Reddit comments by post and extract TF-IDF features."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FIELDS = {"clean_text", "sentiment", "comment_id", "post_id"}
LABELS = {"positive", "neutral", "negative"}
REVIEW_STATUSES = {"pending", "accepted", "corrected"}
METADATA_FIELDS = (
    "row_index",
    "game",
    "sentiment",
    "comment_id",
    "post_id",
    "clean_text",
)
logger = logging.getLogger(__name__)


def _write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _validate_group_splits(
    labels: list[str],
    groups: list[str],
    *,
    test_size: float,
    random_state: int,
    cv_splits: int,
) -> tuple[list[int], list[int], list[tuple[list[int], list[int]]]]:
    """Create group-disjoint holdout and CV splits with every class present."""
    unique_groups = set(groups)
    if len(unique_groups) < cv_splits:
        raise ValueError(
            f"At least {cv_splits} unique post_id groups are required; "
            f"found {len(unique_groups)}."
        )

    all_labels = set(labels)
    splitter = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    cv_indices = list(splitter.split(range(len(labels)), labels, groups))
    for fold_number, (train_idx, validation_idx) in enumerate(cv_indices, start=1):
        train_labels = {labels[index] for index in train_idx}
        validation_labels = {labels[index] for index in validation_idx}
        if train_labels != all_labels or validation_labels != all_labels:
            raise ValueError(
                f"Cross-validation fold {fold_number} does not contain every "
                "sentiment class in both training and validation data."
            )
        train_groups = {groups[index] for index in train_idx}
        validation_groups = {groups[index] for index in validation_idx}
        if train_groups & validation_groups:
            raise ValueError(f"Post leakage detected in cross-validation fold {fold_number}.")

    valid_holdouts: list[tuple[float, int, list[int], list[int]]] = []
    for train_idx, test_idx in cv_indices:
        train_labels = {labels[index] for index in train_idx}
        test_labels = {labels[index] for index in test_idx}
        if train_labels != all_labels or test_labels != all_labels:
            continue
        actual_test_size = len(test_idx) / len(labels)
        valid_holdouts.append(
            (abs(actual_test_size - test_size), len(test_idx), train_idx, test_idx)
        )

    if not valid_holdouts:
        raise ValueError(
            "Could not create a post-disjoint holdout split containing every "
            "sentiment class in both training and test data."
        )

    _, _, train_idx, test_idx = min(valid_holdouts, key=lambda item: item[0])
    return train_idx, test_idx, cv_indices


def extract_tfidf_features(
    input_path: Path,
    output_dir: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int | None = 10_000,
    cv_splits: int = 5,
) -> dict[str, Any]:
    """Split comments by post, fit TF-IDF, and save data and metadata."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if max_features is not None and max_features < 1:
        raise ValueError("max_features must be a positive integer.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        missing_fields = INPUT_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Input CSV is missing required columns: {missing}.")
        input_rows = list(reader)

    if not input_rows:
        raise ValueError("Input CSV contains no data rows.")

    has_review_status = "review_status" in (reader.fieldnames or ())
    records: list[dict[str, str]] = []
    skipped_empty_text = 0
    skipped_unlabeled = 0
    seen_comment_ids: set[str] = set()

    for row_number, row in enumerate(input_rows, start=2):
        comment_id = (row.get("comment_id") or "").strip()
        post_id = (row.get("post_id") or "").strip()
        if not comment_id:
            raise ValueError(f"Row {row_number} has an empty comment_id.")
        if not post_id:
            raise ValueError(f"Row {row_number} has an empty post_id.")
        if comment_id in seen_comment_ids:
            raise ValueError(f"Duplicate comment_id {comment_id!r} at row {row_number}.")
        seen_comment_ids.add(comment_id)

        review_status = (row.get("review_status") or "pending").strip().lower()
        if review_status not in REVIEW_STATUSES:
            raise ValueError(
                f"Row {row_number} has unsupported review_status: "
                f"{review_status!r}. Expected one of {sorted(REVIEW_STATUSES)}."
            )

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
                f"Row {row_number} has unsupported sentiment label: "
                f"{sentiment!r}."
            )

        records.append(
            {
                "clean_text": text,
                "game": (row.get("game") or "").strip(),
                "sentiment": sentiment,
                "comment_id": comment_id,
                "post_id": post_id,
            }
        )

    if not records:
        raise ValueError("No rows with non-empty clean_text and sentiment found.")

    labels = [record["sentiment"] for record in records]
    groups = [record["post_id"] for record in records]
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    if set(label_counts) != LABELS:
        raise ValueError(
            f"Expected all sentiment classes {sorted(LABELS)}; "
            f"found {sorted(label_counts)}."
        )

    train_indices, test_indices, _ = _validate_group_splits(
        labels,
        groups,
        test_size=test_size,
        random_state=random_state,
        cv_splits=cv_splits,
    )
    train_texts = [records[index]["clean_text"] for index in train_indices]
    test_texts = [records[index]["clean_text"] for index in test_indices]

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    duplicate_text_counts: dict[str, int] = {}
    for record in records:
        duplicate_text_counts[record["clean_text"]] = (
            duplicate_text_counts.get(record["clean_text"], 0) + 1
        )
    duplicate_text_rows = sum(
        count for count in duplicate_text_counts.values() if count > 1
    )
    train_text_set = {records[index]["clean_text"] for index in train_indices}
    test_text_set = {records[index]["clean_text"] for index in test_indices}
    shared_texts = train_text_set & test_text_set
    if duplicate_text_rows:
        logger.warning(
            "%d rows share cleaned text with another comment; these are retained "
            "as potentially legitimate repeated comments.",
            duplicate_text_rows,
        )
    if shared_texts:
        logger.warning(
            "%d identical cleaned texts occur in both train and test splits "
            "across different posts.",
            len(shared_texts),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "X_train.npz", x_train)
    save_npz(output_dir / "X_test.npz", x_test)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")

    train_metadata = [
        {
            "row_index": matrix_index,
            **records[source_index],
        }
        for matrix_index, source_index in enumerate(train_indices)
    ]
    test_metadata = [
        {
            "row_index": matrix_index,
            **records[source_index],
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

    test_label_counts = {
        label: sum(labels[index] == label for index in test_indices)
        for label in sorted(LABELS)
    }
    train_label_counts = {
        label: sum(labels[index] == label for index in train_indices)
        for label in sorted(LABELS)
    }
    if any(count == 0 for count in test_label_counts.values()) or any(
        count == 0 for count in train_label_counts.values()
    ):
        raise ValueError("A sentiment class is missing after the grouped split.")

    return {
        "rows": len(records),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "features": x_train.shape[1],
        "unique_posts": len(set(groups)),
        "train_posts": len({groups[index] for index in train_indices}),
        "test_posts": len({groups[index] for index in test_indices}),
        "skipped_empty_text": skipped_empty_text,
        "skipped_unlabeled": skipped_unlabeled,
        "duplicate_text_rows": duplicate_text_rows,
        "shared_train_test_texts": len(shared_texts),
        "label_counts": label_counts,
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "review_status_column_present": has_review_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split comments by post and extract train/test TF-IDF features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "data/processed/sentiment_labels_vader.csv",
        help=(
            "Labeled comments CSV (default: "
            "data/processed/sentiment_labels_vader.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data/processed/tfidf",
        help="Directory for TF-IDF matrices, labels, and vectorizer.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Target fraction of comments for the test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible grouped splits.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10_000,
        help="Maximum vocabulary size; use 0 for no limit.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of stratified group folds (default: 5).",
    )
    args = parser.parse_args()

    result = extract_tfidf_features(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features or None,
        cv_splits=args.cv_splits,
    )
    print(
        f"TF-IDF extraction saved to {args.output_dir}: "
        f"{result['train_rows']} train rows from {result['train_posts']} posts, "
        f"{result['test_rows']} test rows from {result['test_posts']} posts, "
        f"{result['features']} features."
    )
    print(
        f"Train class counts: {result['train_label_counts']}; "
        f"test class counts: {result['test_label_counts']}."
    )
    print(
        f"Skipped {result['skipped_empty_text']} empty-text rows and "
        f"{result['skipped_unlabeled']} unlabeled rows. "
        f"Duplicate cleaned-text rows: {result['duplicate_text_rows']}; "
        f"identical texts shared across splits: {result['shared_train_test_texts']}."
    )
    if not result["review_status_column_present"]:
        print(
            "review_status is absent; all rows are treated as pending review. "
            "Human annotation metrics are not available."
        )


if __name__ == "__main__":
    main()
