"""Automatically label preprocessed Reddit comments with VADER sentiment scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUTPUT_FIELDS = (
    "game",
    "clean_text",
    "sentiment",
    "vader_compound",
    "annotation_status",
    "comment_text",
    "comment_id",
    "post_id",
    "subreddit",
    "post_title",
    "comment_created_at",
)
BASE_DIR = Path(__file__).resolve().parent.parent
GAME_NAMES = {"cs2": "CS2", "valorant": "VALORANT"}


def generate_vader_labels(
    input_path: Path,
    output_path: Path,
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
) -> dict[str, int]:
    """Write VADER labels for preprocessed rows using the original comment text."""
    if not -1 <= negative_threshold < positive_threshold <= 1:
        raise ValueError(
            "Thresholds must satisfy -1 <= negative < positive <= 1."
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must not overwrite the input CSV.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Review CSV already exists: {output_path}. "
            "Choose another --output path or remove it after preserving any edits."
        )

    with input_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        required_fields = {"game", "clean_text", "comment_text"}
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Input CSV is missing required columns: {missing}.")
        records = list(reader)

    analyzer = SentimentIntensityAnalyzer()
    counts = {
        "written": 0,
        "skipped_empty_text": 0,
        "positive": 0,
        "neutral": 0,
        "negative": 0,
    }
    output_rows: list[dict[str, Any]] = []

    for row_number, record in enumerate(records, start=1):
        raw_text = record.get("clean_text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        raw_comment = record.get("comment_text")
        comment_text = (
            raw_comment.strip() if isinstance(raw_comment, str) else ""
        )
        if not text or not comment_text:
            counts["skipped_empty_text"] += 1
            continue

        raw_game = record.get("game")
        if not isinstance(raw_game, str):
            raise ValueError(f"Record {row_number} has no valid game value.")
        game = GAME_NAMES.get(raw_game.strip().lower())
        if game is None:
            raise ValueError(
                f"Record {row_number} has unsupported game value: {raw_game!r}."
            )

        compound = analyzer.polarity_scores(comment_text)["compound"]
        if compound >= positive_threshold:
            sentiment = "positive"
        elif compound <= negative_threshold:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        output_rows.append(
            {
                "game": game,
                "clean_text": text,
                "sentiment": sentiment,
                "vader_compound": compound,
                "annotation_status": "auto_labeled_vader",
                "comment_text": comment_text,
                "comment_id": record.get("comment_id", ""),
                "post_id": record.get("post_id", ""),
                "subreddit": record.get("subreddit", ""),
                "post_title": record.get("post_title", ""),
                "comment_created_at": record.get("comment_created_at", ""),
            }
        )
        counts["written"] += 1
        counts[sentiment] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically label preprocessed Reddit comments with VADER "
            "for Naive Bayes and SVM classification."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "data/processed/comments_preprocessed.csv",
        help=(
            "Preprocessed comments CSV (default: "
            "data/processed/comments_preprocessed.csv)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data/processed/sentiment_labels_vader.csv",
        help=(
            "VADER-labeled CSV output (default: "
            "data/processed/sentiment_labels_vader.csv)."
        ),
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.05,
        help="VADER compound score at or above this value is positive.",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=-0.05,
        help="VADER compound score at or below this value is negative.",
    )
    args = parser.parse_args()

    counts = generate_vader_labels(
        input_path=args.input,
        output_path=args.output,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )
    print(
        f"VADER-labeled CSV saved to {args.output}: "
        f"{counts['written']} comments, "
        f"{counts['skipped_empty_text']} empty comments skipped "
        f"(positive: {counts['positive']}, neutral: {counts['neutral']}, "
        f"negative: {counts['negative']})."
    )
    print(
        "VADER labels are automatic predictions; review them before using "
        "the data for training or evaluation."
    )


if __name__ == "__main__":
    main()
