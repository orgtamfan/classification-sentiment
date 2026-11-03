"""Beri label sentimen otomatis pada komentar Reddit yang telah dibersihkan dengan VADER."""

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
    """Tulis label VADER untuk baris bersih berdasarkan teks komentar asli."""
    if not -1 <= negative_threshold < positive_threshold <= 1:
        raise ValueError(
            "Ambang harus memenuhi -1 <= negatif < positif <= 1."
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Berkas keluaran tidak boleh menimpa berkas CSV masukan.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Berkas CSV masukan tidak ditemukan: {input_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Berkas CSV untuk peninjauan sudah ada: {output_path}. "
            "Pilih lokasi --output lain atau hapus berkas setelah menyimpan perubahan."
        )

    with input_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        required_fields = {"game", "clean_text", "comment_text"}
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Kolom wajib tidak ditemukan pada CSV masukan: {missing}.")
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
            raise ValueError(f"Baris {row_number} tidak memiliki nilai game yang valid.")
        game = GAME_NAMES.get(raw_game.strip().lower())
        if game is None:
            raise ValueError(
                f"Nilai game pada baris {row_number} tidak didukung: {raw_game!r}."
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
            "Beri label sentimen otomatis pada komentar Reddit yang telah "
            "dibersihkan menggunakan VADER untuk klasifikasi Naive Bayes dan SVM."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "data/processed/comments_preprocessed.csv",
        help=(
            "Berkas CSV komentar hasil prapemrosesan (bawaan: "
            "data/processed/comments_preprocessed.csv)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data/processed/sentiment_labels_vader.csv",
        help=(
            "Berkas CSV keluaran berlabel VADER (bawaan: "
            "data/processed/sentiment_labels_vader.csv)."
        ),
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.05,
        help="Skor compound VADER pada atau di atas nilai ini diberi label positif.",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=-0.05,
        help="Skor compound VADER pada atau di bawah nilai ini diberi label negatif.",
    )
    args = parser.parse_args()

    counts = generate_vader_labels(
        input_path=args.input,
        output_path=args.output,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )
    print(
        f"CSV berlabel VADER disimpan ke {args.output}: "
        f"{counts['written']} komentar, "
        f"{counts['skipped_empty_text']} komentar kosong dilewati "
        f"(positif: {counts['positive']}, netral: {counts['neutral']}, "
        f"negatif: {counts['negative']})."
    )
    print(
        "Label VADER adalah hasil prediksi otomatis; tinjau label sebelum "
        "menggunakan data untuk pelatihan atau evaluasi."
"""  """    )


if __name__ == "__main__":
    main()
