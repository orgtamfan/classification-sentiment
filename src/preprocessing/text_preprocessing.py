"""
Text preprocessing module for the research project:
"Reddit Comment Sentiment Classification in Counter-Strike 2 and VALORANT
Communities Using Naive Bayes and SVM Algorithms"

Preprocessing steps:
    1. Decode HTML entities
    2. Remove URLs and links
    3. Remove Reddit mentions (u/user, r/subreddit)
    4. Remove Reddit Markdown syntax (bold, italics, headings, tables, quotes, etc.)
    5. Remove Unicode emoji and emoticons
    6. Remove ASCII text emoticons
    7. Remove non-ASCII and corrupted characters
    8. Convert text to lowercase
    9. Normalize slang and gaming terminology
   10. Remove standalone numbers
   11. Remove punctuation
   12. Remove English stopwords
   13. Tokenize
   14. Remove short tokens (< 2 characters)
   15. Apply Porter stemming
   16. Remove empty or whitespace-only tokens
   17. Join tokens back into a string
"""

import re
import json
import html
import string
import unicodedata
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download required NLTK resources if they are missing
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    ("tokenizers/punkt",          "punkt"),
    ("tokenizers/punkt_tab",      "punkt_tab"),
    ("corpora/stopwords",         "stopwords"),
]

for _path, _pkg in _NLTK_RESOURCES:
    try:
        nltk.data.find(_path)
    except LookupError:
        logger.info("Downloading NLTK resource: %s", _pkg)
        nltk.download(_pkg, quiet=True)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[2]      # project root
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
SLANG_PATH = DATA_DIR / "slang_dict.json"

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the slang and gaming terminology dictionary
# ---------------------------------------------------------------------------
def _load_slang_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        logger.warning("Slang dictionary file not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw: dict = json.load(f)
    # Normalize all keys and values to lowercase and trim extra whitespace.
    return {k.strip().lower(): v.strip().lower() for k, v in raw.items()}

SLANG_DICT: dict[str, str] = _load_slang_dict(SLANG_PATH)
logger.info("Loaded %d slang dictionary entries", len(SLANG_DICT))

# ---------------------------------------------------------------------------
# English stopwords
# ---------------------------------------------------------------------------
NEGATION_WORDS = frozenset(
    {
        "ain", "aint", "aren", "arent", "barely", "cannot", "cant", "couldn",
        "couldnt", "didn", "didnt", "doesn", "doesnt", "don", "dont", "hadn",
        "hadnt", "hasn", "hasnt", "haven", "havent", "hardly", "isn", "isnt",
        "mightn", "mightnt", "mustn", "mustnt", "needn", "neednt", "neither",
        "never", "no", "nobody", "none", "nor", "not", "nothing", "nowhere",
        "shan", "shant", "shouldn", "shouldnt", "scarcely", "wasn", "wasnt",
        "weren", "werent", "without", "won", "wont", "wouldn", "wouldnt",
    }
)
STOP_WORDS: frozenset[str] = frozenset(stopwords.words("english")) - NEGATION_WORDS

# ---------------------------------------------------------------------------
# Porter stemmer
# ---------------------------------------------------------------------------
STEMMER = PorterStemmer()

# ---------------------------------------------------------------------------
# Compile regex patterns once for better efficiency
# ---------------------------------------------------------------------------
# URL / hyperlink
_RE_URL = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+",
    re.IGNORECASE,
)

# Reddit mentions: u/username or r/subreddit (including /r/ and /u/)
_RE_REDDIT_MENTION = re.compile(
    r"/?[ur]/\w+",
    re.IGNORECASE,
)

# Markdown Reddit
_RE_MARKDOWN_HEADER   = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_MARKDOWN_BOLD_IT  = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_RE_MARKDOWN_STRIKE   = re.compile(r"~~(.+?)~~")
_RE_MARKDOWN_CODE     = re.compile(r"`{1,3}.+?`{1,3}", re.DOTALL)
_RE_MARKDOWN_LINK     = re.compile(r"\[([^\]]*)\]\([^\)]*\)")   # [text](url)
_RE_MARKDOWN_QUOTE    = re.compile(r"^>\s*", re.MULTILINE)
_RE_MARKDOWN_TABLE    = re.compile(r"^\|.*\|$", re.MULTILINE)
_RE_MARKDOWN_HR       = re.compile(r"^[-_*]{3,}\s*$", re.MULTILINE)

# ASCII emoticons (must be handled before punctuation is removed)
# Examples: :) :( ;) :D :P :-) xD XD, etc.
_RE_ASCII_EMOTICON = re.compile(
    r"(?::|;|=)(?:-)?(?:\)|D|P|\(|\/|\\|\||\}|\{|@|3|>|<|O|0)"  # eyes, nose, and mouth
    r"|(?:\^_?\^|>.<|T_T|;_;|>_<|O_O|0_0|=_=|@_@|\*_\*|x_x|X_X|UwU|OwO)"
    r"|(?:xD|XD|xd|D:|:D|:\)|:\(|:\||:P|:p|:o|:O|:\*|:3|<3|</3|>_>|<_<)"
    r"|(?:8\)|B\)|\^o\^|\(ﾉ◕ヮ◕\)ﾉ|\(╯°□°\)╯)",
    re.IGNORECASE,
)

# Repeated characters (more than two consecutive occurrences → two)
_RE_REPEAT_CHAR = re.compile(r"(.)\1{2,}")

# Standalone numbers (not part of a word)
_RE_STANDALONE_NUMBER = re.compile(r"\b\d+\b")

# Characters other than letters and whitespace (after earlier processing)
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")

# Repeated whitespace
_RE_MULTI_SPACE = re.compile(r"\s{2,}")

# Corrupted or replacement characters
_RE_CORRUPT = re.compile(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]")


# ---------------------------------------------------------------------------
# Helper: remove Unicode emoji
# ---------------------------------------------------------------------------
def _remove_unicode_emoji(text: str) -> str:
    """Remove emoji and symbols using Unicode character categories."""
    result = []
    for char in text:
        cat = unicodedata.category(char)
        cp  = ord(char)
        # Category "So" means other symbols and includes emoji.
        # Also remove surrogates, private-use characters, and common emoji blocks.
        if cat in {"So", "Cs", "Co"} or (
            0x1F300 <= cp <= 0x1FAFF or   # Miscellaneous symbols, pictographs, and emoticons
            0x2600  <= cp <= 0x27BF or    # Miscellaneous symbols and dingbats
            0xFE00  <= cp <= 0xFE0F or    # Variation selectors
            0x200D  == cp              or  # Zero-width joiner
            0xFE0F  == cp              or  # Variation Selector-16
            0x20E3  == cp              or  # Combining Enclosing Keycap
            0x1F1E0 <= cp <= 0x1F1FF       # Regional indicator symbols (flags)
        ):
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


# ---------------------------------------------------------------------------
# Helper: normalize slang using a single regex alternation pattern.
# This is faster than checking each term with a separate pattern.
# ---------------------------------------------------------------------------
def _build_slang_pattern(slang_dict: dict[str, str]) -> re.Pattern | None:
    """
    Combine all slang keys into a single regex alternation pattern.
    Put longer keys first so multiword phrases take precedence.
    """
    if not slang_dict:
        return None
    sorted_keys = sorted(slang_dict, key=len, reverse=True)
    pattern_str  = r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b"
    return re.compile(pattern_str, re.IGNORECASE)

# Compile once when the module is imported
_SLANG_PATTERN: re.Pattern | None = _build_slang_pattern(SLANG_DICT)

def _normalize_slang(text: str) -> str:
    """
    Replace slang terms and phrases with their standard forms in one regex pass.
    Longer phrases take precedence because keys are sorted by descending length.
    """
    if _SLANG_PATTERN is None:
        return text
    return _SLANG_PATTERN.sub(lambda m: SLANG_DICT[m.group(0).lower()], text)


# ---------------------------------------------------------------------------
# Main single-text preprocessing function
# ---------------------------------------------------------------------------
def preprocess_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    apply_stemming:   bool = True,
    min_token_len:    int  = 2,
) -> str:
    """
    Clean a Reddit comment by applying the complete preprocessing pipeline.

    Parameters
    ----------
    text : str
        Raw Reddit comment text.
    remove_stopwords : bool
        Remove English stopwords (default: True).
    apply_stemming : bool
        Apply Porter stemming (default: True).
    min_token_len : int
        Minimum token length to keep (default: 2).

    Returns
    -------
    str
        Cleaned text ready for feature extraction.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Decode HTML entities (&amp; → &, &lt; → <, &#39; → ', etc.)
    text = html.unescape(text)

    # 2. Remove URLs and hyperlinks
    text = _RE_URL.sub(" ", text)

    # 3. Remove Reddit mentions (u/..., r/...)
    text = _RE_REDDIT_MENTION.sub(" ", text)

    # 4. Remove Reddit Markdown syntax
    text = _RE_MARKDOWN_HEADER.sub("", text)
    text = _RE_MARKDOWN_CODE.sub(" ", text)
    text = _RE_MARKDOWN_LINK.sub(r"\1", text)   # Preserve link text
    text = _RE_MARKDOWN_BOLD_IT.sub(r"\1", text)
    text = _RE_MARKDOWN_STRIKE.sub(r"\1", text)
    text = _RE_MARKDOWN_QUOTE.sub("", text)
    text = _RE_MARKDOWN_TABLE.sub(" ", text)
    text = _RE_MARKDOWN_HR.sub(" ", text)

    # 5. Remove Unicode emoji
    text = _remove_unicode_emoji(text)

    # 6. Remove ASCII emoticons
    text = _RE_ASCII_EMOTICON.sub(" ", text)

    # 7. Remove non-ASCII and corrupted characters (e.g., replacement char U+FFFD)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 8. Convert to lowercase
    text = text.lower()

    # 9. Normalize repeated characters (sooooo → soo, hahaha → haha)
    text = _RE_REPEAT_CHAR.sub(r"\1\1", text)

    # 10. Normalize slang and gaming terminology
    text = _normalize_slang(text)

    # 11. Remove standalone numbers
    text = _RE_STANDALONE_NUMBER.sub(" ", text)

    # 12. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 13. Remove characters other than letters and whitespace
    text = _RE_NON_ALPHA.sub(" ", text)

    # 14. Normalize whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    # 15. Tokenize
    tokens: list[str] = word_tokenize(text)

    # 16. Remove short tokens
    tokens = [t for t in tokens if len(t) >= min_token_len]

    # 17. Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # 18. Apply stemming
    if apply_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]

    # 19. Remove short tokens (stemming may shorten them)
    tokens = [t for t in tokens if len(t) >= min_token_len]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Batch function: preprocess an entire DataFrame
# ---------------------------------------------------------------------------
def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    *,
    remove_stopwords: bool = True,
    apply_stemming:   bool = True,
    min_token_len:    int  = 2,
    clean_col:        str  = "clean_text",
) -> pd.DataFrame:
    """
    Preprocess all text values in a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the raw text column.
    text_col : str
        Name of the raw text column (default: "comment_text").
    clean_col : str
        Name of the cleaned text output column (default: "clean_text").

    Returns
    -------
    pd.DataFrame
        DataFrame with the cleaned text column added.
    """
    logger.info("Preprocessing %d text rows ...", len(df))
    df = df.copy()
    df[clean_col] = df[text_col].apply(
        lambda t: preprocess_text(
            t,
            remove_stopwords=remove_stopwords,
            apply_stemming=apply_stemming,
            min_token_len=min_token_len,
        )
    )
    n_empty = (df[clean_col] == "").sum()
    logger.info(
        "Done. Cleaned text stored in column '%s'. "
        "Empty texts after preprocessing: %d",
        clean_col, n_empty,
    )
    return df


# ---------------------------------------------------------------------------
# Main pipeline: read JSON → preprocess → save CSV
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(
    input_files: Optional[list[Path]] = None,
    output_file: Optional[Path] = None,
    *,
    remove_stopwords: bool = True,
    apply_stemming:   bool = True,
    drop_empty:       bool = True,
) -> pd.DataFrame:
    """
    Complete pipeline: read all raw JSON files → preprocess → save to CSV.

    Parameters
    ----------
    input_files : list[Path] | None
        List of JSON file paths. If None, find all *_raw.json files in data/raw/.
    output_file : Path | None
        Output CSV path. If None, save to
        data/processed/comments_preprocessed.csv.
    drop_empty : bool
        Remove rows with empty cleaned text (default: True).

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    # --- Resolve input files ---
    if input_files is None:
        input_files = sorted(RAW_DIR.glob("*_raw.json"))
    if not input_files:
        raise FileNotFoundError(f"No JSON files found in: {RAW_DIR}")

    # --- Read and combine all files ---
    frames: list[pd.DataFrame] = []
    required_fields = {"game", "post_id", "comment_id", "comment_text"}
    for fp in input_files:
        logger.info("Reading: %s", fp.name)
        tmp = pd.read_json(fp, encoding="utf-8")
        missing_fields = required_fields - set(tmp.columns)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"{fp.name} is missing required columns: {missing}.")
        for field in ("comment_id", "post_id"):
            empty_ids = tmp[field].isna() | tmp[field].astype(str).str.strip().eq("")
            if empty_ids.any():
                raise ValueError(
                    f"{fp.name} contains {int(empty_ids.sum())} rows with an empty "
                    f"{field}."
                )
        frames.append(tmp)
    df_raw = pd.concat(frames, ignore_index=True)
    logger.info("Total raw rows: %d", len(df_raw))

    # --- Remove duplicates by comment_id ---
    duplicate_ids = df_raw[df_raw.duplicated("comment_id", keep=False)]
    for field in ("game", "post_id", "comment_text"):
        conflicts = duplicate_ids.groupby("comment_id")[field].nunique(dropna=False)
        if (conflicts > 1).any():
            conflicting_ids = conflicts[conflicts > 1].index.tolist()[:5]
            raise ValueError(
                f"Conflicting duplicate comment_id values found in {field}: "
                f"{conflicting_ids}."
            )
    before = len(df_raw)
    df_raw.drop_duplicates(subset=["comment_id"], inplace=True)
    logger.info("Duplicates removed: %d → %d rows", before, len(df_raw))

    # --- Preprocess ---
    df_proc = preprocess_dataframe(
        df_raw,
        text_col="comment_text",
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming,
    )

    # --- Remove rows with empty text ---
    if drop_empty:
        before = len(df_proc)
        df_proc = df_proc[df_proc["clean_text"].str.strip() != ""]
        logger.info("Empty rows removed: %d → %d", before, len(df_proc))

    # --- Select columns to save ---
    keep_cols = [
        "game", "subreddit", "post_id", "post_title",
        "comment_id", "comment_text", "clean_text",
        "comment_score", "comment_created_at", "author",
    ]
    keep_cols = [c for c in keep_cols if c in df_proc.columns]
    df_out = df_proc[keep_cols]

    # --- Save the CSV file ---
    if output_file is None:
        output_file = PROC_DIR / "comments_preprocessed.csv"
    df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info("File saved: %s", output_file)

    return df_out


# ---------------------------------------------------------------------------
# Direct usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import io
    # Force UTF-8 output so emoji can be displayed in the Windows terminal.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # ── Single-text preprocessing examples ────────────────────────────────
    samples = [
        "Check this out! \U0001f3ae\U0001f525 https://youtu.be/dQw4w9WgXcQ lmao this guy is so op wtf",
        "VALORANT is sooooo broken rn, jett ult is busted af \U0001f624\U0001f624 #nerf",
        "gg wp bro, that clutch was cracked \U0001f525\U0001f525 definitely goated ngl",
        "This matchmaking is absolute trash \U0001f480 low elo lobbies r/VALORANT smh",
        "Faceit players >>> premier rankers tbh, no cap the gap is insane \U0001f92f",
        "kd 40/5 LMAOOO hacker obviously, get him vac banned asap",
        "The new cs2 update fixed subtick issues??? ty valve <3",
        "###Reddit Automod: You need more interaction before posting links",
        "I am a bot, please [contact moderators](/message/compose/?to=/r/GlobalOffensive)",
    ]

    print("\n" + "="*60)
    print(" TEXT PREPROCESSING EXAMPLES")
    print("="*60)
    for raw in samples:
        clean = preprocess_text(raw)
        # Encode to avoid UnicodeEncodeError in non-UTF-8 terminals.
        safe_raw   = raw.encode("utf-8", errors="replace").decode("utf-8")
        safe_clean = clean.encode("utf-8", errors="replace").decode("utf-8")
        print(f"\n[BEFORE] {safe_raw}")
        print(f"[AFTER]  {safe_clean}")
    print("="*60)

    # ── Run the complete pipeline ────────────────────────────────────────
    print("\nRunning the complete preprocessing pipeline ...")
    df_result = run_preprocessing_pipeline()
    print(f"\nSample output (first 5 rows):")
    pd.set_option("display.max_colwidth", 60)
    print(df_result[["game", "comment_text", "clean_text"]].head().to_string(index=False))
    print(f"\nTotal cleaned rows: {len(df_result)}")
    print(f"Saved to           : {PROC_DIR / 'comments_preprocessed.csv'}")
