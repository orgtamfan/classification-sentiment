"""
Modul prapemrosesan teks untuk penelitian:
"Klasifikasi Sentimen Komentar Reddit pada Komunitas Counter Strike 2 dan VALORANT
Menggunakan Algoritma Naive Bayes dan SVM"

Tahapan prapemrosesan:
    1. Dekode entitas HTML
    2. Hapus URL/tautan
    3. Hapus mention Reddit (u/user, r/subreddit)
    4. Hapus sintaks Markdown Reddit (tebal, miring, judul, tabel, kutipan, dll.)
    5. Hapus emoji & emoticon Unicode
    6. Hapus emotikon teks (emosi ASCII)
    7. Hapus karakter non-ASCII / karakter rusak
    8. Ubah huruf menjadi kecil
    9. Normalisasi slang/istilah permainan
   10. Hapus angka yang berdiri sendiri
   11. Hapus tanda baca
   12. Hapus stopwords bahasa Inggris
   13. Tokenisasi
   14. Hapus token pendek (< 2 karakter)
   15. Stemming (Porter Stemmer)
   16. Hapus token kosong atau hanya berisi spasi
   17. Gabungkan kembali menjadi teks
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
# Atur pencatatan log
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unduh resource NLTK jika belum ada
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
        logger.info("Mengunduh NLTK resource: %s", _pkg)
        nltk.download(_pkg, quiet=True)

# ---------------------------------------------------------------------------
# Konstanta path
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[2]      # root proyek
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
SLANG_PATH = DATA_DIR / "slang_dict.json"

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Muat kamus slang/istilah permainan
# ---------------------------------------------------------------------------
def _load_slang_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        logger.warning("File kamus slang tidak ditemukan: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw: dict = json.load(f)
    # pastikan semua key & value lowercase dan tidak ada spasi berlebih
    return {k.strip().lower(): v.strip().lower() for k, v in raw.items()}

SLANG_DICT: dict[str, str] = _load_slang_dict(SLANG_PATH)
logger.info("Kamus slang dimuat: %d entri", len(SLANG_DICT))

# ---------------------------------------------------------------------------
# Stopwords bahasa Inggris
# ---------------------------------------------------------------------------
STOP_WORDS: frozenset[str] = frozenset(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Porter Stemmer
# ---------------------------------------------------------------------------
STEMMER = PorterStemmer()

# ---------------------------------------------------------------------------
# Pola regex yang dikompilasi sekali (lebih efisien)
# ---------------------------------------------------------------------------
# URL / hyperlink
_RE_URL = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+",
    re.IGNORECASE,
)

# Mention Reddit: u/username atau r/subreddit (inkl. /r/ dan /u/)
_RE_REDDIT_MENTION = re.compile(
    r"/?[ur]/\w+",
    re.IGNORECASE,
)

# Markdown Reddit
_RE_MARKDOWN_HEADER   = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_MARKDOWN_BOLD_IT  = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_RE_MARKDOWN_STRIKE   = re.compile(r"~~(.+?)~~")
_RE_MARKDOWN_CODE     = re.compile(r"`{1,3}.+?`{1,3}", re.DOTALL)
_RE_MARKDOWN_LINK     = re.compile(r"\[([^\]]*)\]\([^\)]*\)")   # [teks](url)
_RE_MARKDOWN_QUOTE    = re.compile(r"^>\s*", re.MULTILINE)
_RE_MARKDOWN_TABLE    = re.compile(r"^\|.*\|$", re.MULTILINE)
_RE_MARKDOWN_HR       = re.compile(r"^[-_*]{3,}\s*$", re.MULTILINE)

# ASCII emotikon (harus sebelum hapus tanda baca)
# Contoh: :) :( ;) :D :P :-) xD XD etc.
_RE_ASCII_EMOTICON = re.compile(
    r"(?::|;|=)(?:-)?(?:\)|D|P|\(|\/|\\|\||\}|\{|@|3|>|<|O|0)"  # mata : hidung - mulut
    r"|(?:\^_?\^|>.<|T_T|;_;|>_<|O_O|0_0|=_=|@_@|\*_\*|x_x|X_X|UwU|OwO)"
    r"|(?:xD|XD|xd|D:|:D|:\)|:\(|:\||:P|:p|:o|:O|:\*|:3|<3|</3|>_>|<_<)"
    r"|(?:8\)|B\)|\^o\^|\(ﾉ◕ヮ◕\)ﾉ|\(╯°□°\)╯)",
    re.IGNORECASE,
)

# Karakter pengganda berlebih (lebih dari 2 karakter berulang → 2)
_RE_REPEAT_CHAR = re.compile(r"(.)\1{2,}")

# Angka standalone (bukan bagian kata)
_RE_STANDALONE_NUMBER = re.compile(r"\b\d+\b")

# Karakter yang bukan huruf atau spasi (setelah proses sebelumnya)
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")

# Spasi berlebih
_RE_MULTI_SPACE = re.compile(r"\s{2,}")

# Karakter rusak / replacement character
_RE_CORRUPT = re.compile(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]")


# ---------------------------------------------------------------------------
# Fungsi bantu: hapus emoji Unicode
# ---------------------------------------------------------------------------
def _remove_unicode_emoji(text: str) -> str:
    """Hapus semua emoji dan simbol Unicode menggunakan kategori karakter."""
    result = []
    for char in text:
        cat = unicodedata.category(char)
        cp  = ord(char)
                # Kategori "So" berarti simbol lain dan mencakup emoji.
                # Hapus juga surrogate, area penggunaan privat, dan blok emoji umum.
        if cat in {"So", "Cs", "Co"} or (
            0x1F300 <= cp <= 0x1FAFF or   # Misc Symbols & Pictographs, Emoticons, dll.
            0x2600  <= cp <= 0x27BF or    # Misc Symbols, Dingbats
            0xFE00  <= cp <= 0xFE0F or    # Variation Selectors
            0x200D  == cp              or  # Zero Width Joiner
            0xFE0F  == cp              or  # Variation Selector-16
            0x20E3  == cp              or  # Combining Enclosing Keycap
            0x1F1E0 <= cp <= 0x1F1FF       # Regional Indicator Symbols (bendera)
        ):
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


# ---------------------------------------------------------------------------
# Fungsi bantu: normalisasi slang dengan satu pola regex alternasi.
# Cara ini lebih cepat daripada memeriksa setiap kata dengan pola terpisah.
# ---------------------------------------------------------------------------
def _build_slang_pattern(slang_dict: dict[str, str]) -> re.Pattern | None:
    """
    Gabungkan semua kunci slang menjadi satu pola alternasi regex.
    Dahulukan kunci terpanjang agar frasa yang terdiri dari beberapa kata diprioritaskan.
    """
    if not slang_dict:
        return None
    sorted_keys = sorted(slang_dict, key=len, reverse=True)
    pattern_str  = r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b"
    return re.compile(pattern_str, re.IGNORECASE)

# Kompilasi satu kali saat modul diimpor
_SLANG_PATTERN: re.Pattern | None = _build_slang_pattern(SLANG_DICT)

def _normalize_slang(text: str) -> str:
    """
    Ganti kata/frasa slang dengan bentuk bakunya dalam satu pemindaian regex.
    Frasa yang lebih panjang diprioritaskan karena kunci diurutkan dari terpanjang.
    """
    if _SLANG_PATTERN is None:
        return text
    return _SLANG_PATTERN.sub(lambda m: SLANG_DICT[m.group(0).lower()], text)


# ---------------------------------------------------------------------------
# Fungsi utama prapemrosesan satu teks
# ---------------------------------------------------------------------------
def preprocess_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    apply_stemming:   bool = True,
    min_token_len:    int  = 2,
) -> str:
    """
    Bersihkan satu teks komentar Reddit melalui seluruh tahapan prapemrosesan.

    Parameters
    ----------
    text : str
        Teks mentah komentar Reddit.
    remove_stopwords : bool
        Hapus kata umum bahasa Inggris (bawaan True).
    apply_stemming : bool
        Terapkan Porter Stemmer (bawaan True).
    min_token_len : int
        Panjang minimum token yang disimpan (bawaan 2).

    Returns
    -------
    str
        Teks bersih yang siap diekstrak fiturnya.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Dekode entitas HTML (&amp; → &, &lt; → <, &#39; → ', dll.)
    text = html.unescape(text)

    # 2. Hapus URL / hyperlink
    text = _RE_URL.sub(" ", text)

    # 3. Hapus mention Reddit (u/..., r/...)
    text = _RE_REDDIT_MENTION.sub(" ", text)

    # 4. Hapus sintaks Markdown Reddit
    text = _RE_MARKDOWN_HEADER.sub("", text)
    text = _RE_MARKDOWN_CODE.sub(" ", text)
    text = _RE_MARKDOWN_LINK.sub(r"\1", text)   # pertahankan teks link
    text = _RE_MARKDOWN_BOLD_IT.sub(r"\1", text)
    text = _RE_MARKDOWN_STRIKE.sub(r"\1", text)
    text = _RE_MARKDOWN_QUOTE.sub("", text)
    text = _RE_MARKDOWN_TABLE.sub(" ", text)
    text = _RE_MARKDOWN_HR.sub(" ", text)

    # 5. Hapus emoji Unicode
    text = _remove_unicode_emoji(text)

    # 6. Hapus ASCII emotikon
    text = _RE_ASCII_EMOTICON.sub(" ", text)

    # 7. Hapus karakter non-ASCII / karakter rusak (mis. replacement char U+FFFD)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 8. Ubah huruf menjadi kecil
    text = text.lower()

    # 9. Normalisasi karakter berulang (sooooo → soo, hahaha → haha)
    text = _RE_REPEAT_CHAR.sub(r"\1\1", text)

    # 10. Normalisasi slang dan istilah permainan
    text = _normalize_slang(text)

    # 11. Hapus angka standalone
    text = _RE_STANDALONE_NUMBER.sub(" ", text)

    # 12. Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 13. Hapus karakter selain huruf & spasi
    text = _RE_NON_ALPHA.sub(" ", text)

    # 14. Normalisasi spasi
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    # 15. Tokenisasi
    tokens: list[str] = word_tokenize(text)

    # 16. Hapus token pendek
    tokens = [t for t in tokens if len(t) >= min_token_len]

    # 17. Hapus stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # 18. Stemming
    if apply_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]

    # 19. Hapus token pendek (setelah stemming bisa lebih pendek)
    tokens = [t for t in tokens if len(t) >= min_token_len]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Fungsi sekaligus: prapemrosesan seluruh DataFrame
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
    Terapkan prapemrosesan pada seluruh isi kolom teks dalam DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame masukan yang memiliki kolom teks mentah.
    text_col : str
        Nama kolom teks mentah (bawaan "comment_text").
    clean_col : str
        Nama kolom keluaran teks bersih (bawaan "clean_text").

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah dilengkapi kolom teks bersih.
    """
    logger.info("Memproses %d baris teks ...", len(df))
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
        "Selesai. Teks bersih tersimpan di kolom '%s'. "
        "Teks kosong setelah preprocessing: %d",
        clean_col, n_empty,
    )
    return df


# ---------------------------------------------------------------------------
# Alur utama: baca JSON → prapemrosesan → simpan CSV
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
    Alur lengkap: baca semua file JSON mentah → prapemrosesan → simpan ke CSV.

    Parameters
    ----------
    input_files : list[Path] | None
        Daftar lokasi berkas JSON. Jika None, cari semua *_raw.json di data/raw/.
    output_file : Path | None
        Lokasi berkas CSV keluaran. Jika None, simpan ke
        data/processed/comments_preprocessed.csv.
    drop_empty : bool
        Hapus baris yang teks bersihnya kosong (bawaan True).

    Returns
    -------
    pd.DataFrame
        DataFrame hasil prapemrosesan.
    """
    # --- Tentukan berkas masukan ---
    if input_files is None:
        input_files = sorted(RAW_DIR.glob("*_raw.json"))
    if not input_files:
        raise FileNotFoundError(f"Tidak ada file JSON di: {RAW_DIR}")

    # --- Baca dan gabungkan semua berkas ---
    frames: list[pd.DataFrame] = []
    for fp in input_files:
        logger.info("Membaca: %s", fp.name)
        tmp = pd.read_json(fp, encoding="utf-8")
        frames.append(tmp)
    df_raw = pd.concat(frames, ignore_index=True)
    logger.info("Total data mentah: %d baris", len(df_raw))

    # --- Hapus duplikat berdasarkan comment_id ---
    if "comment_id" in df_raw.columns:
        before = len(df_raw)
        df_raw.drop_duplicates(subset=["comment_id"], inplace=True)
        logger.info("Duplikat dihapus: %d → %d baris", before, len(df_raw))

    # --- Prapemrosesan ---
    df_proc = preprocess_dataframe(
        df_raw,
        text_col="comment_text",
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming,
    )

    # --- Hapus baris dengan teks kosong ---
    if drop_empty:
        before = len(df_proc)
        df_proc = df_proc[df_proc["clean_text"].str.strip() != ""]
        logger.info("Baris kosong dihapus: %d → %d", before, len(df_proc))

    # --- Pilih kolom untuk disimpan ---
    keep_cols = [
        "game", "subreddit", "post_id", "post_title",
        "comment_id", "comment_text", "clean_text",
        "comment_score", "comment_created_at", "author",
    ]
    keep_cols = [c for c in keep_cols if c in df_proc.columns]
    df_out = df_proc[keep_cols]

    # --- Simpan berkas CSV ---
    if output_file is None:
        output_file = PROC_DIR / "comments_preprocessed.csv"
    df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info("File disimpan: %s", output_file)

    return df_out


# ---------------------------------------------------------------------------
# Contoh penggunaan langsung
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import io
    # Paksa stdout pakai UTF-8 agar emoji bisa ditampilkan di terminal Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # ── Contoh prapemrosesan satu teks ────────────────────────────────────
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
    print(" CONTOH PRAPEMROSESAN TEKS")
    print("="*60)
    for raw in samples:
        clean = preprocess_text(raw)
        # encode untuk menghindari UnicodeEncodeError di terminal non-UTF8
        safe_raw   = raw.encode("utf-8", errors="replace").decode("utf-8")
        safe_clean = clean.encode("utf-8", errors="replace").decode("utf-8")
        print(f"\n[SEBELUM] {safe_raw}")
        print(f"[SESUDAH] {safe_clean}")
    print("="*60)

    # ── Jalankan seluruh alur ─────────────────────────────────────────────
    print("\nMenjalankan seluruh alur prapemrosesan ...")
    df_result = run_preprocessing_pipeline()
    print(f"\nContoh keluaran (5 baris pertama):")
    pd.set_option("display.max_colwidth", 60)
    print(df_result[["game", "comment_text", "clean_text"]].head().to_string(index=False))
    print(f"\nTotal data bersih: {len(df_result)} baris")
    print(f"Tersimpan di    : {PROC_DIR / 'comments_preprocessed.csv'}")
