# Klasifikasi Sentimen Komentar Reddit: CS2 dan VALORANT

Proyek ini mengolah komentar Reddit dari komunitas Counter-Strike 2 (CS2) dan VALORANT, memberi label sentimen positif/netral/negatif, lalu membandingkan algoritma Naive Bayes dan Support Vector Machine (SVM).

## Sumber data

Berkas data mentah berada di `data/raw/`. Subreddit yang tercatat dalam data:

| Permainan | Subreddit | Berkas data |
|---|---|---|
| CS2 | `r/GlobalOffensive` | `cs2_comments_raw.json` |
| VALORANT | `r/VALORANT` | `valorant_comments_raw.json` |
| Gabungan | `r/GlobalOffensive` dan `r/VALORANT` | `reddit_comments_raw.json` |

Data mentah yang tersedia berisi sekitar 5.000 komentar per subreddit (10.000 komentar gabungan). Berkas gabungan dapat memuat komentar yang sama dengan berkas per komunitas. Alur prapemrosesan membaca berkas `*_raw.json` dan menghapus duplikat berdasarkan `comment_id`.

Data yang tersedia hanya mewakili dua subreddit tersebut, bukan seluruh pemain atau komunitas CS2 dan VALORANT. Periode komentar mengikuti nilai `comment_created_at` pada berkas data.

## Struktur proyek

```text
data/
  raw/                 Data komentar Reddit dalam format JSON
  processed/           Hasil prapemrosesan, label, dan fitur TF-IDF
  slang_dict.json      Kamus normalisasi slang/game
src/
  preprocessing/
    text_preprocessing.py
  prepare_labels.py    Pembuatan label sentimen otomatis dengan VADER
  extract_tfidf.py     Pembagian data dan ekstraksi fitur TF-IDF
  train_and_evaluate.py
models/                Model hasil pelatihan
reports/               Metrik dan visualisasi
```

## Alur pemrosesan

1. **Prapemrosesan:** gabungkan berkas JSON di `data/raw/`, hapus komentar duplikat, bersihkan teks, normalkan slang, hapus kata umum bahasa Inggris, dan lakukan stemming Porter. Hasilnya disimpan ke `data/processed/comments_preprocessed.csv`.
2. **Pelabelan:** VADER memberi label berdasarkan teks komentar asli. Ambang bawaan adalah skor compound `>= 0.05` untuk positif, `<= -0.05` untuk negatif, dan selebihnya netral. Hasilnya disimpan ke `data/processed/sentiment_labels_vader.csv`.
3. **Ekstraksi fitur:** komentar dibagi secara terstratifikasi menjadi 80% data latih dan 20% data uji, lalu diubah menjadi fitur TF-IDF unigram dan bigram (maksimum 10.000 fitur). Hasilnya disimpan di `data/processed/tfidf/`.
4. **Pelatihan dan evaluasi:** GridSearchCV dengan validasi silang terstratifikasi 5-fold memilih parameter menggunakan F1 makro untuk Multinomial Naive Bayes, Complement Naive Bayes, dan Linear SVM.
5. **Laporan:** metrik, perbandingan hasil per komunitas, matriks kebingungan, dan grafik distribusi disimpan di `reports/`.

## Menjalankan alur kerja

Gunakan Python dengan pustaka `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `nltk`, dan `vaderSentiment`.

Contoh perintah di PowerShell dari direktori root proyek:

```powershell
python -m pip install pandas numpy scipy scikit-learn joblib matplotlib seaborn nltk vaderSentiment
python -m src.preprocessing.text_preprocessing
python -m src.prepare_labels
python -m src.extract_tfidf
python -m src.train_and_evaluate
```

Modul prapemrosesan mengunduh sumber daya NLTK yang diperlukan jika belum tersedia. `prepare_labels` tidak menimpa berkas keluaran yang sudah ada; gunakan `--output` untuk menentukan nama berkas lain saat membuat ulang label. Untuk melihat pilihan pada tahap ekstraksi fitur atau pelabelan, jalankan modul dengan `--help`.

## Hasil evaluasi yang tersedia

Hasil berikut tersimpan di `reports/model_comparison_metrics.csv` dan dihitung terhadap label VADER pada data uji:

| Model | Model | Akurasi data uji | F1 makro data uji |
|---|---:|---:|
| Multinomial Naive Bayes | 61,66% | 0,5562 |
| Complement Naive Bayes | 63,49% | 0,5790 |
| Linear SVM | 74,34% | 0,7221 |

Linear SVM mendapat skor tertinggi di antara model yang dibandingkan pada evaluasi yang tersedia. Hasil ini bukan bukti bahwa SVM lebih baik pada semua data atau komunitas; interpretasinya terbatas pada kumpulan data dan prosedur evaluasi ini.

## Batasan interpretasi

- Label sentimen dibuat otomatis oleh VADER, bukan melalui anotasi manusia. Metrik model mengukur kesesuaian prediksi model terhadap label otomatis tersebut, bukan secara langsung ketepatan terhadap persepsi manusia. Sebaiknya sebagian data divalidasi oleh anotator.
- Pembagian data latih/uji saat ini dilakukan per komentar, bukan per kiriman. Komentar dari kiriman yang sama dapat muncul di kedua bagian, sehingga hasil evaluasi bisa tampak lebih baik daripada kemampuan model menggeneralisasi ke kiriman baru.
- TF-IDF dilatih sebelum validasi silang pada data latih. Untuk evaluasi pemilihan parameter yang lebih ketat, gabungkan vectorizer dan classifier dalam satu `Pipeline` scikit-learn agar TF-IDF dilatih ulang pada setiap lipatan.
- Prapemrosesan menggunakan daftar kata umum bahasa Inggris dan menghapus emoji; kata negasi atau emoji dapat membawa makna sentimen. Dampaknya perlu diperiksa sebelum menarik kesimpulan.
- Perbandingan komunitas di `reports/community_comparison.csv` menggunakan prediksi Linear SVM pada data uji. Persentase kelas di sana adalah sebaran prediksi model, bukan proporsi sentimen terverifikasi untuk seluruh subreddit.

## Berkas hasil

- `models/`: model hasil pelatihan Naive Bayes dan Linear SVM.
- `reports/model_comparison_metrics.csv`: metrik perbandingan model.
- `reports/community_comparison.csv`: evaluasi dan distribusi prediksi per komunitas.
- `reports/figures/confusion_matrices.png`: matriks kebingungan model.
- `reports/figures/sentiment_comparison_cs2_vs_valorant.png`: grafik sebaran prediksi sentimen tiap komunitas.
