# Reddit Comment Sentiment Classification: CS2 and VALORANT

This project processes Reddit comments from the Counter-Strike 2 (CS2) and VALORANT communities, assigns positive/neutral/negative sentiment labels, and compares Naive Bayes and Support Vector Machine (SVM) algorithms.

## Data sources

Raw data files are located in `data/raw/`. The subreddits represented in the data are:

| Game | Subreddit | Data file |
|---|---|---|
| CS2 | `r/GlobalOffensive` | `cs2_comments_raw.json` |
| VALORANT | `r/VALORANT` | `valorant_comments_raw.json` |
| Combined | `r/GlobalOffensive` and `r/VALORANT` | `reddit_comments_raw.json` |

The available raw data contains approximately 5,000 comments per subreddit (10,000 comments combined). The combined file may contain comments that also appear in the individual community files. The preprocessing pipeline reads `*_raw.json` files and removes duplicates based on `comment_id`.

The available data represents only these two subreddits, not all CS2 and VALORANT players or communities. The comment period is determined by the `comment_created_at` values in the data files.

## Project structure

```text
data/
  raw/                 Reddit comments in JSON format
  processed/           Preprocessed data, labels, and TF-IDF features
  slang_dict.json      Slang and gaming terminology dictionary
src/
  preprocessing/
    text_preprocessing.py
  prepare_labels.py    Automatic sentiment labeling with VADER
  extract_tfidf.py     Data splitting and TF-IDF feature extraction
  train_and_evaluate.py Model training, evaluation, and reporting
models/                Trained models
reports/               Metrics and visualizations
```

## Processing workflow

1. **Preprocessing:** merge the JSON files in `data/raw/`, remove duplicate comments, clean the text, normalize slang, remove English stopwords, and apply Porter stemming. The output is saved to `data/processed/comments_preprocessed.csv`.
2. **Labeling:** VADER assigns labels based on the original comment text. The default thresholds are a compound score of `>= 0.05` for positive, `<= -0.05` for negative, and neutral otherwise. Each new row starts with `review_status=pending`. Reviewers can mark labels `accepted` or `corrected`; corrected rows must have their `sentiment` replaced with the human-reviewed label. The output is saved to `data/processed/sentiment_labels_vader.csv`.
3. **Feature extraction:** comments are split by `post_id` using `StratifiedGroupKFold`, targeting 80% training data and 20% test data. All sentiment classes are checked in both partitions, and overlapping comment IDs or post IDs are rejected. Identical cleaned text is counted and reported, but retained because repeated short comments may be legitimate. TF-IDF artifacts are saved in `data/processed/tfidf/`.
4. **Training and evaluation:** each TF-IDF vectorizer is fitted inside a scikit-learn `Pipeline` during stratified, group-based 5-fold cross-validation. GridSearchCV selects parameters using macro F1 for Multinomial Naive Bayes, Complement Naive Bayes, and Linear SVM. A most-frequent `DummyClassifier` provides a baseline.
5. **Reporting:** metrics, community comparisons, confusion matrices, and distribution charts are saved in `reports/`.

## Running the workflow

Use Python with the following packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `nltk`, and `vaderSentiment`.

Run these commands in PowerShell from the project root:

```powershell
python -m pip install pandas numpy scipy scikit-learn joblib matplotlib seaborn nltk vaderSentiment
python -m src.preprocessing.text_preprocessing
python -m src.prepare_labels
python -m src.extract_tfidf
python -m src.train_and_evaluate
```

The preprocessing module downloads the required NLTK resources if they are not already available. Negation terms are retained during preprocessing. `prepare_labels` does not overwrite an existing output file; use `--output` to specify a different file, or pass `--overwrite` only after preserving any manual edits. Run a module with `--help` to see the available options for feature extraction or labeling.

The generated label CSV includes `review_status`, initially set to `pending`. After human review, set it to `accepted` or `corrected`; for corrected labels, update `sentiment` to the reviewed class before rebuilding the feature artifacts and models.

## Available evaluation results

The following results are stored in `reports/model_comparison_metrics.csv` and are measured against the VADER labels in the test set:

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| Dummy baseline | 47.21% | 0.2138 |
| Multinomial Naive Bayes | 57.40% | 0.4917 |
| Complement Naive Bayes | 60.09% | 0.5524 |
| Linear SVM | 70.84% | 0.6856 |

These results use a post-grouped holdout split and TF-IDF pipelines evaluated with group-based cross-validation. They measure agreement with VADER labels, not human-annotated sentiment.

## Interpretation limitations

- Sentiment labels are generated automatically by VADER, not annotated by humans. The model metrics measure agreement with these automatic labels, not directly with human sentiment judgments. Human annotation of a subset is needed for human-validated performance.
- The grouped split prevents comments from the same post appearing in both train and test data. However, identical cleaned text can still occur across different posts; these overlaps are reported during feature extraction and should be considered when interpreting results.
- Review-status values are validated when present (`pending`, `accepted`, or `corrected`). Existing datasets without this column are treated as pending review; this does not constitute human annotation.
- Negation terms are retained during preprocessing. Emojis are removed and may carry sentiment information, so this remains a limitation.
- Pairwise Cohen's kappa and human-annotation evaluation are not reported because the workspace does not currently contain independent annotator labels. Those analyses require a reviewed dataset with a documented annotator/label schema.
- Linear SVM does not provide calibrated probabilities by default. Inference reports its decision scores instead; Naive Bayes and the Dummy baseline report class probabilities.
- The community comparison in `reports/community_comparison.csv` uses Linear SVM predictions on the test set. The class percentages are the model's predicted distribution, not verified sentiment proportions for the entire subreddit.

## Output files

- `models/`: trained Dummy baseline, Naive Bayes, and Linear SVM models.
- `reports/model_comparison_metrics.csv`: model comparison metrics.
- `reports/community_comparison.csv`: evaluation and prediction distribution by community.
- `reports/figures/confusion_matrices.png`: model confusion matrices.
- `reports/figures/sentiment_comparison_cs2_vs_valorant.png`: predicted sentiment distribution by community.
