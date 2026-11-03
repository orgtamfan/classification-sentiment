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
2. **Labeling:** VADER assigns labels based on the original comment text. The default thresholds are a compound score of `>= 0.05` for positive, `<= -0.05` for negative, and neutral otherwise. The output is saved to `data/processed/sentiment_labels_vader.csv`.
3. **Feature extraction:** comments are split into 80% training data and 20% test data using stratification, then converted into unigram and bigram TF-IDF features (up to 10,000 features). The artifacts are saved in `data/processed/tfidf/`.
4. **Training and evaluation:** GridSearchCV with stratified 5-fold cross-validation selects parameters using macro F1 for Multinomial Naive Bayes, Complement Naive Bayes, and Linear SVM.
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

The preprocessing module downloads the required NLTK resources if they are not already available. `prepare_labels` does not overwrite an existing output file; use `--output` to specify a different file when regenerating labels. Run a module with `--help` to see the available options for feature extraction or labeling.

## Available evaluation results

The following results are stored in `reports/model_comparison_metrics.csv` and are measured against the VADER labels in the test set:

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| Multinomial Naive Bayes | 61.66% | 0.5562 |
| Complement Naive Bayes | 63.49% | 0.5790 |
| Linear SVM | 74.34% | 0.7221 |

Linear SVM has the highest scores among the models in the available evaluation. This does not establish that SVM performs better on all data or communities; the results are limited to this dataset and evaluation procedure.

## Interpretation limitations

- Sentiment labels are generated automatically by VADER, not annotated by humans. The model metrics measure agreement with these automatic labels, not directly with human sentiment judgments. Human annotation of a subset of the data is recommended.
- The current train/test split is performed by comment rather than by post. Comments from the same post may appear in both sets, which can make evaluation results look better than the model's ability to generalize to new posts.
- TF-IDF is fitted before cross-validation on the training data. For more rigorous parameter-selection evaluation, combine the feature vectorizer and classifier in a single scikit-learn `Pipeline` so TF-IDF is refitted in each fold.
- Preprocessing removes English stopwords and emojis; negation words and emojis can carry sentiment. Their impact should be assessed before drawing conclusions.
- The community comparison in `reports/community_comparison.csv` uses Linear SVM predictions on the test set. The class percentages are the model's predicted distribution, not verified sentiment proportions for the entire subreddit.

## Output files

- `models/`: trained Naive Bayes and Linear SVM models.
- `reports/model_comparison_metrics.csv`: model comparison metrics.
- `reports/community_comparison.csv`: evaluation and prediction distribution by community.
- `reports/figures/confusion_matrices.png`: model confusion matrices.
- `reports/figures/sentiment_comparison_cs2_vs_valorant.png`: predicted sentiment distribution by community.
