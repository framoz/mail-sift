

```

888b     d888          d8b 888  .d8888b.  d8b  .d888 888    
8888b   d8888          Y8P 888 d88P  Y88b Y8P d88P"  888    
88888b.d88888              888 Y88b.          888    888    
888Y88888P888  8888b.  888 888  "Y888b.   888 888888 888888 
888 Y888P 888     "88b 888 888     "Y88b. 888 888    888    
888  Y8P  888 .d888888 888 888       "888 888 888    888    
888   "   888 888  888 888 888 Y88b  d88P 888 888    Y88b.  
888       888 "Y888888 888 888  "Y8888P"  888 888     "Y888 
---------------------------------------------------------------
```
# MailSift

A machine learning project developed in the context of the **Artificial Intelligence** course at **FEUP**.

## Overview

MailSift builds a classifier that, given an email, predicts multiple labels along four dimensions:

- **Spam**: whether the email is unsolicited bulk mail.
- **Phishing**: whether the email is an attempt to deceive the recipient into disclosing sensitive information.
- **Urgency**: how time-sensitive the email is.
- **Sentiment**: the overall tone of the message (e.g. positive, neutral, negative).


## Repository Structure

```
.
├── mailsift/                   # Python package, importable as `mailsift`
│   ├── config.py               # Paths: REPO_ROOT, DATASETS_DIR, CURATED_DIR
│   ├── data/                   # Per-task data extraction (raw CSV -> curated parquet)
│   │   ├── spam.py             # 4 spam corpora -> datasets/curated/spam.parquet
│   │   └── phishing_email.py   # phishing_email.csv -> datasets/curated/phishing_email.parquet
│   └── models/                 # Per-task algorithms
│       └── spam.py             # Hand-rolled multinomial Naive Bayes
├── datasets/                   # Raw + curated data (not tracked in git)
│   ├── README.md               # Where to download each source dataset
│   ├── dataset-1/              # 7 phishing/spam corpora
│   ├── dataset-2/              # phishing with persuasion categories
│   ├── dataset-3/              # synthetic sentiment
│   ├── dataset-4/              # GoEmotions (sentiment / emotion)
│   └── curated/                # Output of extraction scripts
├── notebooks/                  # Exploration notebooks (not the deliverable)
├── scripts/                    # One-off scripts (e.g. preprocessing inspection)
├── reports/                    # Metrics tables + figures generated during training
│   ├── figures/
│   └── metrics/
├── pyproject.toml
└── README.md
```

## Getting Started

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Download the source datasets per the instructions in `datasets/README.md`, then build the curated parquets:

```bash
python -m mailsift.data.spam
python -m mailsift.data.phishing_email
```

Run the hand-rolled Naive Bayes on both curated parquets:

```bash
python -m mailsift.models.spam
```

Current baseline (multinomial NB from scratch, α = 1.0, lowercase + whitespace tokenizer):

| Dataset                  | Accuracy |
|--------------------------|----------|
| `spam.parquet`           | 0.9772   |
| `phishing_email.parquet` | 0.9817   |

## Approach

The project follows a layered architecture so that data preparation, model training, and inference each evolve at their own pace:

1. **Data prep** (`mailsift/data/<task>.py`): raw CSVs from `datasets/dataset-*/` are loaded, merged, cleaned, deduplicated, label-balanced, and saved to `datasets/curated/<task>.parquet`. Runs once; re-run only when extraction logic changes.
2. **Modelling** (`mailsift/models/<task>.py`): each task will export an `ALGORITHMS` dict mapping algorithm name to an unfitted pipeline factory. Training, comparison, and the frontend selector all iterate this dict.
3. **Inference** (`mailsift/classify.py`, planned): `classify(text, task, algorithm=None)` returns a prediction, falling back to the per-task winner when no algorithm is specified.
4. **Frontend** (`app/streamlit_app.py`, planned): the user picks a task and one or more algorithms inside the app and watches training, comparison, and learning curves happen live.

