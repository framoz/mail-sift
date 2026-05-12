# Datasets

The CSV files below are not tracked in git. Download them from Kaggle and place
them in the matching folder so the tree looks exactly like the layout shown.

```
data/
├── dataset-1/
│   ├── CEAS_08.csv
│   ├── Enron.csv
│   ├── Ling.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   ├── SpamAssasin.csv
│   └── phishing_email.csv
└── dataset-2/
    └── phishing_dataset_with_category.csv
```

## dataset-1: Phishing Email Dataset

https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data

A collection of seven public email corpora curated for phishing/spam
classification. Every file uses a binary `label` column with the same encoding
( `1` = malicious, `0` = legitimate —) but the *meaning* of "malicious" depends
on the source corpus:

| File                   | Columns                                            | Rows      | What `label = 1` means                           |
| ---------------------- | -------------------------------------------------- | --------- | -------------------------------------------------- |
| `CEAS_08.csv`        | sender, receiver, date, subject, body, label, urls | 1,305,706 | spam (CEAS 2008 challenge corpus)                  |
| `Enron.csv`          | subject, body, label                               | 720,543   | spam (Enron-Spam corpus)                           |
| `Ling.csv`           | subject, body, label                               | 5,475     | spam (Ling-Spam corpus)                            |
| `Nazario.csv`        | sender, receiver, date, subject, body, urls, label | 2,793     | phishing (Nazario phishing corpus)                 |
| `Nigerian_Fraud.csv` | sender, receiver, date, subject, body, urls, label | 156,000   | phishing / 419-style fraud                         |
| `SpamAssasin.csv`    | sender, receiver, date, subject, body, label, urls | 201,450   | spam (Apache SpamAssassin corpus)                  |
| `phishing_email.csv` | text_combined, label                               | 82,486    | phishing or spam (merged across all sources above) |

`phishing_email.csv` is the author's consolidated corpus: subject and body are
concatenated into a single `text_combined` field, and the per-source
spam/phishing distinction is collapsed into one "malicious" class.

## dataset-2: Phishing Email & SMS Dataset with NLP Categories

https://www.kaggle.com/datasets/ahmadtijjani/phishing-urgency-authority-persuasion

Short phishing messages annotated with the persuasion tactic used. Unlike
dataset-1, `label` here is **not** a binary phishing-vs-legitimate flag — every
row is phishing, so `label` is a constant `"phishing"` string. The actual
classification target is `category`, which tags *how* the message tries to
manipulate the reader.

| File                                   | Columns               | Rows  |
| -------------------------------------- | --------------------- | ----- |
| `phishing_dataset_with_category.csv` | text, category, label | 1,000 |

- `text` — the message body.
- `category` — multi-class target, one of:
  - `urgency` — pressures the reader to act immediately ("your account will be closed in 24h").
  - `authority` — impersonates a trusted entity (bank, IRS, IT admin).
  - `persuasion` — softer social-engineering appeals (rewards, favors, sympathy).
- `label` — always `"phishing"`; useful only as a join key when mixing with legitimate samples from another corpus.
