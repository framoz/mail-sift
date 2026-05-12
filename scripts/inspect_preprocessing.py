"""
Detect what (if any) preprocessing has already been applied to each dataset's
text column. Reads a small sample per file and reports fingerprints:
lowercasing, punctuation stripping, HTML, URL/email masking, stopword
removal, stemming, lemmatization.
"""

from collections import Counter

import pandas as pd

from mailsift.config import DATASETS_DIR


# (label shown in report, file path relative to DATASETS_DIR, text column, extra read_csv kwargs)
DATASETS = [
    ("dataset-1 / CEAS_08",        "dataset-1/CEAS_08.csv",                        "body",          {}),
    ("dataset-1 / Enron",          "dataset-1/Enron.csv",                          "body",          {}),
    ("dataset-1 / Ling",           "dataset-1/Ling.csv",                           "body",          {}),
    ("dataset-1 / Nazario",        "dataset-1/Nazario.csv",                        "body",          {}),
    ("dataset-1 / Nigerian_Fraud", "dataset-1/Nigerian_Fraud.csv",                 "body",          {}),
    ("dataset-1 / SpamAssasin",    "dataset-1/SpamAssasin.csv",                    "body",          {}),
    ("dataset-1 / phishing_email", "dataset-1/phishing_email.csv",                 "text_combined", {}),
    ("dataset-2 / phishing_cat",   "dataset-2/phishing_dataset_with_category.csv", "text",          {}),
    ("dataset-3 / sentiments",     "dataset-3/data.csv",                           "messages",      {}),
    ("dataset-4 / train.tsv",      "dataset-4/train.tsv",                          "text",          {"sep": "\t", "header": None, "names": ["text", "emotion_ids", "comment_id"]}),
    ("dataset-4 / dev.tsv",        "dataset-4/dev.tsv",                            "text",          {"sep": "\t", "header": None, "names": ["text", "emotion_ids", "comment_id"]}),
    ("dataset-4 / test.tsv",       "dataset-4/test.tsv",                           "text",          {"sep": "\t", "header": None, "names": ["text", "emotion_ids", "comment_id"]}),
    ("dataset-4 / goemotions_1",   "dataset-4/goemotions_1.csv",                   "text",          {}),
    ("dataset-4 / goemotions_2",   "dataset-4/goemotions_2.csv",                   "text",          {}),
    ("dataset-4 / goemotions_3",   "dataset-4/goemotions_3.csv",                   "text",          {}),
]

SAMPLE_SIZE = 500

STOPWORDS = {"the", "of", "and", "is", "a", "to", "in"}

STEM_PROBES = [
    "studi", "comput", "organ", "babi", "troubl", "arriv",
    "messag", "peopl", "compani", "activ", "servic", "busi",
    "famili", "citi", "polici", "qualiti", "secur",
]


def load_sample(rel_path, text_col, n, read_csv_kwargs):
    path = DATASETS_DIR / rel_path
    kwargs = {"nrows": n, "usecols": [text_col], **read_csv_kwargs}
    df = pd.read_csv(path, **kwargs)
    return df[text_col].dropna().astype(str)


def check_lowercased(texts):
    def first_alpha_is_upper(s):
        for c in s:
            if c.isalpha():
                return c.isupper()
        return False
    return texts.map(first_alpha_is_upper).mean()


def check_punctuation_stripped(texts):
    return texts.str.contains(r"[.,!?;:]", regex=True).mean()


def check_html_present(texts):
    return texts.str.contains(r"<[a-zA-Z/]|&nbsp;|href=", regex=True).mean()


def check_urls_or_emails_masked(texts):
    pattern = r"\b(?:URL|EMAIL|HTTPURL)\b|<URL>|<EMAIL>|\[URL\]|\[EMAIL\]"
    return texts.str.contains(pattern, regex=True).mean()


def _token_counter(texts):
    counter = Counter()
    for s in texts:
        counter.update(s.lower().split())
    return counter


def check_stopwords_removed(texts):
    counter = _token_counter(texts)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum(counter[w] for w in STOPWORDS) / total


def check_stemmed(texts):
    pattern = r"\b(?:" + "|".join(STEM_PROBES) + r")\b"
    return texts.str.contains(pattern, regex=True, case=False).mean()


def check_lemmatized(texts):
    counter = _token_counter(texts)

    def inflection_ratio(inflected_forms, lemma):
        infl = sum(counter[w] for w in inflected_forms)
        lem = counter[lemma]
        if infl + lem == 0:
            return None
        return infl / (infl + lem)

    return {
        "be": inflection_ratio(["is", "was", "are", "were", "been", "being"], "be"),
        "have": inflection_ratio(["has", "had", "having"], "have"),
    }


def run_all_checks(texts):
    return {
        "first_char_uppercase": check_lowercased(texts),
        "has_punctuation":      check_punctuation_stripped(texts),
        "has_html":             check_html_present(texts),
        "has_url_email_mask":   check_urls_or_emails_masked(texts),
        "stopword_share":       check_stopwords_removed(texts),
        "stem_fingerprint":     check_stemmed(texts),
        "inflection_ratio":     check_lemmatized(texts),
    }


def _fmt(val):
    if val is None:
        return "n/a"
    if isinstance(val, dict):
        return " ".join(f"{k}={_fmt(v)}" for k, v in val.items())
    return f"{val:.3f}"


def format_report(label, results):
    print(f"\n--- {label} ---")
    for name, val in results.items():
        print(f"  {name:22s} {_fmt(val)}")


def main():
    for label, rel_path, text_col, kwargs in DATASETS:
        try:
            texts = load_sample(rel_path, text_col, SAMPLE_SIZE, kwargs)
        except FileNotFoundError:
            print(f"\n--- {label} ---  (file missing, skipped)")
            continue
        if texts.empty:
            print(f"\n--- {label} ---  (no rows, skipped)")
            continue
        results = run_all_checks(texts)
        format_report(label, results)


if __name__ == "__main__":
    main()
