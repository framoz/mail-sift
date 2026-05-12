from pathlib import Path

import pandas as pd

from mailsift.config import CURATED_DIR, DATASETS_DIR


SOURCE_PATH: Path = DATASETS_DIR / "dataset-1" / "phishing_email.csv"
OUTPUT_PATH: Path = CURATED_DIR / "phishing_email.parquet"
RANDOM_STATE: int = 42


def inspect(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print("shape :", df.shape)
    print("dtypes:\n", df.dtypes)
    print("nulls :\n", df.isna().sum())
    print("head  :\n", df.head(3))


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["text_combined", "label"],
        dtype={"label": "int8"},
    )
    return df.rename(columns={"text_combined": "text"})


def handle_nulls(corpus: pd.DataFrame) -> pd.DataFrame:
    corpus = corpus.dropna(subset=["text", "label"])
    corpus = corpus[corpus["text"].str.strip() != ""]
    return corpus.reset_index(drop=True)


def deduplicate(corpus: pd.DataFrame) -> pd.DataFrame:
    return corpus.drop_duplicates(subset="text", keep="first").reset_index(drop=True)


def balance_labels(corpus: pd.DataFrame) -> pd.DataFrame:
    ham = corpus[corpus["label"] == 0]
    spam = corpus[corpus["label"] == 1]
    n = min(len(ham), len(spam))
    ham_balanced = ham.sample(n=n, random_state=RANDOM_STATE)
    spam_balanced = spam.sample(n=n, random_state=RANDOM_STATE)
    balanced = pd.concat([ham_balanced, spam_balanced], ignore_index=True)
    return balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def save_curated(corpus: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(path, index=False)
    label_counts = corpus["label"].value_counts().to_dict()
    print(f"\nsaved {len(corpus)} rows to {path} (label counts: {label_counts})")


def main() -> None:
    corpus = load_source(SOURCE_PATH)
    inspect(corpus, "phishing_email source")
    corpus = handle_nulls(corpus)
    corpus = deduplicate(corpus)
    corpus = balance_labels(corpus)
    save_curated(corpus, OUTPUT_PATH)


if __name__ == "__main__":
    main()
