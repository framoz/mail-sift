from pathlib import Path

import pandas as pd

from mailsift.config import CURATED_DIR, DATASETS_DIR


SOURCES: list[str] = ["CEAS_08", "Enron", "Ling", "SpamAssasin"]
ROWS_PER_SOURCE: int = 2859  # roughly matches the smallest source after dedup
OUTPUT_PATH: Path = CURATED_DIR / "spam.parquet"
RANDOM_STATE: int = 42


def inspect(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print("shape :", df.shape)
    print("dtypes:\n", df.dtypes)
    print("nulls :\n", df.isna().sum())
    print("head  :\n", df.head(3))


def load_sources(sources: list[str], rows_per_source: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in sources:
        df = pd.read_csv(
            f"{DATASETS_DIR}/dataset-1/{name}.csv",
            usecols=["subject", "body", "label"],
            dtype={"label": "int8"},
            nrows=rows_per_source,
        )
        df["source"] = name
        frames.append(df)
    corpus = pd.concat(frames, ignore_index=True)
    corpus["body"] = corpus["body"].apply(lambda x: x.replace("\n", " ").replace("\r", " "))
    return corpus


def combine_subject_body(corpus: pd.DataFrame) -> pd.DataFrame:
    text = corpus[["subject", "body"]].fillna("").agg(" ".join, axis=1).str.strip()
    return corpus.assign(text=text).drop(columns=["subject", "body"])


def handle_nulls(corpus: pd.DataFrame) -> pd.DataFrame:
    corpus = corpus.dropna(subset=["label"])
    corpus = corpus[corpus["text"] != ""]
    return corpus.reset_index(drop=True)


def deduplicate(corpus: pd.DataFrame) -> pd.DataFrame:
    return corpus.drop_duplicates(subset="text", keep="first").reset_index(drop=True)


def balance_labels(corpus: pd.DataFrame) -> pd.DataFrame:
    corpus_ham = corpus[corpus["label"] == 0]
    corpus_spam = corpus[corpus["label"] == 1]
    min_size = min(len(corpus_ham), len(corpus_spam))
    corpus_ham_balanced = corpus_ham.sample(n=min_size, random_state=RANDOM_STATE)
    corpus_spam_balanced = corpus_spam.sample(n=min_size, random_state=RANDOM_STATE)
    balanced_corpus = pd.concat([corpus_ham_balanced, corpus_spam_balanced], ignore_index=True)
    return balanced_corpus.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def save_curated(corpus: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(path, index=False)
    label_counts = corpus["label"].value_counts().to_dict()
    print(f"\nsaved {len(corpus)} rows to {path} (label counts: {label_counts})")


def main() -> None:
    corpus = load_sources(SOURCES, ROWS_PER_SOURCE)
    inspect(corpus, "dataset-1 / merged corpus")
    print("rows per source:\n", corpus["source"].value_counts())

    corpus = combine_subject_body(corpus)
    corpus = handle_nulls(corpus)
    corpus = deduplicate(corpus)
    corpus = balance_labels(corpus)
    save_curated(corpus, OUTPUT_PATH)


if __name__ == "__main__":
    main()
