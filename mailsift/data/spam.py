import pandas as pd

from mailsift.config import DATASETS_DIR


def inspect(df, name):
    print(f"\n--- {name} ---")
    print("shape :", df.shape)
    print("dtypes:\n", df.dtypes)
    print("nulls :\n", df.isna().sum())
    print("head  :\n", df.head(3))

sources = ["CEAS_08", "Enron", "Ling", "SpamAssasin"]
frames = []
for name in sources:
    df = pd.read_csv(
        f"{DATASETS_DIR}/dataset-1/{name}.csv",
        usecols=["subject", "body", "label"],
        dtype={"label": "int8"},
    )
    df["source"] = name
    frames.append(df)
corpus = pd.concat(frames, ignore_index=True)
inspect(corpus, "dataset-1 / merged corpus")
print("rows per source:\n", corpus["source"].value_counts())