import math
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from mailsift.config import CURATED_DIR


def tokenize(text: str) -> list[str]:

    return text.lower().split()


class NaiveBayes:
    """
    Multinomial Naive Bayes for binary text classification.
    We are using logs here because of underflow then multiplying all the P(t_i | c) together.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.log_prior: dict[int, float] = {}
        self.log_likelihood: dict[int, dict[str, float]] = {}
        self.class_total: dict[int, int] = {}
        self.vocab_size: int = 0

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "NaiveBayes":

        per_class_counts: dict[int, Counter] = defaultdict(Counter)

        per_class_docs: Counter = Counter()

        vocab: set[str] = set()
        # Loop over the training data once to count up how many times each token appears in each class,
        # and how many documents belong to each class. We'll use these counts to compute the log prior and log likelihood tables.
        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            per_class_counts[label].update(tokens)
            per_class_docs[label] += 1

            # Vocab is updated for Laplace Smoothing down bellow (without it unseen tokens would have to be ignored in predict)
            vocab.update(tokens)

        self.vocab_size = len(vocab)
        n_docs = sum(per_class_docs.values())

        # Compute log prior and log likelihood tables with Laplace smoothing (we dont compute de |V| here because it's already included in the alpha term, all the terms that appear here appear here :p ).
        for c, counts in per_class_counts.items():
            total = sum(counts.values())
            self.class_total[c] = total

            self.log_prior[c] = math.log(per_class_docs[c] / n_docs)

            denom = total + self.alpha * self.vocab_size

            # log likelihood is log P(token | class) = log((count(token, class) + alpha) / (total tokens in class + alpha * vocab size))
            # this will be used in the posterior calc as log P(class | tokens) = log P(class) + sum(log P(token | class)) for each token in the input text.
            self.log_likelihood[c] = {
                tok: math.log((cnt + self.alpha) / denom) for tok, cnt in counts.items()
            }
        return self

    # For each input text, we compute the log posterior for each class and return the class with the highest posterior as the prediction.
    def _log_posterior(self, tokens: list[str], c: int) -> float:

        logp = self.log_prior[c]
        denom = self.class_total[c] + self.alpha * self.vocab_size
        unseen_logp = math.log(self.alpha / denom)
        for tok in tokens:
            # `dict.get` with a default makes the unseen-token path branchless.
            logp += self.log_likelihood[c].get(tok, unseen_logp)
        return logp

    
    def predict(self, texts: Iterable[str]) -> list[int]:

        preds: list[int] = []
        for text in texts:
            tokens = tokenize(text)
            scores = {c: self._log_posterior(tokens, c) for c in self.log_prior}
            preds.append(max(scores, key=scores.get))
        return preds

    # For evaluation, we can compute the accuracy by comparing the predicted labels to the true labels.
    def score(self, texts: Iterable[str], labels: Iterable[int]) -> float:
        """Plain accuracy: fraction of predictions that match the true label."""
        labels = list(labels)
        preds = self.predict(texts)
        return sum(p == y for p, y in zip(preds, labels)) / len(labels)


if __name__ == "__main__":
    # Accuracy: 0.9772
    #               precision    recall  f1-score   support

    #            0       0.97      0.98      0.98       417
    #            1       0.98      0.97      0.98       416

    #     accuracy                           0.98       833
    #    macro avg       0.98      0.98      0.98       833
    # weighted avg       0.98      0.98      0.98       833

    spam_data = pd.read_parquet(CURATED_DIR / "spam.parquet")
    print(spam_data.shape)

    # Accuracy: 0.9817
    #               precision    recall  f1-score   support

    #            0       0.98      0.99      0.98      7847
    #            1       0.99      0.97      0.98      7847

    #     accuracy                           0.98     15694
    #    macro avg       0.98      0.98      0.98     15694
    # weighted avg       0.98      0.98      0.98     15694

    phishing_email = pd.read_parquet(CURATED_DIR / "phishing_email.parquet")

    mydatasets = [spam_data, phishing_email]
    for spam_data in mydatasets:
        X = spam_data["text"]
        y = spam_data["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        model = NaiveBayes(alpha=1.0).fit(X_train, y_train)
        print(f"Accuracy: {model.score(X_test, y_test):.4f}")
        print(classification_report(y_test, model.predict(X_test)))
