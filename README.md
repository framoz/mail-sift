# MailSift 

A machine learning project developed in the context of the **Artificial Intelligence** course at **FEUP**

## Overview

This project aims to build a classifier that, given an email, predicts multiple labels along the following dimensions:

- **Spam** — whether the email is unsolicited bulk mail.
- **Phishing** — whether the email is an attempt to deceive the recipient into disclosing sensitive information.
- **Urgency** — how time-sensitive the email is.
- **Sentiment** — the overall tone of the message (e.g. positive, neutral, negative).

## Goals

- Explore and compare different ML approaches (classical models and, where appropriate, deep learning) for multi-label email classification.
- Build a preprocessing pipeline suitable for raw email text (headers, body, metadata).
- Evaluate models on each task with appropriate metrics and analyse trade-offs between them.

## Repository Structure

```
.
├── data/        # Datasets used for training and evaluation
└── README.md
```

## Status

Early stage — project scaffolding.
