# AltCredit

> Alternative credit scoring for the unbanked using non-traditional signals.

Uses social connectivity, behavioral data, device freshness, and transaction regularity to assess creditworthiness for populations without traditional credit history. Trains and compares four classifiers (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) with 5-fold cross-validation.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.716 |
| Gini | 0.432 |
| KS Statistic | 0.345 |
| F1 Score | 0.416 |
| Accuracy | 0.651 |

All four models are benchmarked on the same holdout set with 5-fold cross-validation.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Dataset overview, class distribution, raw feature table |
| **Model Lab** | Multi-model comparison table, ROC curves, calibration plots, confusion matrix, CV results |
| **Scoring** | Alternative credit score formula, score bands, default rate by band |
| **Behavioral** | Behavioral feature distributions by credit outcome |
| **Inclusion** | Financial inclusion metrics, approval rates by employment sector |

## Repo Structure

```
AltCredit/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic dataset engineered to mirror alternative credit scoring signals: social connectivity score, behavioral score, bill payment consistency, transaction regularity, device freshness, and employment sector.

## License

MIT
