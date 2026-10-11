# Production model card: NASDAQ-100 next day direction (production_v1)

Model card for the production artefact `models/prod_model_training_xgboost.joblib`. All numbers come from `notebooks/03_xgboost_model.ipynb`, run on the original course dataset (2015-02-17 to 2025-02-13, 100 tickers).

## Summary

production_v1 is a gradient boosted tree classifier (XGBoost) that estimates whether a NASDAQ-100 constituent will close higher on the next trading day. For each ticker and date it outputs a class, where 1 means up and 0 means down, and the probability of the up class. The tool is decision support for analysts. It is not an automated trader and it is not investment advice.

## Inputs

The production model reads three features per ticker and date. return_1d is the one day return. vix_close is the closing level of the VIX volatility index. vix_change_1d is the one day change in the VIX. The feature order is fixed and sorted, so the serving frame must present the columns in the order return_1d, vix_change_1d, vix_close.

## Training and validation

The model is evaluated with a walk forward scheme that always trains on past years and tests on the next unseen year, which avoids look ahead leakage. The folds tested the years 2020 through 2025 in turn, with the training window growing from 2015 up to the year before each test year. Applied identically to every fold (same three features, same hyperparameters), the production configuration reached a mean precision of 0.532 (std 0.036), a mean accuracy of 0.528 and a mean F1 of 0.672. A logistic regression on the same three features scores 0.533 precision, and always predicting "up" scores about 0.53 (the share of up days), so these results sit close to a simple always up baseline, which reflects how hard single day direction is to predict. Treat the signal as a small tilt rather than a reliable forecast.

## Intended use and users

The intended users are the project analysts who review the daily predictions table inside this dashboard. The intended use is to surface the model opinion next to a person's own research. Any decision to act stays with the user.

## Limitations

The edge over a naive baseline is small and varies a lot by year, so confidence values near 50 percent carry little information. The model has no view of news, earnings, or events beyond the three numeric features. A market regime shift can degrade performance between retrains. The model must not be used as financial advice and it does not place orders.

## Compliance

This is an AI system that informs a human decision. A visible disclosure is shown on every page of the dashboard in line with EU AI Act Article 50. No personal data is used. Source price and macro data come from public market feeds.
