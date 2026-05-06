# Galaxy Redshift Prediction using Machine Learning

## Overview
This project predicts galaxy redshift using photometric features from the Sloan Digital Sky Survey (SDSS).

The model uses:
- u, g, r, i, z photometric bands
- feature engineering using color indices
- polynomial feature expansion
- XGBoost regression

## Features Used
- u, g, r, i, z magnitudes
- color indices:
  - u-g
  - g-r
  - r-i
  - i-z

## Model
- XGBoost Regressor
- Polynomial Features
- Log-transformed target

## Performance
- R² Score: ~0.75

## Dataset
Data obtained from SDSS SkyServer.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost