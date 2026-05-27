# Housing Price Prediction App

This Streamlit app trains an XGBoost regression model directly from the dataset and predicts housing prices.

## Required files

Make sure these files are in the same GitHub repository:

```text
app.py
requirements.txt
runtime.txt
housing_iteration_6_regression.csv
```

The app no longer requires `tuned_xgboost_housing_model.pkl`, which avoids pickle/joblib compatibility issues on Streamlit Cloud.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

Main file path:

```text
app.py
```

Python version is pinned in:

```text
runtime.txt
```