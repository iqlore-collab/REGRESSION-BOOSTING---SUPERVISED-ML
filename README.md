# 🏠 Housing Price Prediction — Advanced Regression ML Project

<div align="center">

### End-to-End Machine Learning Workflow for Real Estate Price Prediction

Built with:

`Python` • `Scikit-learn` • `XGBoost` • `SHAP` • `Pandas` • `Seaborn`

</div>

---

# 📌 Project Overview

This project focuses on predicting housing prices using advanced supervised machine learning techniques.

What started as a simple regression notebook gradually evolved into a much more complete machine learning workflow including:

- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering
- ✅ Feature Selection
- ✅ Ensemble Learning
- ✅ Boosting Algorithms
- ✅ Hyperparameter Tuning
- ✅ SHAP Explainability
- ✅ Learning Curves
- ✅ Error Diagnostics
- ✅ Pipeline-Based Preprocessing
- ✅ Cross-Validation
- ✅ Model Persistence

The final notebook combines strong visual analysis with advanced regression modeling using XGBoost and modern ML workflows.

---

# 🎯 Main Objective

The goal of this project is to predict:

```python
SalePrice
```

using structured housing market data such as:
- property size
- neighborhood information
- garage features
- quality indicators
- remodeling history
- basement metrics
- outdoor space
- construction characteristics

---

# 📂 Dataset

Dataset used in Google Colab:

```python
/content/drive/MyDrive/Colab Notebooks/housing_iteration_6_regression.csv
```

The dataset contains:
- numerical features
- categorical variables
- missing values
- real-world housing attributes
- structured tabular data

---

# 🛠️ Technologies Used

## 📊 Data Analysis & Visualization
- Pandas
- NumPy
- Matplotlib
- Seaborn

## 🤖 Machine Learning
- Scikit-learn
- XGBoost
- SHAP

## ☁️ Environment
- Google Colab
- Google Drive

## 💾 Model Persistence
- Joblib

---

# 🔎 Exploratory Data Analysis (EDA)

The notebook includes extensive visual analysis such as:

### ✔ Missing Values Analysis
- missing value distribution
- top columns with missing data

### ✔ Correlation Analysis
- correlation heatmaps
- strongest relationships with `SalePrice`

### ✔ Feature Visualizations
- scatterplots
- regression trend plots
- boxplots
- neighborhood price comparisons
- engineered feature analysis

### ✔ Distribution Analysis
- target distribution
- log-transformed target
- residual distributions
- error distributions

---

# 🧠 Feature Engineering

Several custom engineered features were created to improve model performance:

| Engineered Feature | Description |
|---|---|
| `TotalLivingArea` | Basement + above-ground area |
| `HouseAge` | Property age |
| `YearsSinceRemodel` | Years since remodeling |
| `TotalBathrooms` | Weighted bathroom count |
| `TotalOutdoorArea` | Combined outdoor area |
| `HasGarage` | Binary garage indicator |

---

# ⚙️ Preprocessing Pipeline

The project uses full preprocessing pipelines with:

```python
Pipeline(...)
ColumnTransformer(...)
```

### Numeric Features
- median imputation
- feature scaling

### Categorical Features
- most frequent imputation
- one-hot encoding

This creates a clean and reproducible machine learning workflow.

---

# 📈 Regression Models

Several regression models were trained and compared.

## 1️⃣ Linear Regression
Simple baseline model.

## 2️⃣ Ridge Regression
Linear regression with regularization.

## 3️⃣ Random Forest Regressor
Strong ensemble tree model.

## 4️⃣ XGBoost Regressor
The strongest and most advanced model in the notebook.

---

# 🚀 Why XGBoost Matters

XGBoost uses **boosting**, meaning:

1. one model makes predictions
2. another model learns previous errors
3. the next model improves those errors
4. the process repeats iteratively

Unlike Random Forest, XGBoost continuously improves previous predictions.

This makes it extremely powerful for:
- structured datasets
- regression tasks
- housing prediction
- Kaggle competitions
- tabular machine learning

---

# 🧪 Feature Selection

The notebook also explores advanced feature selection methods:

## ✔ Variance Threshold
Removes nearly constant features.

## ✔ Correlation Filtering
Removes highly correlated variables.

## ✔ SelectKBest
Selects statistically important predictors.

## ✔ SelectFromModel
Uses model importance to choose features.

## ✔ Recursive Feature Elimination (RFE)
Iteratively removes weaker features.

---

# 🔥 Advanced ML Features

## ✔ Cross-Validation
More reliable model evaluation across multiple folds.

## ✔ Hyperparameter Tuning
RandomizedSearchCV optimization for XGBoost.

## ✔ Learning Curves
Train vs validation performance analysis.

## ✔ SHAP Explainability
Advanced model interpretability and feature impact analysis.

## ✔ Advanced Error Analysis
Identification of large prediction errors and outliers.

## ✔ Model Persistence
Saving and loading trained models using Joblib.

---

# 📊 Model Evaluation

Models are evaluated using:

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Explained Variance |

Additional diagnostics include:
- actual vs predicted plots
- residual analysis
- percentage error analysis
- feature importance visualizations
- SHAP summary plots

---

# 📉 Visualizations Included

The notebook contains extensive visualizations including:

- ✅ Correlation heatmaps
- ✅ Scatterplots
- ✅ Regression lines
- ✅ Boxplots
- ✅ Missing value charts
- ✅ Feature importance plots
- ✅ Residual plots
- ✅ Error distributions
- ✅ Learning curves
- ✅ SHAP plots
- ✅ Model comparison charts

---

# 🧱 Project Structure

```text
housing-price-regression/
│
├── housing_price_regression_advanced_ml_project.ipynb
├── housing_iteration_6_regression.csv
├── tuned_xgboost_housing_model.pkl
├── README.md
```

---

# ▶ Running the Notebook

## Google Colab Setup

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Install Required Libraries

```python
!pip install xgboost shap -q
```

---

# 💡 Key Takeaways

This project demonstrates:
- end-to-end ML workflow design
- preprocessing pipelines
- advanced regression modeling
- boosting algorithms
- explainable AI
- feature engineering
- feature selection
- hyperparameter tuning
- model diagnostics
- reproducible machine learning workflows

---

# 🏁 Final Thoughts

The most valuable part of this project is not only the final score.

The real value comes from understanding:
- what drives housing prices
- how boosting improves predictions
- how feature selection changes model behavior
- how residual analysis reveals weaknesses
- how explainability improves trust in models
- how modern machine learning workflows are structured

This notebook evolved into a much more realistic portfolio-level machine learning project rather than a simple regression exercise.
