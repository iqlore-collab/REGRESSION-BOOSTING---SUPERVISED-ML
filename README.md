# 🏠 Housing Price Prediction — Full Regression Machine Learning Project

<div align="center">

### End-to-End Supervised Machine Learning Workflow for Real Estate Price Prediction

Built with:

`Python` • `Scikit-learn` • `XGBoost` • `Pandas` • `Seaborn`

</div>

---

# 📌 Project Overview

This project focuses on predicting housing prices using modern regression machine learning techniques.

What started as a simple regression notebook gradually evolved into a much more complete ML workflow including:

- ✅ Exploratory Data Analysis
- ✅ Feature Engineering
- ✅ Feature Selection
- ✅ Ensemble Learning
- ✅ Boosting Algorithms
- ✅ Model Diagnostics
- ✅ Pipeline-Based Preprocessing
- ✅ Regression Performance Evaluation

The final notebook combines strong visual analysis with advanced supervised machine learning models such as XGBoost.

---

# 🎯 Main Objective

The goal of this project is to predict:

```python
SalePrice
```

using structured housing data such as:
- property size
- garage information
- quality indicators
- neighborhood data
- renovation history
- basement features
- outdoor space

---

# 📂 Dataset

Dataset path used in Google Colab:

```python
/content/drive/MyDrive/Colab Notebooks/housing_iteration_6_regression.csv
```

Dataset includes:
- numerical features
- categorical variables
- missing values
- engineered housing metrics
- real-world tabular structure

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

## ☁️ Environment
- Google Colab
- Google Drive integration

---

# 🔎 Exploratory Data Analysis (EDA)

The notebook includes extensive visual analysis:

### ✔ Missing Values Analysis
- top missing columns
- missing value distribution

### ✔ Correlation Analysis
- correlation heatmaps
- strongest predictors of `SalePrice`

### ✔ Visual Feature Exploration
- scatterplots
- boxplots
- neighborhood analysis
- engineered feature analysis

### ✔ Distribution Analysis
- target distribution
- log-transformed target distribution
- residual distributions

---

# 🧠 Feature Engineering

Several custom features were created to improve model performance:

| Engineered Feature | Description |
|---|---|
| `TotalLivingArea` | Basement + above-ground area |
| `HouseAge` | Age of the property |
| `YearsSinceRemodel` | Time since renovation |
| `TotalBathrooms` | Weighted bathroom count |
| `TotalOutdoorArea` | Combined porch/deck space |
| `HasGarage` | Binary garage indicator |

---

# ⚙️ Preprocessing Pipeline

The project uses a full preprocessing pipeline with:

```python
Pipeline(...)
ColumnTransformer(...)
```

### Numeric Features
- median imputation
- scaling

### Categorical Features
- missing value imputation
- one-hot encoding

This makes the workflow reproducible and production-style.

---

# 📈 Regression Models

Several regression algorithms were trained and compared.

## 1️⃣ Linear Regression
Simple baseline model.

## 2️⃣ Ridge Regression
Linear model with regularization.

## 3️⃣ Random Forest Regressor
Powerful ensemble tree-based method.

## 4️⃣ XGBoost Regressor
The strongest and most advanced model in the project.

---

# 🚀 Why XGBoost Matters

XGBoost uses **boosting**, meaning:

1. one model makes predictions
2. another model learns the previous errors
3. the next model improves the mistakes again
4. the process repeats iteratively

Instead of building trees independently like Random Forest, XGBoost continuously improves previous predictions.

This makes it extremely powerful for:
- tabular datasets
- housing prediction
- Kaggle competitions
- structured regression problems

---

# 🧪 Feature Selection

The notebook also explores advanced feature selection techniques:

## ✔ Variance Threshold
Removes nearly constant features.

## ✔ Correlation Filtering
Removes highly correlated variables.

## ✔ SelectKBest
Selects strongest statistically related features.

## ✔ SelectFromModel
Uses model importance to select variables.

## ✔ Recursive Feature Elimination (RFE)
Iteratively removes weaker predictors.

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
- error distribution plots
- feature importance visualizations

---

# 📉 Visualizations Included

The notebook contains many visualizations including:

- ✅ Correlation heatmaps
- ✅ Scatterplots
- ✅ Regression trend lines
- ✅ Boxplots
- ✅ Missing value charts
- ✅ Feature importance plots
- ✅ Residual plots
- ✅ Error distributions
- ✅ Model comparison charts

---

# 🧱 Project Structure

```text
housing-price-regression/
│
├── housing_price_regression_full_project_xgboost.ipynb
├── housing_iteration_6_regression.csv
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

### Install XGBoost (if needed)

```python
!pip install xgboost -q
```

---

# 💡 Key Takeaways

This project demonstrates much more than simple regression.

It shows:
- full ML workflow design
- preprocessing pipelines
- ensemble learning
- boosting algorithms
- feature engineering
- model diagnostics
- feature selection
- reproducible machine learning development

---

# 🏁 Final Thoughts

The most valuable part of this project is not only the final score.

The real value comes from understanding:
- which variables drive housing prices
- how boosting improves prediction quality
- how feature selection changes model behavior
- how residual analysis reveals model weaknesses
- how modern ML workflows are structured

This notebook evolved into a much more realistic portfolio-level machine learning project rather than a basic classroom exercise.
