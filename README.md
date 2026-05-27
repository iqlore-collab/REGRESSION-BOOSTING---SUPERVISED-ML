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

# 🏠 Predykcja Cen Domów — Zaawansowany Projekt Machine Learning (Regresja)

<div align="center">

### Kompleksowy Workflow Machine Learning do Predykcji Cen Nieruchomości

Zbudowany z użyciem:

`Python` • `Scikit-learn` • `XGBoost` • `SHAP` • `Pandas` • `Seaborn`

</div>

---

# 📌 Opis Projektu

Projekt skupia się na przewidywaniu cen nieruchomości przy użyciu zaawansowanych metod supervised machine learning.

To, co zaczęło się jako prosty notebook regresyjny, zostało rozbudowane do pełnego workflow machine learning obejmującego:

- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering
- ✅ Feature Selection
- ✅ Ensemble Learning
- ✅ Boosting Algorithms
- ✅ Hyperparameter Tuning
- ✅ SHAP Explainability
- ✅ Learning Curves
- ✅ Analizę błędów modeli
- ✅ Pipeline preprocessing
- ✅ Cross-Validation
- ✅ Zapisywanie modelu

Finalny notebook łączy rozbudowaną analizę wizualną z nowoczesnymi modelami regresyjnymi opartymi o XGBoost.

---

# 🎯 Główny Cel Projektu

Celem projektu jest przewidywanie:

```python
SalePrice
```

na podstawie danych dotyczących nieruchomości, takich jak:
- powierzchnia domu
- informacje o garażu
- jakość wykonania
- lokalizacja / neighborhood
- historia remontów
- piwnice
- powierzchnia zewnętrzna
- cechy konstrukcyjne

---

# 📂 Dataset

Dataset używany w Google Colab:

```python
/content/drive/MyDrive/Colab Notebooks/housing_iteration_6_regression.csv
```

Dataset zawiera:
- zmienne numeryczne
- zmienne kategoryczne
- brakujące wartości
- dane rynku nieruchomości
- strukturalne dane tabularne

---

# 🛠️ Technologie

## 📊 Analiza Danych i Wizualizacja
- Pandas
- NumPy
- Matplotlib
- Seaborn

## 🤖 Machine Learning
- Scikit-learn
- XGBoost
- SHAP

## ☁️ Środowisko
- Google Colab
- Google Drive

## 💾 Zapisywanie Modelu
- Joblib

---

# 🔎 Exploratory Data Analysis (EDA)

Notebook zawiera rozbudowaną analizę danych:

### ✔ Analiza Missing Values
- liczba brakujących wartości
- kolumny z największą ilością braków

### ✔ Analiza Korelacji
- heatmapy korelacji
- najsilniejsze zależności z `SalePrice`

### ✔ Wizualizacja Feature'ów
- scatterploty
- regression trend lines
- boxploty
- analiza neighborhood
- analiza engineered features

### ✔ Analiza Rozkładów
- rozkład targetu
- log-transformacja targetu
- rozkład residuals
- rozkład błędów

---

# 🧠 Feature Engineering

Projekt zawiera custom engineered features:

| Engineered Feature | Opis |
|---|---|
| `TotalLivingArea` | Powierzchnia mieszkalna + piwnica |
| `HouseAge` | Wiek nieruchomości |
| `YearsSinceRemodel` | Lata od remontu |
| `TotalBathrooms` | Łączna liczba łazienek |
| `TotalOutdoorArea` | Łączna powierzchnia zewnętrzna |
| `HasGarage` | Informacja o garażu |

---

# ⚙️ Preprocessing Pipeline

Projekt wykorzystuje pełne pipeline preprocessing:

```python
Pipeline(...)
ColumnTransformer(...)
```

### Numeric Features
- imputacja medianą
- scaling

### Categorical Features
- imputacja najczęstszej wartości
- one-hot encoding

Dzięki temu workflow jest:
- czysty
- powtarzalny
- production-style

---

# 📈 Modele Regresyjne

Projekt porównuje kilka modeli regresyjnych.

## 1️⃣ Linear Regression
Bazowy model liniowy.

## 2️⃣ Ridge Regression
Regresja liniowa z regularyzacją.

## 3️⃣ Random Forest Regressor
Silny ensemble tree model.

## 4️⃣ XGBoost Regressor
Najbardziej zaawansowany model w projekcie.

---

# 🚀 Dlaczego XGBoost?

XGBoost wykorzystuje **boosting**, czyli:

1. jeden model wykonuje predykcję
2. kolejny model uczy się błędów poprzedniego
3. następny model poprawia wcześniejsze błędy
4. proces powtarza się iteracyjnie

W przeciwieństwie do Random Forest, XGBoost stale poprawia wcześniejsze predykcje.

To sprawia, że świetnie działa dla:
- danych tabularnych
- regresji
- predykcji cen
- Kaggle competitions
- structured machine learning

---

# 🧪 Feature Selection

Notebook zawiera również zaawansowane techniki selekcji cech:

## ✔ Variance Threshold
Usuwanie cech o bardzo małej wariancji.

## ✔ Correlation Filtering
Usuwanie mocno skorelowanych zmiennych.

## ✔ SelectKBest
Wybór najważniejszych statystycznie feature'ów.

## ✔ SelectFromModel
Wybór feature'ów na podstawie ważności modelu.

## ✔ Recursive Feature Elimination (RFE)
Iteracyjne usuwanie słabszych feature'ów.

---

# 🔥 Zaawansowane Elementy ML

## ✔ Cross-Validation
Bardziej wiarygodna ocena modeli.

## ✔ Hyperparameter Tuning
Optymalizacja parametrów XGBoost przy użyciu RandomizedSearchCV.

## ✔ Learning Curves
Analiza underfitting / overfitting.

## ✔ SHAP Explainability
Explainable AI oraz interpretacja wpływu feature'ów.

## ✔ Advanced Error Analysis
Analiza największych błędów predykcji.

## ✔ Model Persistence
Zapisywanie i ładowanie modeli `.pkl`.

---

# 📊 Ocena Modeli

Modele oceniane są przy użyciu:

| Metryka | Znaczenie |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Explained Variance |

Dodatkowo notebook zawiera:
- actual vs predicted plots
- residual analysis
- analiza błędów procentowych
- feature importance
- SHAP summary plots

---

# 📉 Wizualizacje

Projekt zawiera dużą ilość wizualizacji:

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

# 🧱 Struktura Projektu

```text
housing-price-regression/
│
├── housing_price_regression_advanced_ml_project.ipynb
├── housing_iteration_6_regression.csv
├── tuned_xgboost_housing_model.pkl
├── README.md
```

---

# ▶ Uruchomienie Notebooka

## Google Colab Setup

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Instalacja Bibliotek

```python
!pip install xgboost shap -q
```

---

# 💡 Najważniejsze Elementy Projektu

Projekt pokazuje:
- kompletny workflow ML
- preprocessing pipelines
- boosting algorithms
- explainable AI
- feature engineering
- feature selection
- hyperparameter tuning
- model diagnostics
- reproducible machine learning

---

# 🏁 Final Thoughts

Największą wartością projektu nie jest sam wynik modelu.

Najważniejsze jest zrozumienie:
- które feature'y wpływają na ceny nieruchomości
- jak boosting poprawia predykcje
- jak feature selection wpływa na model
- jak residual analysis pokazuje słabe strony modelu
- jak działa nowoczesny workflow machine learning

Projekt ewoluował z prostego notebooka regresyjnego do znacznie bardziej realistycznego portfolio machine learning.
