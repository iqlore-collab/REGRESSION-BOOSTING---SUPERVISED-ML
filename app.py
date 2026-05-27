
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Housing Price Prediction App")
st.markdown("### XGBoost-powered Machine Learning Prediction")

DATA_PATH = "housing_iteration_6_regression.csv"
TARGET = "SalePrice"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data


def add_engineered_features(df):
    df = df.copy()

    if {"TotalBsmtSF", "GrLivArea"}.issubset(df.columns):
        df["TotalLivingArea"] = df["TotalBsmtSF"] + df["GrLivArea"]

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    bath_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    if set(bath_cols).issubset(df.columns):
        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
    existing_porch_cols = [col for col in porch_cols if col in df.columns]
    if existing_porch_cols:
        df["TotalOutdoorArea"] = df[existing_porch_cols].sum(axis=1)

    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

    return df


@st.cache_resource
def train_model(data):
    df = add_engineered_features(data)

    if TARGET not in df.columns:
        st.error(f"Target column '{TARGET}' was not found in the dataset.")
        st.stop()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return model, X, metrics


try:
    housing = load_data()
except FileNotFoundError:
    st.error(
        "Dataset file not found. Please upload housing_iteration_6_regression.csv "
        "to the same GitHub repository as app.py."
    )
    st.stop()

model, X_template, metrics = train_model(housing)

st.success("Model trained successfully from the dataset!")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"${metrics['MAE']:,.0f}")
col2.metric("RMSE", f"${metrics['RMSE']:,.0f}")
col3.metric("R²", f"{metrics['R2']:.3f}")

st.markdown("---")
st.sidebar.header("House Features")

# Create a default row using median/mode values from the training data
sample = X_template.copy().iloc[[0]].copy()

for col in X_template.columns:
    if pd.api.types.is_numeric_dtype(X_template[col]):
        sample[col] = X_template[col].median()
    else:
        mode_value = X_template[col].mode(dropna=True)
        sample[col] = mode_value.iloc[0] if len(mode_value) else X_template[col].iloc[0]


def set_if_exists(column, value):
    if column in sample.columns:
        sample[column] = value


overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area", min_value=200, max_value=6000, value=1500)
garage_cars = st.sidebar.slider("Garage Capacity", 0, 5, 2)
garage_area = st.sidebar.number_input("Garage Area", min_value=0, max_value=2000, value=500)
total_bsmt_sf = st.sidebar.number_input("Basement Area", min_value=0, max_value=4000, value=900)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 5, 2)
half_bath = st.sidebar.slider("Half Bathrooms", 0, 3, 1)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2026, value=2000)
year_remod = st.sidebar.number_input("Year Remodeled", min_value=1800, max_value=2026, value=2005)
yr_sold = st.sidebar.number_input("Year Sold", min_value=2000, max_value=2026, value=2010)
lot_area = st.sidebar.number_input("Lot Area", min_value=1000, max_value=100000, value=8500)

set_if_exists("OverallQual", overall_qual)
set_if_exists("GrLivArea", gr_liv_area)
set_if_exists("GarageCars", garage_cars)
set_if_exists("GarageArea", garage_area)
set_if_exists("TotalBsmtSF", total_bsmt_sf)
set_if_exists("FullBath", full_bath)
set_if_exists("HalfBath", half_bath)
set_if_exists("YearBuilt", year_built)
set_if_exists("YearRemodAdd", year_remod)
set_if_exists("YrSold", yr_sold)
set_if_exists("LotArea", lot_area)

# Recreate engineered features for the single prediction row
sample = add_engineered_features(sample)

prediction = model.predict(sample)[0]

st.markdown("## Predicted Price")
st.metric("Estimated Sale Price", f"${prediction:,.0f}")

st.markdown("## Selected Input Features")
display_cols = [
    col for col in [
        "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
        "TotalBsmtSF", "FullBath", "HalfBath", "YearBuilt",
        "YearRemodAdd", "YrSold", "LotArea", "TotalLivingArea",
        "HouseAge", "TotalBathrooms", "HasGarage"
    ]
    if col in sample.columns
]

st.dataframe(sample[display_cols])

st.markdown("---")
st.caption("Built with Streamlit, Scikit-learn and XGBoost.")
