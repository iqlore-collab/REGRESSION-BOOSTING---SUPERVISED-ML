
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Housing Price Predictor", layout="wide")

st.title("🏠 Housing Price Prediction App")
st.markdown("### XGBoost-powered Machine Learning Prediction")

MODEL_PATH = Path("tuned_xgboost_housing_model.pkl")

if not MODEL_PATH.exists():
    st.error("Model file not found. Upload or place tuned_xgboost_housing_model.pkl in the project directory.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("House Features")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area", min_value=200, max_value=6000, value=1500)
garage_cars = st.sidebar.slider("Garage Capacity", 0, 5, 2)
total_bsmt_sf = st.sidebar.number_input("Basement Area", min_value=0, max_value=4000, value=900)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 5, 2)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2026, value=2000)
lot_area = st.sidebar.number_input("Lot Area", min_value=1000, max_value=100000, value=8500)

sample = pd.DataFrame({
    "OverallQual": [overall_qual],
    "GrLivArea": [gr_liv_area],
    "GarageCars": [garage_cars],
    "TotalBsmtSF": [total_bsmt_sf],
    "FullBath": [full_bath],
    "YearBuilt": [year_built],
    "LotArea": [lot_area]
})

prediction = model.predict(sample)[0]

st.metric("Predicted House Price", f"${prediction:,.0f}")

st.markdown("---")
st.markdown("## Input Features")
st.dataframe(sample)

st.markdown("---")
st.markdown("Built with Streamlit, Scikit-learn and XGBoost.")
