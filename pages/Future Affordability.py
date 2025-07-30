import streamlit as st
import pandas as pd
import joblib
from datetime import date

# --- CACHED LOADING ---
@st.cache_resource
def load_model():
    return joblib.load("./model/best_rf_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("./model/label_encoders.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/User/Documents/projects/SG Resale HDB Price Prediction/data/final_data/resale_data.csv")

@st.cache_data
def predict_all(df_encoded: pd.DataFrame, _model) -> pd.DataFrame:
    df_model_input = df_encoded[_model.feature_names_in_].copy()
    df_encoded = df_encoded.copy()
    df_encoded["predicted_price"] = _model.predict(df_model_input)
    return df_encoded

# --- LOAD ONCE ---
model = load_model()
encoders = load_encoders()
df_raw = load_data()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Future Affordability", layout="wide")
st.title("🔮 What Can I Afford in the Future?")

# --- USER INPUT: DATE AND BUDGET ---
col1, col2 = st.columns(2)
future_date = col1.date_input("Target Date", value=date(2026, 1, 1), min_value=date(2025, 8, 1))
budget = col2.number_input("Your Budget ($)", 100000, 1500000, 600000, step=10000)

year = future_date.year
month_num = future_date.month

# --- PREPARE DATA FOR MODEL ---
df = df_raw.drop(columns=["resale_price"], errors="ignore").copy()
df["year"] = year
df["month_num"] = month_num

# --- ADD READABLE CATEGORICALS ---
df["town_str"] = encoders["town"].inverse_transform(df["town"])
df["flat_type_str"] = encoders["flat_type"].inverse_transform(df["flat_type"])

# --- RUN PREDICTION ONCE (CACHED) ---
df = predict_all(df, model)

# --- FILTER UI ---
with st.expander("📊 Optional Filters", expanded=True):
    town_options = sorted(df["town_str"].unique())
    flat_options = sorted(df["flat_type_str"].unique())

    selected_towns = st.multiselect("Filter by Town", town_options, default=town_options)
    selected_flats = st.multiselect("Filter by Flat Type", flat_options, default=flat_options)
    min_area = st.slider("Minimum Floor Area (sqm)", 30, 150, 60, step=5)

# --- APPLY FILTERS ON PREDICTED DATA ---
df_filtered = df[
    (df["town_str"].isin(selected_towns)) &
    (df["flat_type_str"].isin(selected_flats)) &
    (df["floor_area_sqm"] >= min_area) &
    (df["predicted_price"] <= budget)
].sort_values("floor_area_sqm", ascending=False)

# --- DISPLAY RESULTS ---
st.markdown(f"### Flats under ${budget:,.0f} in {future_date.strftime('%B %Y')}")

if df_filtered.empty:
    st.warning("No flats found within your budget and filters.")
else:
    df_display = df_filtered[["town", "flat_type", "floor_area_sqm", "storey_range", "predicted_price"]].copy()
    df_display["town"] = encoders["town"].inverse_transform(df_display["town"])
    df_display["flat_type"] = encoders["flat_type"].inverse_transform(df_display["flat_type"])
    df_display["storey_range"] = encoders["storey_range"].inverse_transform(df_display["storey_range"])

    st.dataframe(
        df_display.rename(columns={
            "town": "Town",
            "flat_type": "Flat Type",
            "floor_area_sqm": "Floor Area (sqm)",
            "storey_range": "Storey",
            "predicted_price": "Predicted Price ($)"
        }).reset_index(drop=True),
        use_container_width=True
    )
