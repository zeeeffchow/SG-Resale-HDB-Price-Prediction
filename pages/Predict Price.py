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

# --- LOAD MODEL + ENCODERS ---
model = load_model()
encoders = load_encoders()
le_town = encoders["town"]
le_flat_type = encoders["flat_type"]
le_flat_model = encoders["flat_model"]
le_storey = encoders["storey_range"]

# --- UI CONFIG ---
st.set_page_config(page_title="Predict HDB Price", layout="wide")
st.title("📈 Predict HDB Resale Price")

with st.form("prediction_form"):
    st.subheader("📋 Enter HDB Flat Details")

    town = st.selectbox("Town", le_town.classes_.tolist())
    flat_type = st.selectbox("Flat Type", le_flat_type.classes_.tolist())
    flat_model = st.selectbox("Flat Model", le_flat_model.classes_.tolist())
    storey_range = st.selectbox("Storey Range", le_storey.classes_.tolist())

    floor_area_sqm = st.number_input("Floor Area (sqm)", 30.0, 200.0, step=1.0)
    lease_commence_date = st.slider("Lease Commencement Year", 1960, date.today().year, 2000)
    remaining_lease = st.slider("Remaining Lease (Years)", 1, 99, 80)

    resale_month = st.date_input("Resale Month", value=date.today())
    year = resale_month.year
    month_num = resale_month.month

    latitude = st.number_input("Latitude", value=1.35)
    longitude = st.number_input("Longitude", value=103.8)

    st.markdown("### 📍 Location-based Features")
    mrt_dist = st.number_input("Distance to Nearest MRT (km)", 0.0, 10.0, 0.5)
    mrt_count = st.slider("MRTs within 1km", 0, 10, 2)
    hawker_dist = st.number_input("Distance to Nearest Hawker (km)", 0.0, 5.0, 0.3)
    hawker_count = st.slider("Hawkers within 1km", 0, 10, 3)
    library_dist = st.number_input("Distance to Nearest Library (km)", 0.0, 5.0, 1.0)
    library_count = st.slider("Libraries within 1km", 0, 10, 1)
    hospital_dist = st.number_input("Distance to Nearest Hospital (km)", 0.0, 10.0, 2.0)
    hospital_count = st.slider("Hospitals within 1km", 0, 10, 1)
    childcare_dist = st.number_input("Distance to Nearest Childcare (km)", 0.0, 5.0, 0.5)
    childcare_count = st.slider("Childcares within 1km", 0, 10, 2)
    park_dist = st.number_input("Distance to Nearest Park (km)", 0.0, 5.0, 0.5)
    park_count = st.slider("Parks within 1km", 0, 10, 2)
    cc_dist = st.number_input("Distance to Nearest Community Club (km)", 0.0, 5.0, 0.5)
    cc_count = st.slider("Community Clubs within 1km", 0, 10, 2)
    kindergarten_dist = st.number_input("Distance to Nearest Kindergarten (km)", 0.0, 5.0, 0.5)
    kindergarten_count = st.slider("Kindergartens within 1km", 0, 10, 2)
    school_dist = st.number_input("Distance to Nearest School (km)", 0.0, 5.0, 0.5)
    school_count = st.slider("Schools within 1km", 0, 10, 3)
    downtown_dist = st.number_input("Distance to Downtown Core (km)", 0.0, 30.0, 10.0)

    submitted = st.form_submit_button("Predict Price")

# --- PREDICT ---
if submitted:
    input_data = pd.DataFrame([{
        "town": le_town.transform([town])[0],
        "flat_type": le_flat_type.transform([flat_type])[0],
        "storey_range": le_storey.transform([storey_range])[0],
        "floor_area_sqm": floor_area_sqm,
        "flat_model": le_flat_model.transform([flat_model])[0],
        "lease_commence_date": lease_commence_date,
        "remaining_lease": remaining_lease,
        "latitude": latitude,
        "longitude": longitude,
        "distance_to_nearest_mrt_km": mrt_dist,
        "num_mrt_within_1km": mrt_count,
        "distance_to_nearest_hawker_km": hawker_dist,
        "num_hawkers_within_1km": hawker_count,
        "distance_to_nearest_library_km": library_dist,
        "num_librarys_within_1km": library_count,
        "distance_to_nearest_hospital_km": hospital_dist,
        "num_hospitals_within_1km": hospital_count,
        "distance_to_nearest_childcare_km": childcare_dist,
        "num_childcares_within_1km": childcare_count,
        "distance_to_nearest_park_km": park_dist,
        "num_parks_within_1km": park_count,
        "distance_to_nearest_communityclub_km": cc_dist,
        "num_communityclubs_within_1km": cc_count,
        "distance_to_nearest_kindergarten_km": kindergarten_dist,
        "num_kindergartens_within_1km": kindergarten_count,
        "distance_to_nearest_school_km": school_dist,
        "num_schools_within_1km": school_count,
        "year": year,
        "month_num": month_num,
        "distance_to_downtown_core_km": downtown_dist
    }])

    predicted_price = model.predict(input_data)[0]
    st.success(f"🏷️ Estimated Resale Price: **${predicted_price:,.0f}**")
    st.caption(f"💡 Fair Price Range: ${predicted_price*0.9:,.0f} - ${predicted_price*1.1:,.0f}")
