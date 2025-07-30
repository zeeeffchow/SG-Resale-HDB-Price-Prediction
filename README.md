![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?logo=streamlit)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

# 🏠 SG HDB Resale Price Prediction

A personal machine learning and data science project to predict resale prices of public housing (HDB flats) in Singapore. This project was built to explore real-world modeling workflows and end-to-end deployment using Streamlit and a trained Random Forest model.

Unlike similar projects, I decided a stronger emphasis on **geospatial distance-based features** — such as proximity to MRT stations, hawker centres, and schools — to test the hypothesis that accessibility and location amenities play a significant role in influencing HDB resale prices.

This app allows users to:
- 📈 Predict the fair resale price of an individual HDB unit based on its attributes
- 🔮 Explore what HDB flats could be affordable at a given future date and budget
- 🧪 Interact with real resale transaction data from 2015 to 2025
- 📍 Incorporate geographic features (distances to MRTs, schools, hawkers, etc.)

---

## 🚀 Live Features

- **Predict Price**: Input town, flat type, area, lease info, and nearby amenities — get the model's estimated price.
- **Future Affordability**: Choose a future date and budget — discover what flats you could afford then.
- **Streamlit UI**: Intuitive, real-time interface for exploration and prediction.

---

## 📂 Dataset

This project uses public HDB resale transaction data from [data.gov.sg](https://data.gov.sg), combined with geospatial distance-based information (MRT, hawker centres, parks, schools, etc.). To keep this repo lightweight, the full dataset is hosted externally.

🔽 **Download the full dataset (300+MB):**  
[📁 Google Drive: Full /data Folder](https://drive.google.com/drive/folders/1QKTWNTkc8RoB7dt_kUlKGi4F_aUvH1pR?usp=sharing)

**To use:**
1. Download the full contents of the Drive folder
2. Place the folder into your project root as `./data/`
3. The final structure should look like:
 ``` 
 SG Resale HDB Price Prediction/
├── data/
│ ├── final_data/
│ └── raw_data/
├── model/
├── pages/
└── streamlit_app.py
```
---

## 📊 Data Preparation & EDA

We used HDB resale data from **2015 to 2025** (≈250,000 records). Data prior to 2015 was excluded because it did not contain the `remaining_lease` feature, which we consider crucial for resale price prediction.

Key steps:
- Removed unnecessary columns (e.g. block number, full address)
- Ensured consistent data types
- Merged geospatial features from OneMap API and custom crawlers:
  - Distance to nearest MRT, hawker, school, park, hospital, etc.
  - Number of amenities within 1km

---

## 🛠 Feature Engineering

We engineered several key features:
- `floor_area_sqm` and `storey_range` (ordinal-encoded)
- Distance to key amenities using `scipy.spatial.cKDTree`
- Number of amenities within 1km (e.g. `num_mrts_within_1km`)
- Distance to Downtown Core (CBD) via centroid lookup (Raffles Place MRT as the centroid)

**Categorical features** like `town`, `flat_type`, `flat_model`, `storey_range` were encoded using `LabelEncoder`.

---

## 🤖 Modeling & Evaluation

### Models Evaluated:
- Linear Regression, Ridge, Lasso
- Random Forest
- Gradient Boosting, XGBoost, LightGBM, CatBoost

### Final Model: **Random Forest (tuned)**

After model comparison and hyperparameter tuning (via `RandomizedSearchCV`), the Random Forest model produced the best balance of accuracy and generalization.

### ✅ Final Evaluation:
- **MAE**: 18,983.65
- **RMSE**: 27,101.33
- **R²**: 0.9772
- **MAPE**: 3.81%

This means the average predicted resale price is within ±3.8% of actual transaction prices — a strong result given the diversity of flats and locations.

---

## 🧱 Project Structure 

```
SG Resale HDB Price Prediction/
├── data/ # Processed and raw datasets (external download)
│ ├── final_data/
│ └── raw_data/
├── model/ # Trained ML model + encoders
│ ├── best_rf_model.pkl
│ └── label_encoders.pkl
├── notebooks/ # Jupyter notebooks for EDA & modeling
├── pages/ # Streamlit multipage UI
│ ├── Predict Price.py
│ └── Future Affordability.py
├── helpers/ # Scripts (e.g. OneMap API helpers)
├── streamlit_app.py # Landing page
├── pyproject.toml # Project metadata and dependencies
├── requirements.txt # Installable dependencies
└── README.md # You're here!
```

---

## ⚙️ Setup Instructions

### 1. Create & activate a virtual environment (optional but recommended)

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

### 2. Install dependencies (using uv or pip)

#### With uv (preferred if installed)
uv pip install -r requirements.txt

#### Or with pip
pip install -r requirements.txt

### 3. Run the Streamlit app
streamlit run Home.py

---

## 🧠 Models & Techniques
- Random Forest Regressor (primary deployed model)
- Baseline models: Linear, Ridge, Lasso, Gradient Boosting, XGBoost, LightGBM, CatBoost
- Feature Engineering:
<ul>
    <li>Remaining lease</li>
    <li>Proximity to amenities (MRT, schools, parks, etc.)</li>
    <li>Encoded categorical features (LabelEncoder)</li>
    <li>Trained using 10 years of resale transaction data (from 2015-2025)</li>
</ul>

---

## 🛠 Tech Stack
- Python 3.13+
- Streamlit 1.47+
- scikit-learn
- pandas / numpy
- xgboost / lightgbm / catboost
- OneMap API
- cKDTree for fast geospatial distance calculations

---

## 🚧 Future Improvements

This project is already functional, but there are several meaningful enhancements we plan to explore:

### 🔍 1. Explainable AI (XAI)
- Integrate **SHAP** or **LIME** to visualize feature importance on a per-prediction basis
- Help users understand *why* a certain resale price is predicted

### 🌐 2. Interactive Map View
- Use **Folium**, **Pydeck**, or **Mapbox** to plot predicted flats on a map
- Allow spatial filtering (e.g. only show flats within 2km of a specific MRT)

### 💬 3. Natural Language Query Interface
- Let users ask questions like “What can I buy in Toa Payoh for under $600k in 2027?”
- Implement using **LLM + RAG** or Streamlit’s chatbot integrations

### 🗃 4. Model Version Comparison
- Allow toggling between different models (Linear, XGBoost, etc.)
- Display comparative performance or prediction differences

### 🧠 5. Retrainable Pipelines
- Add ability to retrain model with updated resale data
- Expose retraining as a CLI or admin-only page

### 📤 6. Price Sensitivity / Scenario Simulation
- Add a “what-if” mode to simulate impact of changing floor area, lease years, or location on predicted price

### 🌍 7. Multilingual Support
- Add Mandarin or Malay support to make the app more accessible to local users

### 🧱 8. Containerization & Deployment
- Add `Dockerfile` for reproducible builds
- Deploy on Streamlit Cloud, Render, or a custom VPS with CI/CD

---

## 🧾 License
MIT License

---

## 🙋‍♂️ Contributing
Pull requests and feature ideas welcome!
Please open an issue or submit a PR for improvements, enhancements, or bug fixes.

---

## 👤 Author
Built by Zheng Feng
Email: zhengfengchow@gmail.com
GitHub: @zeeeffchow

---
