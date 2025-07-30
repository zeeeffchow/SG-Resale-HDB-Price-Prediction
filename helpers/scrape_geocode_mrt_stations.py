import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv

# --- Step 1: Load API token ---
load_dotenv()
ONEMAP_API_TOKEN = os.getenv("ONEMAP_API_TOKEN")

# --- Step 2: Load MRT station file ---
df = pd.read_csv("./data/raw_data/mrt code and station names.csv", encoding="ISO-8859-1")

# Normalize column names and clean values
df.columns = df.columns.str.strip()
df['Code'] = df['Code'].astype(str).str.strip()
df['Station'] = df['Station'].astype(str).str.strip()

# Split codes with spaces into multiple rows
df['Code'] = df['Code'].str.split()
df = df.explode('Code').reset_index(drop=True)

# Create search query for OneMap
df['search_query'] = df['Station'] + " MRT Station (" + df['Code'] + ")"

# --- Step 3: Geocode function ---
def geocode_onemap(search_val):
    url = "https://www.onemap.gov.sg/api/common/elastic/search"
    headers = {
        "Authorization": f"Bearer {ONEMAP_API_TOKEN}"
    }
    params = {
        "searchVal": search_val,
        "returnGeom": "Y",
        "getAddrDetails": "Y",
        "pageNum": 1
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        time.sleep(0.5)  # rate limiting
        data = response.json().get("results", [])
        if data:
            return float(data[0]['LATITUDE']), float(data[0]['LONGITUDE'])
    except Exception as e:
        print(f"Error geocoding {search_val}: {e}")
    return None, None

# --- Step 4: Apply geocoding ---
coords = []
for query in df['search_query']:
    lat, lon = geocode_onemap(query)
    coords.append((lat, lon))

df[['latitude', 'longitude']] = pd.DataFrame(coords, index=df.index)

# --- Step 5: Save result ---
df.to_csv("geocoded_mrt_stations.csv", index=False)
print("✅ Saved geocoded MRT station list to geocoded_mrt_stations.csv")
