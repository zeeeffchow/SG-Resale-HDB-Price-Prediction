import requests
import pandas as pd

def fetch_amenity_coordinates(queryname: str, token: str) -> pd.DataFrame:
    url = f"https://www.onemap.gov.sg/api/public/themesvc/retrieveTheme?queryName={queryname}"
    headers = {"Authorization": token}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    records = []
    for item in data.get("SrchResults", []):
        try:
            lat = float(item["LATITUDE"])
            lon = float(item["LONGITUDE"])
            name = item.get("NAME", queryname)
            records.append({"name": name, "latitude": lat, "longitude": lon})
        except (KeyError, ValueError):
            continue

    return pd.DataFrame(records)
