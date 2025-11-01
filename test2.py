import requests
import pandas as pd

# API key and URLs
api_key = "579b464db66ec23bdd0000014217397ea1e84a266eb6381bb712d5f2"

crop_url = "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de?api-key=579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b&format=json&limit=100000"
rain_url = f"https://api.data.gov.in/resource/8e0bd482-4aba-4d99-9cb9-ff124f6f1c2f?api-key={api_key}&format=json&limit=100000"

# Fetch Crop data
crop_res = requests.get(crop_url)
crop_data = crop_res.json()
df_crop = pd.DataFrame(crop_data['records'])
print("✅ Crop data loaded:", df_crop.shape)

# Fetch Rainfall data
rain_res = requests.get(rain_url)
rain_data = rain_res.json()
df_rain = pd.DataFrame(rain_data['records'])
print("✅ Rainfall data loaded:", df_rain.shape)

# -------------------------------
# DESCRIBE BOTH DATASETS
# -------------------------------

print("\n📊 --- CROP DATASET INFO ---")
print("Columns:", df_crop.columns.tolist())
print("\nData Types:")
print(df_crop.dtypes)
print("\nMissing Values:")
print(df_crop.isnull().sum())
print("\nStatistical Summary (Numeric Columns):")
print(df_crop.describe())

print("\n🔹 First 5 Rows of Crop Data:")
print(df_crop.head())

print("\n🌧 --- RAINFALL DATASET INFO ---")
print("Columns:", df_rain.columns.tolist())
print("\nData Types:")
print(df_rain.dtypes)
print("\nMissing Values:")
print(df_rain.isnull().sum())
print("\nStatistical Summary (Numeric Columns):")
print(df_rain.describe())

print("\n🔹 First 5 Rows of Rainfall Data:")
print(df_rain.head())
