import requests
import pandas as pd

api_key = "579b464db66ec23bdd0000014217397ea1e84a266eb6381bb712d5f2"

crop_url = "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de?api-key=579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b&format=json&limit=100000"

crop_res = requests.get(crop_url)
crop_data = crop_res.json()
df_crop = pd.DataFrame(crop_data['records'])
print("✅ Crop data loaded:", df_crop.shape)


rain_url = f"https://api.data.gov.in/resource/8e0bd482-4aba-4d99-9cb9-ff124f6f1c2f?api-key={api_key}&format=json&limit=100000"

rain_res = requests.get(rain_url)
rain_data = rain_res.json()
df_rain = pd.DataFrame(rain_data['records'])
print("✅ Rainfall data loaded:", df_rain.shape)

# print("\nCrop Columns:\n", df_crop.columns)
# print("\nRainfall Columns:\n", df_rain.columns)

df_crop.rename(columns={
    'state_name': 'state',
    'district_name': 'district',
    'crop_year': 'year'
}, inplace=True)

df_rain.rename(columns={
    'subdivision': 'state',  # if rainfall dataset has sub_division instead of state
    'annual': 'rainfall'  # rename for clarity
}, inplace=True)

# Convert both 'year' columns to integer
df_crop['year'] = df_crop['year'].astype(int)
df_rain['year'] = df_rain['year'].astype(int)

df_crop['state'] = df_crop['state'].str.lower().str.replace('&', 'and').str.strip()
df_rain['state'] = df_rain['state'].str.lower().str.replace('&', 'and').str.strip()


merged = pd.merge(df_crop, df_rain, on=['state', 'year'], how='inner')
print("\n✅ Merged Data Shape:", merged.shape)
# print(merged.head())

# merged.to_csv("combined_agri_climate_data.csv", index=False)
# print("📁 Combined dataset saved as combined_agri_climate_data.csv")

# --- DATA CLEANING ---
import numpy as np

# Remove missing or invalid values
merged.replace(['', 'NA', None], np.nan, inplace=True)
merged.dropna(subset=['rainfall', 'production_', 'area_'], inplace=True)

# Convert numeric columns to float
merged['rainfall'] = merged['rainfall'].astype(float)
merged['production_'] = merged['production_'].astype(float)
merged['area_'] = merged['area_'].astype(float)

# Calculate yield (production per area)
merged['yield'] = merged['production_'] / merged['area_']

print("✅ Cleaned data and calculated yield.")
print(merged[['state', 'year', 'rainfall', 'production_', 'area_', 'yield']].head())


correlation = merged[['rainfall', 'yield']].corr().iloc[0, 1]
print(f"🌦️ Correlation between rainfall and crop yield: {correlation:.2f}")

summary = merged.groupby(['state', 'crop']).agg({
    'rainfall': 'mean',
    'yield': 'mean'
}).reset_index()

print(summary.head())

# ----------------------
def answer_question(question, data):
    question = question.lower()

    if "average rainfall" in question:
        state = question.split("in")[-1].strip().capitalize()
        result = data[data['state'].str.contains(state, case=False)]
        if result.empty:
            return f"❌ No data found for {state}."
        avg_rain = result['rainfall'].mean()
        return f"🌧️ The average rainfall in {state} is {avg_rain:.2f} mm (based on available data)."

    elif "yield" in question or "production" in question:
        crop = question.split("of")[-1].strip().capitalize()
        result = data[data['crop'].str.contains(crop, case=False)]
        if result.empty:
            return f"❌ No data found for crop {crop}."
        avg_yield = result['yield'].mean()
        return f"🌾 The average yield for {crop} is {avg_yield:.2f} units (production per area)."

    else:
        return "🤖 I can answer questions about rainfall or crop yield. Try: 'Average rainfall in Maharashtra' or 'Yield of Rice'."

while True:
    q = input("\nAsk your question (or type 'exit' to quit): ")
    if q.lower() == "exit":
        break
    print(answer_question(q, merged))
