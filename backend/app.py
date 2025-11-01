import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# FETCH DATA
# -----------------------------
@st.cache_data
def load_data():
    api_key = "579b464db66ec23bdd0000014217397ea1e84a266eb6381bb712d5f2"

    crop_url = "https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de?api-key=579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b&format=json&limit=10000000"
    rain_url = f"https://api.data.gov.in/resource/8e0bd482-4aba-4d99-9cb9-ff124f6f1c2f?api-key={api_key}&format=json&limit=10000000"

    df_crop = pd.DataFrame(requests.get(crop_url).json()['records'])
    df_rain = pd.DataFrame(requests.get(rain_url).json()['records'])

    # Rename & clean
    df_crop.rename(columns={'state_name': 'state', 'district_name': 'district', 'crop_year': 'year'}, inplace=True)
    df_rain.rename(columns={'subdivision': 'state', 'annual': 'rainfall'}, inplace=True)

    df_crop['year'] = df_crop['year'].astype(int)
    df_rain['year'] = df_rain['year'].astype(int)

    df_crop['state'] = df_crop['state'].str.lower().str.replace('&', 'and').str.strip()
    df_rain['state'] = df_rain['state'].str.lower().str.replace('&', 'and').str.strip()

    merged = pd.merge(df_crop, df_rain, on=['state', 'year'], how='inner')
    merged.replace(['', 'NA', None], np.nan, inplace=True)
    merged.dropna(subset=['rainfall', 'production_', 'area_'], inplace=True)

    merged['rainfall'] = merged['rainfall'].astype(float)
    merged['production_'] = merged['production_'].astype(float)
    merged['area_'] = merged['area_'].astype(float)
    merged['yield'] = merged['production_'] / merged['area_']

    return merged


data = load_data()

data = data[(data['rainfall'] > 0) & (data['rainfall'] < 4000)]
data = data[(data['area_'] > 0) & (data['production_'] > 0)]
data = data[data['yield'] < data['yield'].quantile(0.95)]  # remove extreme outliers
data['yield'] = np.log1p(data['yield'])
data['rainfall'] = np.log1p(data['rainfall'])
data['area_'] = np.log1p(data['area_'])


# -----------------------------
# TRAIN MODEL
# -----------------------------
X = data[['rainfall', 'area_', 'production_']]
y = data['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Smart Agri Analyzer", layout="wide")
st.title("🌾 Smart Agriculture Data & Yield Predictor")

st.sidebar.header("Navigation")
menu = st.sidebar.radio("Select View", ["📊 Dashboard", "💬 Ask AI", "🌦️ Predict Yield"])

# -----------------------------
# 1️⃣ DASHBOARD
# -----------------------------
if menu == "📊 Dashboard":
    st.subheader("Agriculture & Rainfall Data Overview")
    st.write(f"✅ Total Records: {data.shape[0]}")
    st.dataframe(data.head(10))

    st.metric("Model Accuracy (R²)", f"{r2:.2f}")
    st.bar_chart(data.groupby('state')['yield'].mean().sort_values(ascending=False).head(10))

# -----------------------------
# 2️⃣ Q&A SECTION
# -----------------------------
elif menu == "💬 Ask AI":
    st.subheader("Ask about rainfall or crop yield")
    q = st.text_input("Type your question:", "")

    def answer_question(question):
        question = question.lower()
        if "average rainfall" in question:
            try:
                state = question.split("in")[-1].strip()
                result = data[data['state'].str.contains(state, case=False)]
                if result.empty:
                    return f"❌ No data found for {state}."
                avg_rain = result['rainfall'].mean()
                return f"🌧️ Average rainfall in {state.capitalize()} is {avg_rain:.2f} mm."
            except:
                return "⚠️ Please specify the state name."

        elif "yield of" in question:
            crop = question.split("of")[-1].strip()
            result = data[data['crop'].str.contains(crop, case=False)]
            if result.empty:
                return f"❌ No data found for crop {crop}."
            avg_yield = result['yield'].mean()
            return f"🌾 Average yield for {crop.capitalize()} is {avg_yield:.2f} units."

        else:
            return "🤖 Try: 'Average rainfall in Maharashtra' or 'Yield of Rice'."

    if st.button("Get Answer"):
        st.success(answer_question(q))

# -----------------------------
# 3️⃣ YIELD PREDICTION
# -----------------------------
elif menu == "🌦️ Predict Yield":
    st.subheader("Predict Crop Yield Based on Rainfall and Area")

    rainfall = st.number_input("Enter Rainfall (mm):", min_value=0.0, step=0.1)
    area = st.number_input("Enter Crop Area:", min_value=0.0, step=0.1)

    if st.button("Predict Yield"):
        pred = model.predict([[rainfall, area]])[0]
        st.success(f"🌾 Predicted Yield: {pred:.2f} units")

    st.caption("Model trained on real Govt. of India crop and rainfall data.")
