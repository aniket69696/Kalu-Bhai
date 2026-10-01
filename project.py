import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Crop Mantra",
    page_icon="🌱",
    layout="wide"
)


st.title("🌱 Crop Mantra")

st.subheader("Smart Crop Recommendation System")

st.write(
    "Enter the soil and environmental conditions "
    "to get the recommended crop."
)



df = pd.read_csv("plant.csv")

numeric_columns = ["N","P","K","temperature","humidity", "ph","rainfall"]

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].mean()
    )


df["label"] = df["label"].fillna(
    df["label"].mode()[0]
)


X = df[["N","P","K","temperature", "humidity","ph","rainfall"]]

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=10
)

model = RandomForestClassifier(
    n_estimators=500,
    criterion="entropy",
    random_state=10
)

model.fit(X_train, y_train)
st.divider()

st.header("🌾 Enter Crop Conditions")


col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("🧪 Soil Nutrients")

    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=200.0,
        value=90.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        max_value=150.0,
        value=42.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        max_value=200.0,
        value=43.0
    )

with col2:

    st.subheader("🌡️ Climate")

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

with col3:

    st.subheader("💧 Other Conditions")

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=200.0
    )


st.divider()

if st.button(
    "🌱 Recommend Crop",
    use_container_width=True
):
    input_data = pd.DataFrame(
        [[ N,P,K,temperature,humidity,ph,rainfall ]],
        columns=[ "N","P","K","temperature","humidity","ph","rainfall"])

    prediction = model.predict(input_data)[0]
    st.success("Prediction completed!")

    st.markdown(
        f"""
        ## 🌾 Recommended Crop

        # **{str(prediction).upper()}**
        """
    )