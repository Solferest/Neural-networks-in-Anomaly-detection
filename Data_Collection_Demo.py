import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time

# --- Dashboard Title ---
st.title("🚨 Real-Time Anomaly Detection in Bearing system")


# --- Simulated Sensor Data ---
def generate_sensor_data():
    return {
        "temperature": np.random.normal(50, 5),
        "pressure": np.random.normal(100, 10),
        "vibration": np.random.normal(5, 1)
    }


# --- Collect Initial Data ---
data = pd.DataFrame([generate_sensor_data() for _ in range(100)])

# --- Data Preprocessing ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# --- Train Anomaly Detection Model (Isolation Forest) ---
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

# --- Real-Time Data Streaming ---
st.sidebar.header("Live Data Stream")
placeholder = st.empty()

while True:
    # Generate New Sensor Data
    new_data = generate_sensor_data()
    data.loc[len(data)] = new_data
    X_new_scaled = scaler.transform([list(new_data.values())])

    # Predict Anomaly (1 = Normal, -1 = Anomaly)
    anomaly = model.predict(X_new_scaled)[0]
    anomaly_label = "🟢 Normal" if anomaly == 1 else "🔴 Anomaly"

    # --- Display Updated Data ---
    with placeholder.container():
        st.subheader("📊 Sensor Readings")
        st.write(new_data)

        st.subheader("📉 Anomaly Status")
        st.write(f"**Status:** {anomaly_label}")

        fig = px.line(data[-50:], y=["temperature", "pressure", "vibration"], title="Live Sensor Data")
        st.plotly_chart(fig)

    time.sleep(2)  # Update every 2 seconds

