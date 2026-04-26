import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("drift_results.csv")

# =========================
# TITLE
# =========================
st.title("✈️ Turbofan Engine RUL Monitoring Dashboard")

st.markdown("### AI Model Drift Monitoring using LSTM Activations")

# =========================
# TILE 1 — TEMPORAL GRAPH
# =========================
st.subheader("📈 Drift Over Time")

fig, ax = plt.subplots()
ax.plot(df["batch"], df["drift_score"], marker='o')
ax.set_xlabel("Batch (Time)")
ax.set_ylabel("Drift Score")
ax.set_title("Drift Progression Over Time")

st.pyplot(fig)

# =========================
# TILE 2 — CATEGORICAL
# =========================
st.subheader("📊 Drift Level Distribution")

def classify(score):
    if score < 0.1:
        return "Low"
    elif score < 0.3:
        return "Medium"
    else:
        return "High"

df["drift_level"] = df["drift_score"].apply(classify)

counts = df["drift_level"].value_counts()

st.bar_chart(counts)

# =========================
# METRIC TILE
# =========================
st.subheader("⚠️ Current Drift Status")

latest_score = df["drift_score"].iloc[-1]

if latest_score < 0.1:
    level = "Low"
elif latest_score < 0.3:
    level = "Medium"
else:
    level = "High"

st.metric(label="Latest Drift Score", value=f"{latest_score:.3f}", delta=level)

# =========================
# EXPLANATION
# =========================
st.markdown("""
### 📌 Insights
- Drift increases over time due to simulated operational changes
- High drift indicates potential model degradation
- Monitoring activations helps detect issues without labels
""")