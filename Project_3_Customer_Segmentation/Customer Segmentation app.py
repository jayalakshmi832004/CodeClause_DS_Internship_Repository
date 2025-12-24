import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation Tool",
    page_icon="👥",
    layout="centered"
)

st.markdown("""
<style>
/* Remove Streamlit header & top black bar */
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}

/* Remove extra top padding */
.block-container {
    padding-top: 1rem;
}

/* Main card */
.main-container {
    background: rgba(0, 0, 0, 0.75);
    padding: 30px;
    border-radius: 16px;
}

/* Input styling */
input, .stSlider {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

data = load_data()


X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

@st.cache_resource
def train_model(X_scaled):
    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    model.fit(X_scaled)
    return model

model = train_model(X_scaled)


st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>👥 Customer Segmentation Tool</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Segment customers based on behavior using K-Means</p>",
    unsafe_allow_html=True
)

st.markdown("### 🧾 Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
income = st.number_input("Annual Income (k$)", min_value=10, max_value=150, value=60)
score = st.slider("Spending Score (1–100)", 1, 100, 50)

if st.button("🔍 Segment Customer"):
    user_data = [[age, income, score]]
    user_scaled = scaler.transform(user_data)
    segment = model.predict(user_scaled)[0]

    st.markdown("---")
    st.success(f"🎯 **Customer belongs to Segment {segment}**")

    if segment == 0:
        st.info("🟢 High Income – High Spending Customer")
    elif segment == 1:
        st.info("🟡 Average Income – Average Spending Customer")
    elif segment == 2:
        st.info("🔵 High Income – Low Spending Customer")
    elif segment == 3:
        st.info("🟠 Low Income – High Spending Customer")
    else:
        st.info("🔴 Low Income – Low Spending Customer")

st.markdown("</div>", unsafe_allow_html=True)
