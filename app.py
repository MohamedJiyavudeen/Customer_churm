import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:white;
}

.subtitle{
text-align:center;
color:#dbeafe;
font-size:18px;
margin-bottom:25px;
}

.stButton>button{
width:100%;
height:50px;
font-size:18px;
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"<h1 class='title'>📊 Customer Churn Prediction</h1>",
unsafe_allow_html=True
)

st.markdown(
"<p class='subtitle'>Machine Learning using Logistic Regression</p>",
unsafe_allow_html=True
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Menu",
    [
        "Prediction",
        "Dataset",
        "About"
    ]
)
# ==========================
# PART 2 - DATASET & MODEL
# ==========================

@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")

    df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].mean()
    )

    encoder = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    return df


data = load_data()


X = data.drop("Churn", axis=1)
y = data["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


model = LogisticRegression(max_iter=3000)

model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)


# ==========================
# DATASET PAGE
# ==========================

if page == "Dataset":

    st.title("📊 Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", data.shape[0])

    c2.metric("Columns", data.shape[1])

    c3.metric("Accuracy", f"{accuracy*100:.2f}%")

    st.divider()

    st.dataframe(data.head(10), use_container_width=True)
    # ==========================
# PART 3 - PREDICTION PAGE
# ==========================

if page == "Prediction":

    st.title("🔍 Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox("Gender", ["Male", "Female"])

        senior = st.selectbox("Senior Citizen", ["No", "Yes"])

        partner = st.selectbox("Partner", ["No", "Yes"])

        dependents = st.selectbox("Dependents", ["No", "Yes"])

        tenure = st.slider("Tenure (Months)", 0, 72, 24)

        phone = st.selectbox("Phone Service", ["No", "Yes"])

        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

    with col2:

        backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            0.0,
            200.0,
            70.0
        )

        total = st.number_input(
            "Total Charges",
            0.0,
            10000.0,
            1500.0
        )

    st.divider()

    predict = st.button(
        "🚀 Predict Customer Churn",
        use_container_width=True
    )
    # ==========================
# PART 4 - PREDICTION LOGIC
# ==========================

if page == "Prediction" and predict:

    with st.spinner("Predicting customer churn..."):
        time.sleep(1.5)

    customer = pd.DataFrame([{
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone == "Yes" else 0,

        "MultipleLines":
            2 if multiple == "Yes"
            else 1 if multiple == "No phone service"
            else 0,

        "InternetService":
            1 if internet == "Fiber optic"
            else 0 if internet == "DSL"
            else 2,

        "OnlineSecurity":
            2 if security == "Yes"
            else 1 if security == "No internet service"
            else 0,

        "OnlineBackup":
            2 if backup == "Yes"
            else 1 if backup == "No internet service"
            else 0,

        "DeviceProtection":
            2 if device == "Yes"
            else 1 if device == "No internet service"
            else 0,

        "TechSupport":
            2 if support == "Yes"
            else 1 if support == "No internet service"
            else 0,

        "StreamingTV":
            2 if tv == "Yes"
            else 1 if tv == "No internet service"
            else 0,

        "StreamingMovies":
            2 if movies == "Yes"
            else 1 if movies == "No internet service"
            else 0,

        "Contract":
            0 if contract == "Month-to-month"
            else 1 if contract == "One year"
            else 2,

        "PaperlessBilling":
            1 if paperless == "Yes" else 0,

        "PaymentMethod":
            {
                "Bank transfer (automatic)": 0,
                "Credit card (automatic)": 1,
                "Electronic check": 2,
                "Mailed check": 3
            }[payment],

        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    st.progress(int(probability * 100))

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("❌ Customer Will Churn")
    else:
        st.success("✅ Customer Will Stay")

    st.balloons()
    # ==========================
# PART 5 - ANALYTICS & DASHBOARD
# ==========================

elif page == "About":

    st.title("📖 About Project")

    st.markdown("""
### 📊 Customer Churn Prediction

This project predicts whether a customer will leave the telecom company.

### 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Logistic Regression

### 📂 Dataset

- Telco Customer Churn Dataset

### 🎯 Features

✅ Beautiful UI

✅ Machine Learning Prediction

✅ Probability Score

✅ Dataset Analytics

✅ Responsive Dashboard
""")

# ---------- Dashboard ----------

st.sidebar.markdown("---")

st.sidebar.metric("Rows", data.shape[0])

st.sidebar.metric("Columns", data.shape[1])

st.sidebar.metric(
    "Accuracy",
    f"{accuracy*100:.2f}%"
)

# ---------- Home Footer ----------

st.markdown("---")

c1, c2, c3 = st.columns(3)

with c1:
    st.info("📊 Dataset Ready")

with c2:
    st.success("🤖 ML Model Loaded")

with c3:
    st.warning("🚀 Streamlit Dashboard")

st.markdown(
"""
<center>

Made with ❤️ using Python & Streamlit

</center>
""",
unsafe_allow_html=True
)
# ==========================
# PART 6 - CHARTS & STYLING
# ==========================

import plotly.express as px

if page == "Dataset":

    st.subheader("📊 Churn Distribution")

    churn_count = data["Churn"].value_counts().reset_index()
    churn_count.columns = ["Churn", "Count"]

    fig = px.pie(
        churn_count,
        values="Count",
        names="Churn",
        hole=0.5,
        title="Customer Churn Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💰 Monthly Charges")

    fig2 = px.histogram(
        data,
        x="MonthlyCharges",
        nbins=30,
        title="Monthly Charges Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📅 Customer Tenure")

    fig3 = px.histogram(
        data,
        x="tenure",
        nbins=25,
        title="Customer Tenure"
    )

    st.plotly_chart(fig3, use_container_width=True)