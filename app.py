import streamlit as st


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)
import pandas as pd

data = pd.read_csv("Telco-Customer-Churn.csv")

data = data.drop("customerID", axis=1)

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].mean()
)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])
    from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
st.title("📊 Customer Churn Prediction System")
st.markdown("### Enter Customer Details")

col1, col2 = st.columns(2)



with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])

with col2:
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device = st.selectbox("Device Protection", ["No", "Yes"])
    tech = st.selectbox("Tech Support", ["No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes"])

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paper = st.selectbox(
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
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

if st.button("🔍 Predict Churn", use_container_width=True):

    # Convert UI values to model values
    gender = 1 if gender == "Male" else 0
    senior = 1 if senior == "Yes" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0
    phone = 1 if phone == "Yes" else 0
    multiple = 1 if multiple == "Yes" else 0

    internet = {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    }[internet]

    online_security = 1 if online_security == "Yes" else 0
    online_backup = 1 if online_backup == "Yes" else 0
    device = 1 if device == "Yes" else 0
    tech = 1 if tech == "Yes" else 0
    tv = 1 if tv == "Yes" else 0
    movies = 1 if movies == "Yes" else 0

    contract = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }[contract]

    paper = 1 if paper == "Yes" else 0

    payment = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }[payment]

    customer = [[
        gender,
        senior,
        partner,
        dependents,
        tenure,
        phone,
        multiple,
        internet,
        online_security,
        online_backup,
        device,
        tech,
        tv,
        movies,
        contract,
        paper,
        payment,
        monthly,
        total
    ]]

    prediction = model.predict(customer)[0]

    if prediction == 1:
        st.error("❌ Customer Will Churn")
    else:
        st.success("✅ Customer Will Stay")