import os
print("RUNNING FILE:", os.path.abspath(__file__))

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler



# ---------------------------------------------------
# Load model + threshold + engagement bins
# ---------------------------------------------------
pipeline = joblib.load("churn_model.pkl")
threshold = joblib.load("churn_threshold.pkl")
engagement_bins = joblib.load("engagement_bins.pkl")

preprocess = pipeline.named_steps['preprocess']
model = pipeline.named_steps['model']


# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:

    # Convert percentage inputs (0–100) back to 0–1 for model
    percent_cols = [
        "Email_Open_Rate",
        "Cart_Abandonment_Rate",
        "Discount_Usage_Rate",
        "Returns_Rate"
    ]
    for col in percent_cols:
        df[col] = df[col] / 100.0

    engagement_features = [
        'Login_Frequency',
        'Session_Duration_Avg',
        'Pages_Per_Session',
        'Mobile_App_Usage'
    ]

    scaler = StandardScaler()
    scaled_engagement = scaler.fit_transform(df[engagement_features])
    df['Engagement_Score'] = scaled_engagement.mean(axis=1)

    df['Inactivity_Score'] = (
        df['Days_Since_Last_Purchase'] - df['Engagement_Score']
    )

    df['Engagement_Level'] = pd.cut(
        df['Engagement_Score'],
        bins=engagement_bins,
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    return df


# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
def predict_churn(pipeline, X: pd.DataFrame, threshold: float):
    proba = pipeline.predict_proba(X)[:, 1][0]
    pred = int(proba >= threshold)
    return proba, pred


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# -----------------------------
# Sidebar Section
# -----------------------------
st.sidebar.title("Customer Churn Prediction")

st.sidebar.info("""
This application predicts customer churn risk
using a trained Machine Learning model.

**Steps:**
1. Enter customer information  
2. Click **Predict Churn**  
3. Review risk level and recommendations  
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 About This App")
st.sidebar.write("Built by Aneesh using Machine Learning & Streamlit.")


# -----------------------------
# Main Title
# -----------------------------
st.title("🔮 Customer Churn Prediction Dashboard")
st.write("Enter customer details below to predict churn probability.")

# ---------------------------------------------------
# COUNTRY → CITY MAPPING
# ---------------------------------------------------
country_city_map = {
    "France": ["Paris", "Marseille", "Nice", "Lyon", "Toulouse"],
    "UK": ["London", "Manchester", "Glasgow", "Leeds", "Birmingham"],
    "Canada": ["Toronto", "Vancouver", "Calgary", "Ottawa", "Montreal"],
    "USA": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    "India": ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai"],
    "Japan": ["Tokyo", "Osaka", "Yokohama", "Nagoya", "Kyoto"],
    "Germany": ["Berlin", "Munich", "Frankfurt", "Cologne", "Hamburg"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Adelaide", "Perth"]
}

# ---------------------------------------------------
# CUSTOMER PROFILE
# ---------------------------------------------------
st.header("👤 Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 18, 100, 30, step=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    Country = st.selectbox("Country", list(country_city_map.keys()))

with col3:
    City = st.selectbox("City", country_city_map[Country])

# ---------------------------------------------------
# ACCOUNT & MEMBERSHIP
# ---------------------------------------------------
st.header("💳 Account & Membership")
col4, col5, col6 = st.columns(3)

with col4:
    Membership_Years = st.number_input("Membership Years (e.g., 3.1)", 0.0, 20.0, 3.0, step=0.1)
    Lifetime_Value = st.number_input("Lifetime Value", 0.0, 100000.0, 5000.0)

with col5:
    Credit_Balance = st.number_input("Credit Balance", 0.0, 50000.0, 1000.0)
    Payment_Method_Diversity = st.selectbox("Payment Method Diversity", [1, 2, 3, 4, 5])

with col6:
    Signup_Quarter = st.selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])

# ---------------------------------------------------
# ENGAGEMENT BEHAVIOR
# ---------------------------------------------------
st.header("📱 Engagement Behavior")
col7, col8, col9 = st.columns(3)

with col7:
    Login_Frequency = st.number_input("Login Frequency", 0, 100, 10, step=1)
    Session_Duration_Avg = st.number_input("Avg Session Duration (mins)", 0.0, 60.0, 5.0)

with col8:
    Pages_Per_Session = st.number_input("Pages Per Session", 1, 50, 8, step=1)
    Mobile_App_Usage = st.selectbox("Mobile App Usage", [0, 1])

with col9:
    Email_Open_Rate = st.number_input("Email Open Rate (%)", 0.00, 100.00, 50.00, step=0.10)
    Social_Media_Engagement_Score = st.number_input("Social Media Engagement Score", 0, 100, 20, step=1)

# ---------------------------------------------------
# SHOPPING BEHAVIOR
# ---------------------------------------------------
st.header("🛒 Shopping & Purchase Behavior")
col10, col11, col12 = st.columns(3)

with col10:
    Total_Purchases = st.number_input("Total Purchases", 0, 500, 10, step=1)
    Average_Order_Value = st.number_input("Average Order Value", 0.0, 2000.0, 50.0)

with col11:
    Wishlist_Items = st.number_input("Wishlist Items", 0, 100, 5, step=1)
    Cart_Abandonment_Rate = st.number_input("Cart Abandonment Rate (%)", 0.00, 100.00, 20.00, step=0.10)

with col12:
    Discount_Usage_Rate = st.number_input("Discount Usage Rate (%)", 0.00, 100.00, 30.00, step=0.10)
    Returns_Rate = st.number_input("Returns Rate (%)", 0.00, 100.00, 10.00, step=0.10)
    Product_Reviews_Written = st.number_input("Product Reviews Written", 0, 50, 2, step=1)

# ---------------------------------------------------
# INACTIVITY & SUPPORT
# ---------------------------------------------------
st.header("⏳ Inactivity & Support")
col13, col14 = st.columns(2)

with col13:
    Days_Since_Last_Purchase = st.number_input("Days Since Last Purchase", 0, 365, 30, step=1)

with col14:
    Customer_Service_Calls = st.number_input("Customer Service Calls", 0, 50, 1, step=1)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("Predict Churn"):

    input_data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "Country": Country,
        "City": City,
        "Membership_Years": Membership_Years,
        "Login_Frequency": Login_Frequency,
        "Session_Duration_Avg": Session_Duration_Avg,
        "Pages_Per_Session": Pages_Per_Session,
        "Cart_Abandonment_Rate": Cart_Abandonment_Rate,
        "Wishlist_Items": Wishlist_Items,
        "Total_Purchases": Total_Purchases,
        "Average_Order_Value": Average_Order_Value,
        "Days_Since_Last_Purchase": Days_Since_Last_Purchase,
        "Discount_Usage_Rate": Discount_Usage_Rate,
        "Returns_Rate": Returns_Rate,
        "Email_Open_Rate": Email_Open_Rate,
        "Customer_Service_Calls": Customer_Service_Calls,
        "Product_Reviews_Written": Product_Reviews_Written,
        "Social_Media_Engagement_Score": Social_Media_Engagement_Score,
        "Mobile_App_Usage": Mobile_App_Usage,
        "Payment_Method_Diversity": Payment_Method_Diversity,
        "Lifetime_Value": Lifetime_Value,
        "Credit_Balance": Credit_Balance,
        "Signup_Quarter": Signup_Quarter
    }])

    input_data = compute_engineered_features(input_data)
    proba, pred = predict_churn(pipeline, input_data, threshold)

    # BEAUTIFUL RESULT CARD
    st.markdown("""
        <style>
        .result-card {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            border-left: 6px solid #6f42c1;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .prob {
            font-size: 32px;
            font-weight: bold;
            color: #6f42c1;
        }
        .pred-good {
            font-size: 26px;
            font-weight: bold;
            color: green;
        }
        .pred-bad {
            font-size: 26px;
            font-weight: bold;
            color: red;
        }
        </style>
    """, unsafe_allow_html=True)

    churn_text = "Customer Will Churn" if pred == 1 else "Customer Will Not Churn"
    churn_class = "pred-bad" if pred == 1 else "pred-good"

    st.markdown(f"""
        <div class="result-card">
            <div class="prob">Churn Probability: {proba:.4f}</div>
            <div class="{churn_class}">{churn_text}</div>
        </div>
    """, unsafe_allow_html=True)
