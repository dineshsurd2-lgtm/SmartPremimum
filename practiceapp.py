
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Premimum Loan",layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("/content/smartpremimum_pipeline.joblib")

final_pipeline = load_pipeline()

st.title("Smart Premium Loan")
st.write("Insurance Premimum Prediction System")
st.write(
    "Enter the customer,health,insurance and property"
    " information to get the insurance premimum"
)
st.header("Customer Information")
col1,col2 = st.columns(2)
with col1:
  age = st.number_input("Age",min_value=18,max_value=100,value=24,step=1)
  gender = st.selectbox("Gender",["Male","Female"])
  annual_income = st.number_input("Annual Income",min_value=0.0,max_value=150000.0,value=50000.0,step=1000.0)
  marital_status = st.selectbox("Marital Status",["Yes","No"])
with col2:
  number_of_dependents = st.number_input("Number of dependents",min_value=0.0,max_value=5.0,value=2.3,step=0.1)
  education_level = st.selectbox("Education Level",["High School","Bachelor's","Master's","PhD"])
  occupation = st.selectbox("Occupation",["Employed","Self-Employed","Unemployed"])
  location = st.selectbox("Location",["Urban","Suburban","Rural"])

st.header("Health & Lifestyle")
col1,col2 = st.columns(2)
with col1:
  health_score = st.number_input("Health score",min_value=0.0,max_value=60.0,value=10.0,step=0.1)
  previous_claims = st.number_input("Previous Claims",min_value=0.0,max_value=10.0,value=3.5,step=0.1)
  credict_score = st.number_input("Credict score",min_value=0.0,max_value=850.0,value=350.0,step=5.0)
with col2:
  smoking_status = st.selectbox("Smoking status",["Yes","No"])
  exercise_frequency = st.selectbox("Exercise Frequency",["Daily","Weekly","Monthly","Rarely"])
  customer_feedback = st.selectbox("Customer Feedback",["Good","Average","Poor"])

st.header("Insurance Information")
col1,col2 = st.columns(2)
with col1:
  policy_type = st.selectbox("Policy Type",["Basic","Comprehensive","Premium"])
  vehicle_age = st.number_input("Vehicle Age",min_value=0.0,max_value=50.0,value=5.0,step=1.0)
with col2:
  insurance_duration = st.number_input("Insurance Duration",min_value=0.0,max_value=50.0,value=5.0,step=1.0)
  property_type = st.selectbox("Property Type",["House","Apartment","Condo"])

if st.button("Predict Premium", type="primary", use_container_width=True):
    try:
        customer_data = pd.DataFrame({
            "Age": [float(age)],
            "Gender": [gender],
            "Annual Income": [float(annual_income)],
            "Marital Status": [marital_status],
            "Number of Dependents": [float(number_of_dependents)],
            "Education Level": [education_level],
            "Occupation": [occupation],
            "Location": [location],
            "Health Score": [float(health_score)],
            "Policy Type": [policy_type],
            "Previous Claims": [float(previous_claims)],
            "Vehicle Age": [float(vehicle_age)],
            "Credit Score": [float(credict_score)],
            "Insurance Duration": [float(insurance_duration)],
            "Customer Feedback": [customer_feedback],
            "Smoking Status": [smoking_status],
            "Exercise Frequency": [exercise_frequency],
            "Property Type": [property_type]
        })

        prediction = final_pipeline.predict(customer_data)
        premium = max(0, float(prediction[0]))

        st.success("Premium prediction generated successfully")
        st.subheader(f"Estimated Insurance Premium: ₹{premium:,.2f}")

    except Exception as e:
        st.error("An error occurred while generating the prediction.")
        st.exception(e)
