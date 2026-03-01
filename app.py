import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 CreditWise Loan Approval System")
st.markdown("### AI-Powered Intelligent Loan Decision System")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open('loan_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Predict Loan"])

# Home Page
if page == "🏠 Home":
    st.header("Welcome to CreditWise Loan Approval System")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "87.5%")
    with col2:
        st.metric("Precision", "79.0%")
    with col3:
        st.metric("F1-Score", "79.7%")
    
    st.markdown("---")
    st.subheader("🎯 Problem Statement")
    st.write("""
    SecureTrust Bank faces challenges with their manual loan approval process:
    - ⏱️ Time-consuming verification
    - 🤷 Biased & inconsistent decisions
    - 💰 Good customers rejected, risky ones approved
    """)
    
    st.subheader("💡 Solution")
    st.write("""
    Our ML-powered system provides:
    -  Fast, automated screening
    -  Unbiased, data-driven decisions
    -  87.5% prediction accuracy
    """)

# Predict Page
elif page == "📊 Predict Loan":
    st.header("Loan Application Form")
    st.write("Fill in the applicant details to get instant loan prediction")
    
    # Create form
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Married", "Single"])
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            education_level = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Undergraduate"])
        
        with col2:
            st.subheader("Employment & Income")
            employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Business"])
            employer_category = st.selectbox("Employer Category", ["Government", "Private", "MNC", "Self", "Unemployed"])
            applicant_income = st.number_input("Monthly Income (₹)", min_value=0, max_value=1000000, value=50000)
            coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, max_value=1000000, value=0)
            savings = st.number_input("Savings Balance (₹)", min_value=0, max_value=10000000, value=100000)
        
        with col3:
            st.subheader("Loan Details")
            loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, max_value=100000000, value=1000000)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 360])
            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            collateral_value = st.number_input("Collateral Value (₹)", min_value=0, max_value=100000000, value=500000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=0)
            dti_ratio = st.slider("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        
        submitted = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)
    
    if submitted:
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Create input with same encoding as training
        input_dict = {
            'Applicant_Income': applicant_income,
            'Coapplicant_Income': coapplicant_income,
            'Age': age,
            'Dependents': dependents,
            'Existing_Loans': existing_loans,
            'Savings': savings,
            'Collateral_Value': collateral_value,
            'Loan_Amount': loan_amount,
            'Loan_Term': loan_term,
            'DTI_Ratio': dti_ratio,
            'Credit_Score': credit_score,
            'DTI_Ratio_sq': dti_ratio ** 2,
            'Credit_Score_sq': credit_score ** 2
        }
        
        # Add categorical encodings
        input_dict['Employment_Status_Self-employed'] = 1 if employment_status == "Self-Employed" else 0
        input_dict['Employment_Status_Unemployed'] = 1 if employment_status == "Business" else 0
        input_dict['Marital_Status_Single'] = 1 if marital_status == "Single" else 0
        input_dict['Loan_Purpose_Education'] = 1 if loan_purpose == "Education" else 0
        input_dict['Loan_Purpose_Home'] = 1 if loan_purpose == "Home" else 0
        input_dict['Loan_Purpose_Personal'] = 1 if loan_purpose == "Personal" else 0
        input_dict['Property_Area_Semiurban'] = 1 if property_area == "Semiurban" else 0
        input_dict['Property_Area_Urban'] = 1 if property_area == "Urban" else 0
        input_dict['Education_Level_Not Graduate'] = 1 if education_level == "Undergraduate" else 0
        input_dict['Education_Level_PG'] = 1 if education_level == "Postgraduate" else 0
        input_dict['Gender_Male'] = 1 if gender == "Male" else 0
        input_dict['Employer_Category_MNC'] = 1 if employer_category == "MNC" else 0
        input_dict['Employer_Category_Private'] = 1 if employer_category == "Private" else 0
        input_dict['Employer_Category_Self'] = 1 if employer_category == "Self" else 0
        input_dict['Employer_Category_Unemployed'] = 1 if employer_category == "Unemployed" else 0
        
        # Create dataframe
        input_data = pd.DataFrame([input_dict])
        
        # Ensure we have all features in the right order
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[feature_names]
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        if prediction == 1:
            st.success(f" **LOAN APPROVED**")
            st.info(f"Approval Confidence: {prediction_proba[1]*100:.2f}%")
        else:
            st.error(f" **LOAN REJECTED**")
            st.warning(f"Rejection Confidence: {prediction_proba[0]*100:.2f}%")