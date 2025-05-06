import streamlit as st
import pandas as pd
import pickle

def load_artifacts(model_path='models/model.pkl', preprocessor_path='models/preprocessor.pkl'):
    """Load the trained model and preprocessor."""
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model

def preprocess(df):
    """Apply feature mapping and engineering consistent with training."""
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    df['loan_grade'] = df['loan_grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
    df['int_rate_x_percent_income'] = df['loan_int_rate'] * df['loan_percent_income']
    return df

@st.cache_resource
def get_model_and_preprocessor():
    return load_artifacts()


st.title("🏦 Loan Approval Prediction App")
st.write("Enter the applicant's details to predict loan approval status.")

# Load model and preprocessor
try:
    preprocessor, model = get_model_and_preprocessor()
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")

# Input fields with validation
with st.form("loan_form"):
    col1, col2 = st.columns(2)
        
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Applicant's age (18-100)")
        person_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, help="Annual income in dollars")
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], help="Home ownership status")
        person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=5.0, help="Years employed")
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], help="Purpose of the loan")
            
    with col2:
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], help="Loan grade (A-G)")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0, help="Requested loan amount")
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, help="Loan interest rate")
        loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.1, help="Loan amount as fraction of income (0-1)")
        cb_person_default_on_file = st.selectbox("Default on File", ["N", "Y"], help="History of default (Y/N)")
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5, help="Years of credit history")

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create DataFrame from inputs
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    }
    df = pd.DataFrame([input_data])

    # Preprocess and predict
    try:
        df = preprocess(df)
        X_transformed = preprocessor.transform(df)
        prediction = model.predict(X_transformed)[0]
        probability = model.predict_proba(X_transformed)[0][1]

        # Display result
        st.subheader("Prediction Result")
        if prediction == 1:
          st.success("Loan is likely to be **Approved**")
        else:
          st.error("Loan is likely to be **Denied**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Batch prediction
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 
                            'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 
                            'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']
        if all(col in df.columns for col in required_columns):
            df = preprocess(df)
            X_transformed = preprocessor.transform(df)
            predictions = model.predict(X_transformed)
            probabilities = model.predict_proba(X_transformed)[:, 1]
                
            # Save predictions
            pred_df = df[['id']] if 'id' in df.columns else pd.DataFrame(index=df.index)
            pred_df['loan_status'] = predictions
            pred_df['approval_probability'] = probabilities
            st.write(pred_df)
                
            # Download button
            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
        else:
            st.error(f"CSV must contain columns: {', '.join(required_columns)}")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")

# Model info
with st.expander("About the Model"):
    st.write("""
    This app uses a machine learning model to predict loan approval.
    - **Model**: XGBoost
    - **Features**: Age, Income, Home Ownership, Employment Length, Loan Intent, Loan Grade, Loan Amount, Interest Rate, Loan % Income, Default on File, Credit History Length
    - **Preprocessing**: Standard Scaler / PowerTransformer for numerical features and OneHotEncoding for categorical features """)
