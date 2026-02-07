import streamlit as st
import pandas as pd
import os
import joblib

# Load models and encoders
model = joblib.load("dropout_model.pkl")
le_fee = joblib.load("fee_encoder.pkl")
le_fin = joblib.load("fin_encoder.pkl")
le_risk = joblib.load("risk_encoder.pkl")

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------
# MODERN UI CSS
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
    border: none;
}
.metric-card {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOGIN DETAILS
# ---------------------------------
USERNAME = "admin"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------------------------
# RISK PREDICTION LOGIC
# ---------------------------------
def predict_risk(attendance, marks, fee_status, financial):
    try:
        # Ensure inputs are valid for the encoders
        f_status = fee_status if fee_status in le_fee.classes_ else le_fee.classes_[0]
        f_diff = financial if financial in le_fin.classes_ else le_fin.classes_[0]
        
        fee_encoded = le_fee.transform([f_status])[0]
        fin_encoded = le_fin.transform([f_diff])[0]
        
        prediction = model.predict([[attendance, marks, fee_encoded, fin_encoded]])
        return le_risk.inverse_transform(prediction)[0]
    except Exception as e:
        return "Unknown"

# ---------------------------------
# LOGIN PAGE
# ---------------------------------
if not st.session_state.logged_in:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h2 style='text-align:center; color:white;'>🔐 Secure Login</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔑 Password", type="password")
            login = st.form_submit_button("Login")
        if login:
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

# ---------------------------------
# DASHBOARD
# ---------------------------------
else:
    col_left, col_right = st.columns([1, 9])
    with col_left:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    with col_right:
        st.markdown("<h1 style='color:white;'>🎓 Student Dropout Prediction Dashboard</h1>", unsafe_allow_html=True)

    # ---------------------------------
    # DATA LOADING & CLEANING
    # ---------------------------------
    csv_file = "students.csv"
    columns = ["ID", "Name", "Attendance", "Marks", "FeeStatus", "FinancialDifficulty"]
    
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)

    # Ensure all required columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    # Clean data types for processing
    df["Attendance"] = pd.to_numeric(df["Attendance"], errors="coerce").fillna(0)
    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce").fillna(0)
    df["FeeStatus"] = df["FeeStatus"].astype(str).replace("nan", "Paid")
    df["FinancialDifficulty"] = df["FinancialDifficulty"].astype(str).replace("nan", "None")

    # ---------------------------------
    # CALCULATE RISK LEVELS
    # ---------------------------------
    # We calculate this on the fly to ensure "Unknown" is replaced by actual model logic
    df["Risk Level"] = df.apply(lambda row: predict_risk(
        row["Attendance"], 
        row["Marks"], 
        row["FeeStatus"], 
        row["FinancialDifficulty"]
    ), axis=1)

    # ---------------------------------
    # STUDENT RISK OVERVIEW
    # ---------------------------------
    st.subheader("📋 Student Risk Overview")
    display_df = df.copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)

    # ---------------------------------
    # RISK SUMMARY METRICS
    # ---------------------------------
    high = len(df[df["Risk Level"] == "High Risk"])
    warn = len(df[df["Risk Level"] == "Warning"])
    safe = len(df[df["Risk Level"] == "Safe"])

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><span style='font-size:30px;'>🔴</span><br><h2>{high}</h2><p>High Risk</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><span style='font-size:30px;'>🟡</span><br><h2>{warn}</h2><p>Warning</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><span style='font-size:30px;'>🟢</span><br><h2>{safe}</h2><p>Safe</p></div>", unsafe_allow_html=True)

    # ---------------------------------
    # ADD NEW STUDENT
    # ---------------------------------
    st.markdown("---")
    st.subheader("➕ Predict & Add New Student")

    with st.form("predict_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)
        with col_a:
            new_id = st.number_input("Student ID", min_value=1, value=101)
            new_name = st.text_input("Student Name")
            new_att = st.slider("Attendance (%)", 0, 100, 75)
        with col_b:
            new_marks = st.slider("Marks (%)", 0, 100, 50)
            new_fee = st.selectbox("Fee Status", ["Paid", "Pending"])
            new_fin = st.selectbox("Financial Difficulty", ["None", "Moderate", "Severe"])
        
        submit = st.form_submit_button("Predict & Save Student")

    if submit:
        if not new_name.strip():
            st.error("❌ Please enter a student name.")
        else:
            new_data = pd.DataFrame([[new_id, new_name, new_att, new_marks, new_fee, new_fin]], columns=columns)
            new_data.to_csv(csv_file, mode='a', header=False, index=False)
            st.success(f"✅ Student {new_name} added to database!")
            st.rerun()

    # ---------------------------------
    # DELETE STUDENT 
    # ---------------------------------
    st.markdown("---")
    st.subheader("🗑️ Remove Student")

    if not df.empty:
        student_list = df["Name"].tolist()
        to_delete = st.selectbox("Select Student to Remove", student_list)
        
        if st.button("Delete Selected Student"):
            df_updated = df[df["Name"] != to_delete]
            # Save only the core columns back to CSV
            df_updated[columns].to_csv(csv_file, index=False)
            st.success(f"✅ {to_delete} removed.")
            st.rerun()