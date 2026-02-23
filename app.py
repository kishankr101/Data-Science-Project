import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Loan Analytics Dashboard", layout="wide")

# -----------------------------
# GRADIENT BACKGROUND
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #329F6C, #8662F3);
}
/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #329F6C, #8662F3);
    color: white;
}

/* Sidebar text color */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Success prediction box */
.approve-box {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

/* Reject prediction box */
.reject-box {
    background-color: #f8d7da;
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv")


    df["Credit_History"] = df["Credit_History"].fillna(0)
    df["Self_Employed"] = df["Self_Employed"].fillna("No")
    df = df.dropna()


    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanToIncomeRatio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)

    df["Loan_Status"] = (
        (df["Credit_History"] == 1) &
        (df["LoanToIncomeRatio"] < 0.35)
    ).astype(int)

    return df

data = load_data()

# -----------------------------
# MODEL
# -----------------------------
X = data[["TotalIncome", "LoanAmount", "LoanToIncomeRatio", "Credit_History"]]
y = data["Loan_Status"]

model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=data["Gender"].unique(),
    default=data["Gender"].unique()
)

married_filter = st.sidebar.multiselect(
    "Married",
    options=data["Married"].dropna().unique(),
    default=data["Married"].dropna().unique()
)

self_emp_filter = st.sidebar.multiselect(
    "Self Employed",
    options=data["Self_Employed"].unique(),
    default=data["Self_Employed"].unique()
)

property_filter = st.sidebar.multiselect(
    "Property Area",
    options=data["Property_Area"].unique(),
    default=data["Property_Area"].unique()
)

credit_filter = st.sidebar.multiselect(
    "Credit History",
    options=data["Credit_History"].unique(),
    default=data["Credit_History"].unique()
)

filtered = data[
    (data["Gender"].isin(gender_filter)) &
    (data["Married"].isin(married_filter)) &
    (data["Self_Employed"].isin(self_emp_filter)) &
    (data["Property_Area"].isin(property_filter)) &
    (data["Credit_History"].isin(credit_filter))
]

# -----------------------------
# HEADER
# -----------------------------
st.title("Bank Loan Analytics Dashboard")

# -----------------------------
# METRICS
# -----------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Applicants", len(filtered))
m2.metric("Average Loan", round(filtered["LoanAmount"].mean(), 1))
m3.metric("Approval Rate", f"{round(filtered['Loan_Status'].mean()*100,1)}%")

st.divider()



# -----------------------------
# PREDICTION PANEL
# -----------------------------
st.subheader("Loan Approval Predictor")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Applicant Income", 0, 50000, 5000)
    co_income = st.number_input("Coapplicant Income", 0, 50000, 0)

with col2:
    loan_amount = st.number_input("Loan Amount", 1, 500, 120)
    credit = st.selectbox("Credit History", [1, 0])

total_income = income + co_income
ratio = loan_amount / (total_income + 1)

st.write("Total Income:", total_income)
st.write("Loan-to-Income Ratio:", round(ratio, 3))

if st.button("Predict Loan Approval"):
    input_df = pd.DataFrame(
        [[total_income, loan_amount, ratio, credit]],
        columns=X.columns
    )

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("Loan Likely Approved")
    else:
        st.error("Loan Likely Rejected")