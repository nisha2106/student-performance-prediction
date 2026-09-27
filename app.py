import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load ML Model
# -----------------------------
model = joblib.load("student_risk_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Prediction System")
st.subheader("AI-based Academic Risk Detection")

st.write(
    "This system uses Machine Learning to identify students "
    "who may be academically at risk based on their academic indicators."
)

st.divider()

# -----------------------------
# Student Input Section
# -----------------------------
st.header("📊 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    internal_marks = st.number_input(
        "Internal Marks",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

with col2:
    assignment_score = st.number_input(
        "Assignment Score",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=2.0
    )

with col3:
    gpa = st.number_input(
        "GPA",
        min_value=0.0,
        max_value=10.0,
        value=6.0
    )

    previous_failures = st.number_input(
        "Previous Failures",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Student Risk", use_container_width=True):

    input_data = pd.DataFrame({
        "attendance": [attendance],
        "internal_marks": [internal_marks],
        "assignment_score": [assignment_score],
        "study_hours": [study_hours],
        "gpa": [gpa],
        "previous_failures": [previous_failures]
    })

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability_data = pd.DataFrame({
        "Risk Level": classes,
        "Probability": probabilities
    })

    # -----------------------------
    # Display Result
    # -----------------------------
    st.header("📌 Prediction Result")

    if prediction == "High":
        st.error("🔴 HIGH RISK")
        st.write(
            "The model identifies this student as potentially "
            "academically at risk."
        )

    elif prediction == "Medium":
        st.warning("🟡 MEDIUM RISK")
        st.write(
            "The student shows some indicators that may require "
            "academic attention."
        )

    else:
        st.success("🟢 LOW RISK")
        st.write(
            "The model identifies relatively lower academic risk "
            "based on the entered indicators."
        )

    # -----------------------------
    # Probability
    # -----------------------------
    st.subheader("📈 Model Prediction Probabilities")

    for _, row in probability_data.iterrows():
        st.write(
            f"{row['Risk Level']}: "
            f"{row['Probability'] * 100:.2f}%"
        )

        st.progress(float(row["Probability"]))

    # -----------------------------
    # Student Summary
    # -----------------------------
    st.subheader("📋 Student Performance Summary")

    summary = pd.DataFrame({
        "Indicator": [
            "Attendance",
            "Internal Marks",
            "Assignment Score",
            "Study Hours",
            "GPA",
            "Previous Failures"
        ],
        "Value": [
            f"{attendance}%",
            internal_marks,
            assignment_score,
            study_hours,
            gpa,
            previous_failures
        ]
    })
 
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Recommendations
    # -----------------------------
    st.subheader("💡 Recommended Actions")

    recommendations = []

    if attendance < 75:
        recommendations.append(
            "Improve class attendance."
        )

    if internal_marks < 50:
        recommendations.append(
            "Focus on internal examinations and core subjects."
        )

    if assignment_score < 50:
        recommendations.append(
            "Complete assignments regularly and on time."
        )

    if study_hours < 2:
        recommendations.append(
            "Increase consistent daily study time."
        )

    if gpa < 6:
        recommendations.append(
            "Focus on improving overall academic performance."
        )

    if previous_failures > 0:
        recommendations.append(
            "Provide additional support in previously failed subjects."
        )

    if not recommendations:
        recommendations.append(
            "Continue the current academic habits and maintain consistency."
        )

    for recommendation in recommendations:
        st.write("•", recommendation)

# -----------------------------
# Risk Factor Analysis
# -----------------------------
st.subheader("⚠️ Risk Factor Analysis")

risk_factors = []

if attendance < 75:
    risk_factors.append("Low attendance")

if internal_marks < 50:
    risk_factors.append("Low internal marks")

if assignment_score < 50:
    risk_factors.append("Low assignment score")

if study_hours < 2:
    risk_factors.append("Low study hours")

if gpa < 6:
    risk_factors.append("Low GPA")

if previous_failures > 0:
    risk_factors.append("Previous academic failures")

if risk_factors:
    st.warning("The following factors may affect student performance:")

    for risk in risk_factors:
        st.write("•", risk)
else:
    st.success("No major risk factors identified.")

st.divider()
st.caption(
    "Student Performance Prediction System | "
    "Python + Pandas + Scikit-learn + Streamlit"
) 