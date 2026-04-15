import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==============================
# LOAD MODEL (FIXED - CACHED)
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("student_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Predict student performance and understand influencing factors")

# ==============================
# USER INPUTS (FIXED USING FORM)
# ==============================
with st.form("prediction_form"):

    gender = st.selectbox("Gender", ["Male", "Female"])
    study_hours = st.slider("Study Hours per Week", 0, 20, 5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    previous_grade = st.slider("Previous Grade", 0, 100, 60)
    activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    parent_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])
    online_classes = st.slider("Online Classes Taken", 0, 50, 10)

    submit = st.form_submit_button("🔍 Predict Performance")

# ==============================
# ENCODING
# ==============================
gender_val = 1 if gender == "Male" else 0
activities_val = 1 if activities == "Yes" else 0
parent_map = {"Low": 0, "Medium": 1, "High": 2}
parent_val = parent_map[parent_support]

# ==============================
# CREATE INPUT DATA
# ==============================
input_data = pd.DataFrame([[
    gender_val,
    attendance,
    study_hours,
    previous_grade,
    activities_val,
    parent_val,
    online_classes
]], columns=[
    "Gender",
    "Attendance",
    "StudyHours",
    "PreviousGrade",
    "ExtracurricularActivities",
    "ParentalSupport",
    "Online Classes Taken"
])

# Match training feature order
input_data = input_data[scaler.feature_names_in_]

# ==============================
# PREDICTION
# ==============================
if submit:

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)

    # Label mapping
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    result = label_map.get(prediction, str(prediction))

    st.subheader("📊 Prediction Result")
    st.success(f"Performance Level: {result}")

    # ==============================
    # PROBABILITY
    # ==============================
    st.write("### 🔢 Prediction Probability")

    classes = model.classes_
    probs = probability[0]

    prob_dict = {}
    for i, cls in enumerate(classes):
        label = label_map.get(cls, str(cls))
        prob_dict[label] = round(probs[i], 2)

    st.write(prob_dict)

    # ==============================
    # SHAP (UNCHANGED - AS OLD)
    # ==============================
    st.subheader("🧠 Model Explanation (SHAP)")

    try:
        explainer = shap.Explainer(model, scaled_data)
        shap_values = explainer(scaled_data)

        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig, clear_figure=True)

    except Exception:
        st.warning("SHAP explanation not available for this model.")
