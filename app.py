#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[6]:


# pip install shap


# In[7]:


# pip install xgboost


# In[8]:


import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Explainability
import shap

import warnings
warnings.filterwarnings("ignore")


# # Load Dataset

# In[9]:


df = pd.read_csv("student_performance_updated_1000.csv")

df.head()


# # Data Cleaning

# ### Drop useless columns

# In[10]:


df = df.drop(columns=["StudentID", "Name"])


# ### Fix duplicate columns

# In[11]:


# Keep consistent columns
df = df.drop(columns=["Study Hours", "Attendance (%)"])

df.rename(columns={
    "StudyHoursPerWeek": "StudyHours",
    "AttendanceRate": "Attendance"
}, inplace=True)


# ### Check missing values

# In[12]:


df.isnull().sum()


# In[13]:


df = df.dropna()


# In[14]:


df.isnull().sum()


# # EDA (Exploratory Data Analysis)

# ### Basic Overview

# In[15]:


# Dataset shape
print(df.shape)

# Data types
print(df.dtypes)

# Summary statistics
df.describe()


# # Encode Categorical Variables

# In[16]:


le = LabelEncoder()

categorical_cols = [
    "Gender",
    "ExtracurricularActivities",
    "ParentalSupport"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# # Convert Target Variable

# In[17]:


def grade_category(grade):
    if grade >= 75:
        return "High"
    elif grade >= 50:
        return "Medium"
    else:
        return "Low"

df["Performance"] = df["FinalGrade"].apply(grade_category)

df.drop("FinalGrade", axis=1, inplace=True)


# ### Encode target

# In[18]:


df["Performance"] = le.fit_transform(df["Performance"])


# # Feature & Target Split

# In[19]:


X = df.drop("Performance", axis=1)
y = df["Performance"]


# # Train-Test Split

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# # Feature Scaling

# In[21]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Train Models

# ## Decision Tree

# In[22]:


dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)


# ## Random Forest

# In[23]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


# ## Logistic Regression

# In[24]:


lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)


# ## Support Vector Machine (SVM)

# In[25]:


svm = SVC(probability=True)   # probability=True needed for predict_proba
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)


# ## XGBoost

# In[26]:


xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)


# # Evaluation

# ## Accuracy

# In[27]:


print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))
print("SVM:", accuracy_score(y_test, y_pred_svm))
print("XGBoost:", accuracy_score(y_test, y_pred_xgb))
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))


# ## Confusion Matrix

# In[28]:


sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# ## Classification Report

# In[29]:


print(classification_report(y_test, y_pred_lr))


# # Model Selection

# In[30]:


final_model = lr


# # Explainable AI (SHAP)

# In[31]:


explainer = shap.Explainer(final_model, X_train)
shap_values = explainer(X_test)


# ## SHAP Summary Plot

# In[32]:


shap.summary_plot(shap_values, X_test)


# # Save Model

# In[33]:


import joblib

joblib.dump(final_model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Predict student performance and understand influencing factors")

# ==============================
# USER INPUTS
# ==============================

gender = st.selectbox("Gender", ["Male", "Female"])
study_hours = st.slider("Study Hours per Week", 0, 20, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_grade = st.slider("Previous Grade", 0, 100, 60)
activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
parent_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])
online_classes = st.slider("Online Classes Taken", 0, 50, 10)

# ==============================
# ENCODING
# ==============================

gender = 1 if gender == "Male" else 0
activities = 1 if activities == "Yes" else 0
parent_map = {"Low": 0, "Medium": 1, "High": 2}
parent_support = parent_map[parent_support]

# ==============================
# CREATE INPUT DATA
# ==============================

input_data = pd.DataFrame([[
    gender,
    attendance,
    study_hours,
    previous_grade,
    activities,
    parent_support,
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

if st.button("🔍 Predict Performance"):

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)

    # ==============================
    # SAFE LABEL HANDLING (FIXED)
    # ==============================

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    result = label_map.get(prediction, str(prediction))

    st.subheader("📊 Prediction Result")
    st.success(f"Performance Level: {result}")

    # ==============================
    # SAFE PROBABILITY DISPLAY (FIXED)
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
    # SHAP EXPLANATION (FIXED)
    # ==============================

    st.subheader("🧠 Model Explanation (SHAP)")

    try:
        explainer = shap.Explainer(model, scaled_data)
        shap_values = explainer(scaled_data)

        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.warning("SHAP explanation not available for this model.")


# # Visualization

# ### Distribution of Target Variable

# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Performance', data=df)
plt.title("Distribution of Student Performance")
plt.show()


# ### Performance Distribution

# In[37]:


sns.boxplot(x='Performance', y='StudyHours', data=df)
plt.title("Study Hours vs Performance")
plt.show()


# ### Study Behavior Impact

# In[38]:


sns.boxplot(x='Performance', y='Attendance', data=df)
plt.title("Attendance vs Performance")
plt.show()


# ## Correlation Heatmap

# In[39]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




