import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_data.csv")

# Features and target
X = df[["Hours", "Attendance", "Previous Marks", "Assignments"]]
y = df["Final Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Website title
st.title("Student Performance Predictor")

st.write("Enter the student's details to predict their final marks.")

# User inputs
hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_marks = st.number_input(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    value=65.0
)

assignments = st.number_input(
    "Assignment Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

# Prediction button
if st.button("Predict Final Marks"):

    new_student = [[
        hours,
        attendance,
        previous_marks,
        assignments
    ]]

    prediction = model.predict(new_student)

    st.success(
        f"Predicted Final Marks: {prediction[0]:.2f}"
    )
