import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import load_model
from utils import preprocess_input, analyze_performance, generate_recommendations, create_study_plan, future_guidance

st.set_page_config(page_title="Student Performance ML", layout="wide")

st.title("🎓 Student Performance Prediction System")

reg, clf, metrics = load_model()

# Sidebar Input
st.sidebar.header("Student Input")

name = st.sidebar.text_input("Name")
student_class = st.sidebar.text_input("Class")
branch = st.sidebar.text_input("Branch")

study_time = st.sidebar.slider("Study Time", 1, 10, 3)
free_time = st.sidebar.slider("Free Time", 1, 10, 3)

difficult = st.sidebar.text_input("Difficult Subjects")
easy = st.sidebar.text_input("Easy Subjects")

past_marks = st.sidebar.slider("Past Marks", 0, 100, 60)
target = st.sidebar.slider("Target Marks", 0, 100, 85)

resources = st.sidebar.text_input("Study Resources")

if st.sidebar.button("Predict"):

    X = preprocess_input(study_time, free_time, past_marks, difficult, resources)

    pred_marks = reg.predict(X)[0]
    pred_grade = clf.predict(X)[0]

    gap = analyze_performance(pred_marks, target)

    recs = generate_recommendations(gap, difficult, easy, study_time, free_time)
    plan = create_study_plan(study_time)

    st.header(f"📊 Results for {name}")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Marks", f"{pred_marks:.2f}")
    col2.metric("Predicted Grade", pred_grade)

    st.subheader("📉 Performance Gap")
    st.write(f"{gap:.2f}")

    # Chart using matplotlib (SAFE)
    fig, ax = plt.subplots()
    ax.bar(["Past", "Predicted", "Target"], [past_marks, pred_marks, target])
    ax.set_ylabel("Marks")
    ax.set_title("Performance Comparison")
    st.pyplot(fig)

    st.subheader("⚠️ Weak Areas")
    st.write(difficult if difficult else "None")

    st.subheader("🤝 Mentor Advice")
    for r in recs:
        st.write("- ", r)

    st.subheader("📅 Study Plan")
    st.text(plan)

    st.subheader("🚀 Future Guidance")
    st.info(future_guidance(pred_marks, target))

    st.subheader("📊 Model Metrics")
    st.write(metrics)
