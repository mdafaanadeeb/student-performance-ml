import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import load_model
from utils import preprocess_input, analyze_performance, generate_recommendations, create_study_plan, future_guidance

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Student Mentor", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
h1, h2, h3 {
    color: #ffffff;
}

/* Card Style */
.card {
    background-color: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Button */
.stButton button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}

/* Metric */
[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align:center;'>🎓 AI Student Performance Mentor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Your personal academic improvement assistant 🚀</p>", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
reg, clf, metrics = load_model()

# -------------------- SIDEBAR --------------------
st.sidebar.header("📥 Enter Student Details")

name = st.sidebar.text_input("Name")
student_class = st.sidebar.text_input("Class")
branch = st.sidebar.text_input("Branch")

study_time = st.sidebar.slider("Study Time (hrs/day)", 1, 10, 3)
free_time = st.sidebar.slider("Free Time (hrs/day)", 1, 10, 3)

difficult = st.sidebar.text_input("Difficult Subjects")
easy = st.sidebar.text_input("Easy Subjects")

past_marks = st.sidebar.slider("Past Marks", 0, 100, 60)
target = st.sidebar.slider("Target Marks", 0, 100, 85)

resources = st.sidebar.text_input("Study Resources")

# -------------------- PREDICTION --------------------
if st.sidebar.button("🚀 Predict Performance"):

    X = preprocess_input(study_time, free_time, past_marks, difficult, resources)

    pred_marks = reg.predict(X)[0]
    pred_grade = clf.predict(X)[0]

    gap = analyze_performance(pred_marks, target)
    recs = generate_recommendations(gap, difficult, easy, study_time, free_time)
    plan = create_study_plan(study_time)

    # -------------------- WELCOME --------------------
    st.success(f"✨ Welcome {name}! Let's improve your performance step by step.")

    # -------------------- METRICS --------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Predicted Marks", f"{pred_marks:.2f}")
    col2.metric("🎯 Grade", pred_grade)
    col3.metric("📉 Gap", f"{gap:.2f}")

    # -------------------- PROGRESS BAR --------------------
    progress = int((pred_marks / target) * 100) if target != 0 else 0
    st.progress(min(progress, 100))

    # -------------------- CHART --------------------
    st.markdown("### 📈 Performance Overview")

    fig, ax = plt.subplots()
    ax.bar(["Past", "Predicted", "Target"], [past_marks, pred_marks, target],
           color=["#ff4b4b", "#00c6ff", "#00ff94"])
    ax.set_ylabel("Marks")
    ax.set_title("Performance Comparison")
    st.pyplot(fig)

    # -------------------- WEAK AREAS --------------------
    st.markdown("### ⚠️ Weak Areas")
    st.write(difficult if difficult else "None")

    # -------------------- RECOMMENDATIONS --------------------
    st.markdown("### 🤝 Mentor Advice")
    for r in recs:
        st.markdown(f"- {r}")

    # -------------------- STUDY PLAN --------------------
    st.markdown("### 📅 Study Plan")
    st.code(plan)

    # -------------------- FUTURE GUIDANCE --------------------
    st.markdown("### 🚀 Future Guidance")
    st.info(future_guidance(pred_marks, target))

    # -------------------- METRICS --------------------
    st.markdown("### 📊 Model Performance")
    st.write(metrics)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed by <b>Your Name</b> 🚀</p>", unsafe_allow_html=True)
