import numpy as np

def preprocess_input(study_time, free_time, past_marks, difficult, resources):
    difficulty_score = len(difficult.split(",")) if difficult else 1
    resource_score = len(resources.split(",")) if resources else 1

    return np.array([[study_time, free_time, past_marks, difficulty_score, resource_score]])


def analyze_performance(predicted_marks, target):
    return target - predicted_marks


def generate_recommendations(gap, difficult, easy, study_time, free_time):
    recs = []

    if gap > 15:
        recs.append("🚨 You are far from your target. Start improving step by step.")
    elif gap > 5:
        recs.append("⚠️ You are close to your goal. Stay consistent.")
    else:
        recs.append("🎉 Great! You are on track!")

    if study_time < 3:
        recs.append("Increase your daily study time gradually.")
    else:
        recs.append("Your study time is good. Focus on quality.")

    recs.append("❗ Avoid too much free time.")
    recs.append("❗ Revise regularly.")
    recs.append("❗ Avoid last-minute study.")

    if difficult:
        recs.append(f"📉 Focus on: {difficult}")

    if easy:
        recs.append(f"📈 Strengths: {easy}")

    recs.append("🧠 Practice tests weekly.")
    recs.append("🤝 You can improve with consistency.")

    return recs


def future_guidance(predicted_marks, target):
    if predicted_marks < target:
        return """
📌 Improve Gradually:
- Focus on weak subjects
- Track weekly progress
- Take mock tests
- Stay consistent

🚀 You can achieve your goal!
"""
    else:
        return """
🌟 Keep Improving:
- Maintain routine
- Practice advanced questions
- Help others learn

🎯 Aim higher!
"""


def create_study_plan(study_time):
    return f"""
Daily Plan:
- {study_time} hrs study
- 1 hr revision
- 30 min practice
- Break every 1 hour
"""
