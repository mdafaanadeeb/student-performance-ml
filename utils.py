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
        recs.append("🚨 You are significantly below your target. Let’s improve step by step.")
    elif gap > 5:
        recs.append("⚠️ You are close to your goal, but need consistency.")
    else:
        recs.append("🎉 Great! You're on track. Keep it up!")

    if study_time < 3:
        recs.append("Increase your daily study time gradually.")
    else:
        recs.append("Your study time is good. Focus on quality learning.")

    recs.append("❗ Avoid excessive free time.")
    recs.append("❗ Do not skip revision.")
    recs.append("❗ Avoid last-minute cramming.")

    if difficult:
        recs.append(f"📉 Focus on difficult subjects: {difficult}")

    if easy:
        recs.append(f"📈 Use strengths in: {easy}")

    recs.append("🧠 Use practice tests and active recall.")
    recs.append("🤝 You're not alone. Small daily progress matters.")

    return recs


def future_guidance(predicted_marks, target):
    if predicted_marks < target:
        return """
📌 Future Growth Plan:
- Improve weak subjects step by step
- Track weekly progress
- Take mock tests regularly
- Stay consistent

🚀 You can reach your goal!
"""
    else:
        return """
🌟 Excellence Plan:
- Maintain consistency
- Practice advanced questions
- Help others learn

🎯 Aim even higher!
"""


def create_study_plan(study_time):
    return f"""
Daily Plan:
- {study_time} hrs focused study
- 1 hr revision
- 30 min practice tests
- Break every 1 hour
"""