import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

MODEL_FILE = "models.pkl"

def generate_synthetic_data(n=500):
    np.random.seed(42)

    data = pd.DataFrame({
        "study_time": np.random.randint(1, 6, n),
        "free_time": np.random.randint(1, 6, n),
        "past_marks": np.random.randint(40, 95, n),
        "difficulty_score": np.random.randint(1, 10, n),
        "resource_score": np.random.randint(1, 10, n),
    })

    data["predicted_marks"] = (
        data["past_marks"] +
        data["study_time"] * 5 -
        data["free_time"] * 2 -
        data["difficulty_score"] * 1.5 +
        data["resource_score"] * 2 +
        np.random.normal(0, 5, n)
    ).clip(0, 100)

    def grade(m):
        if m >= 85: return "A"
        elif m >= 70: return "B"
        elif m >= 55: return "C"
        else: return "D"

    data["grade"] = data["predicted_marks"].apply(grade)

    return data


def train_model():
    df = generate_synthetic_data()

    X = df.drop(["predicted_marks", "grade"], axis=1)
    y_reg = df["predicted_marks"]
    y_clf = df["grade"]

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2)

    reg = RandomForestRegressor()
    clf = RandomForestClassifier()

    reg.fit(X_train, y_reg_train)
    clf.fit(X_train, y_clf_train)

    reg_pred = reg.predict(X_test)
    clf_pred = clf.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_reg_test, reg_pred),
        "accuracy": accuracy_score(y_clf_test, clf_pred)
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((reg, clf, metrics), f)

    return reg, clf, metrics


def load_model():
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    except:
        return train_model()