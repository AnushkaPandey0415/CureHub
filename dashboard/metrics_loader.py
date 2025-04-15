import json
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import load_and_preprocess

def load_metrics():
    with open("results/metrics.json", "r") as f:
        return json.load(f)

def load_predictions():
    df_test = load_and_preprocess("data/drugsComTest_raw.csv")

    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/diagnosis_model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = vectorizer.transform(df_test["review"])
    y_pred = model.predict(X_test)
    y_true = df_test["condition"]

    return y_true, y_pred