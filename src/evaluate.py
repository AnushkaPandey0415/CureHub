from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from preprocess import load_and_preprocess
import json
import os

def evaluate_model():
    df_test = load_and_preprocess("data/drugsComTest_raw.csv")
    X_test = df_test["review"]
    y_test = df_test["condition"]

    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/diagnosis_model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # Save performance metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    os.makedirs("results", exist_ok=True)

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("✅ metrics.json created!")

    # Save detailed recommendations
    recommendations = []
    for review, actual, predicted in zip(X_test, y_test, y_pred):
        recommendations.append({
            "review": review,
            "actual_condition": actual,
            "predicted_condition": predicted
        })

    with open("results/recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=4)
    print("✅ recommendations.json created!")

if __name__ == "__main__":
    evaluate_model()