from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from preprocess import load_and_preprocess

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

    print("Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

if __name__ == "__main__":
    evaluate_model()
