import pickle
from preprocess import clean_text, load_and_preprocess
import pandas as pd

def recommend_drug(symptom_description):
    with open("models/diagnosis_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    cleaned = clean_text(symptom_description)
    X = vectorizer.transform([cleaned])
    predicted_condition = model.predict(X)[0]

    # Find best drug from training data
    df = load_and_preprocess("data/drugsComTrain_raw.csv")
    condition_df = df[df["condition"] == predicted_condition]
    top_drug = condition_df.groupby("drugName")["rating"].mean().idxmax()

    return predicted_condition, top_drug