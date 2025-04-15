import pandas as pd
import re

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.drop(columns=["usefulCount", "uniqueID", "date"], inplace=True)
    df.dropna(subset=["review", "condition", "drugName"], inplace=True)
    df["review"] = df["review"].apply(clean_text)
    return df