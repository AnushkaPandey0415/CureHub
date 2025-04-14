import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import warnings
import pickle
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Load datasets once
train_df = pd.read_csv("data/drugsComTrain_raw.csv")
patient_df = pd.read_csv("data/patient_profile.csv")

# Preprocessing
def preprocess_df(df):
    df = df.dropna(subset=['condition', 'review', 'drugName']).copy()
    df['text'] = df['condition'] + " " + df['review']
    return df

train_df = preprocess_df(train_df)
patient_df = preprocess_df(patient_df)

# Sentiment analysis
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

if 'sentiment' not in train_df.columns:
    train_df['sentiment'] = train_df['review'].apply(get_sentiment)

if 'sentiment' not in patient_df.columns:
    patient_df['sentiment'] = patient_df['review'].apply(get_sentiment)

# Sample train data for performance
train_df = train_df.sample(n=15000, random_state=42).reset_index(drop=True)

# Load or compute embeddings
embedding_file = "embeddings/train_embeddings.pkl"
print("Loading MiniLM model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(embedding_file):
    print("✅ Loaded cached embeddings.")
    with open(embedding_file, 'rb') as f:
        train_embeddings = pickle.load(f)
else:
    print("⚙ Computing embeddings...")
    train_embeddings = model.encode(train_df['text'].tolist(), batch_size=32, show_progress_bar=True)
    os.makedirs("embeddings", exist_ok=True)
    with open(embedding_file, 'wb') as f:
        pickle.dump(train_embeddings, f)
    print("✅ Saved embeddings to disk.")

# Recommend drugs for all patients
def recommend_for_all_patients(top_n=3):
    print("🔍 Generating recommendations for all patients...")
    results = []

    # Precompute train_df groupby
    train_grouped = train_df.copy()
    train_grouped['embedding'] = list(train_embeddings)

    for patient_id in patient_df['patient_id'].unique():
        patient_data = patient_df[patient_df['patient_id'] == patient_id]
        if patient_data.empty:
            continue

        patient_text = " ".join(patient_data['text'].tolist())
        patient_embedding = model.encode([patient_text])[0]

        similarities = cosine_similarity([patient_embedding], train_embeddings)[0]
        train_grouped['similarity'] = similarities

        # Hybrid scoring
        train_grouped['hybrid_score'] = (
            0.7 * train_grouped['rating'] +
            0.3 * (train_grouped['sentiment'] * 10) +
            0.1 * (train_grouped['similarity'] * 10)
        )

        top_drugs = (
            train_grouped.groupby('drugName')['hybrid_score']
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        for _, row in top_drugs.iterrows():
            results.append({
                'patient_id': int(patient_id),
                'recommended_drug': str(row['drugName']),
                'hybrid_score': float(round(row['hybrid_score'], 2))
            })

    results_df = pd.DataFrame(results)

    # Save results in formats more compatible with Streamlit
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/all_recommendations.csv", index=False)
    results_df.to_parquet("results/all_recommendations.parquet", index=False)
    print("✅ Saved recommendations for all patients as CSV and Parquet.")

    return results_df

# Run recommendation and evaluation pipeline
if __name__ == "__main__":
    all_results_df = recommend_for_all_patients(top_n=3)
    print("\nSample Output:\n", all_results_df.head())
