import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import warnings
import json
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
    return TextBlob(text).sentiment.polarity

train_df['sentiment'] = train_df['review'].apply(get_sentiment)
patient_df['sentiment'] = patient_df['review'].apply(get_sentiment)

# Sample training data to improve performance
train_df = train_df.sample(n=15000, random_state=42).reset_index(drop=True)

# Load MiniLM model once
print("Loading MiniLM model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Check if embeddings are already cached
embedding_file = "embeddings/train_embeddings.pkl"
if os.path.exists(embedding_file):
    print("Loading precomputed embeddings from pickle file...")
    with open(embedding_file, 'rb') as f:
        train_embeddings = pickle.load(f)
else:
    # Compute training embeddings if not cached
    print("Computing training embeddings...")
    train_embeddings = model.encode(train_df['text'].tolist(), batch_size=32, show_progress_bar=True)
    
    # Cache the embeddings to pickle
    with open(embedding_file, 'wb') as f:
        pickle.dump(train_embeddings, f)
    print(f"Embeddings saved to {embedding_file}")

# Function to recommend for a single patient
def recommend_for_patient(patient_id, top_n=3):
    patient_data = patient_df[patient_df['patient_id'] == patient_id]
    if patient_data.empty:
        return None, f"❌ No data found for patient ID: {patient_id}"

    # Compute patient embedding
    patient_text = " ".join(patient_data['text'].tolist())
    patient_embedding = model.encode([patient_text])[0]

    # Compute cosine similarity
    similarities = cosine_similarity([patient_embedding], train_embeddings)[0]
    train_df['similarity'] = similarities

    # Hybrid score
    train_df['hybrid_score'] = (
        0.7 * train_df['rating'] +
        0.3 * (train_df['sentiment'] * 10) +
        0.1 * (train_df['similarity'] * 10)
    )

    # Top recommended drugs
    top_drugs = (
        train_df.groupby('drugName')['hybrid_score']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    results = []
    for _, row in top_drugs.iterrows():
        results.append({
            'patient_id': patient_id,
            'recommended_drug': row['drugName'],
            'hybrid_score': round(row['hybrid_score'], 2)
        })

    # Save to CSV and JSON
    result_df = pd.DataFrame(results)
    result_df.to_csv("results/personalized_recommendation_single.csv", index=False)

    with open("results/personalized_recommendation_single.json", "w") as f:
        json.dump({patient_id: results}, f, indent=2)

    print(f"✔ Recommendations for patient {patient_id} saved to personalized_recommendation_single.csv and .json")
    return result_df, None

# Example usage:
patient_id_input = 103  # Replace this ID dynamically from your app or CLI
recommend_for_patient(patient_id_input)
