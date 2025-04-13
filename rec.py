import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load datasets
train_df = pd.read_csv("drugsComTrain_raw.csv")
patient_df = pd.read_csv("patient_profile.csv")

# Basic preprocessing
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

# Limit training set size to speed up embedding
MAX_ROWS = 15000  # you can tweak this if you have more time
train_df = train_df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

# Initialize SentenceTransformer (MiniLM)
print("Loading MiniLM model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
print("Computing embeddings for training data...")
train_embeddings = model.encode(train_df['text'].tolist(), batch_size=32, show_progress_bar=True)
train_df['embeddings'] = list(train_embeddings)

# Recommendation Engine
def recommend_from_patient_history(patient_id, top_n=3):
    patient_history = patient_df[patient_df['patient_id'] == patient_id]
    if patient_history.empty:
        return None, f"No data found for patient ID: {patient_id}"

    patient_text = " ".join(patient_history['text'].tolist())
    patient_embedding = model.encode([patient_text])

    similarities = cosine_similarity(patient_embedding, train_embeddings)[0]
    train_df['similarity'] = similarities

    train_df['hybrid_score'] = (
        0.7 * train_df['rating'] +
        0.3 * (train_df['sentiment'] * 10) +
        0.1 * (train_df['similarity'] * 10)
    )

    top_drugs = (
        train_df.groupby('drugName')['hybrid_score']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    return top_drugs, None

# Generate recommendations
results = []
for pid in patient_df['patient_id'].unique():
    top_drugs, error = recommend_from_patient_history(pid, top_n=3)
    if error:
        print(error)
        continue
    for _, row in top_drugs.iterrows():
        results.append({
            'patient_id': pid,
            'recommended_drug': row['drugName'],
            'hybrid_score': round(row['hybrid_score'], 2)
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("personalized_recommendations.csv", index=False)

# JSON output for website
results_json = results_df.groupby('patient_id').apply(
    lambda x: x[['recommended_drug', 'hybrid_score']].to_dict('records')
).to_dict()
with open("personalized_recommendations.json", "w") as f:
    json.dump(results_json, f, indent=2)

# Feasibility metrics
condition_coverage = len(patient_df['condition'].unique()) / len(train_df['condition'].unique()) * 100
avg_confidence = results_df['hybrid_score'].mean()

print(f"✔ Recommendations saved to personalized_recommendations.csv and .json")
print(f"Feasibility Metrics:")
print(f"- Condition Coverage: {condition_coverage:.2f}% of training conditions")
print(f"- Average Recommendation Confidence: {avg_confidence:.2f}")

# Visualization
plt.figure(figsize=(8, 4))
sns.histplot(results_df['hybrid_score'], bins=20, kde=True, color="teal")
plt.xlabel("Hybrid Score")
plt.ylabel("Frequency")
plt.title("Distribution of Recommendation Scores Across Patients")
plt.tight_layout()
plt.savefig("recommendation_score_distribution.png")
plt.close()
