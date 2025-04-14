import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from config import EMBEDDINGS_DIR
from logger import logger

def train_model(train_df, embeddings):
    """Train recommendation model and cluster scores"""
    model_file = os.path.join(EMBEDDINGS_DIR, "model.pkl")
    if os.path.exists(model_file):
        logger.info("Loading cached model...")
        with open(model_file, 'rb') as f:
            return pickle.load(f)

    logger.info("Training model...")
    drug_scores = train_df.groupby(['condition', 'drugName']).agg({
        'rating': 'mean',
        'recency': 'mean',
        'drugName': 'count'
    }).rename(columns={'drugName': 'count'}).reset_index()

    drug_scores['score'] = (
        drug_scores['rating'] * 0.7 +
        (1 - drug_scores['recency']) * 0.2 +
        np.log1p(drug_scores['count']) * 0.1
    )

    clusters = {}
    for condition in drug_scores['condition'].unique():
        cond_drugs = drug_scores[drug_scores['condition'] == condition].copy()
        if len(cond_drugs) > 3:
            features = cond_drugs[['score', 'rating']].values
            kmeans = KMeans(n_clusters=min(3, len(cond_drugs)), random_state=42)
            cond_drugs.loc[:, 'cluster'] = kmeans.fit_predict(features)
            clusters[condition] = cond_drugs

    model = {'drug_scores': drug_scores, 'clusters': clusters}
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    return model
