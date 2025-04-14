import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from model.config import EMBEDDINGS_DIR
from model.logger import logger
from sklearn.metrics.pairwise import cosine_similarity

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

    # Base score using rating + recency + popularity
    drug_scores['score'] = (
        drug_scores['rating'] * 0.6 +
        (1 - drug_scores['recency']) * 0.2 +
        np.log1p(drug_scores['count']) * 0.2
    )

    # Collaborative Filtering-like component: User-Drug similarity matrix (condition-based)
    logger.info("Applying condition-specific embedding similarity...")
    condition_means = []
    for condition in drug_scores['condition'].unique():
        indices = train_df[train_df['condition'] == condition].index.tolist()
        if not indices:
            continue
        condition_embs = embeddings[indices]
        condition_df = train_df.iloc[indices].copy()

        # Compute average embeddings for drugs
        drug_vecs = {}
        for drug in condition_df['drugName'].unique():
            drug_idx = condition_df[condition_df['drugName'] == drug].index.tolist()
            if len(drug_idx) < 2:
                continue
            vec = np.mean(embeddings[drug_idx], axis=0)
            drug_vecs[drug] = vec

        # Compute pairwise similarities
        drug_names = list(drug_vecs.keys())
        if len(drug_names) < 2:
            continue

        mat = np.array([drug_vecs[drug] for drug in drug_names])
        sim = cosine_similarity(mat)
        sim_df = pd.DataFrame(sim, index=drug_names, columns=drug_names)

        # Use mean similarity as a new feature
        mean_sim = sim_df.mean(axis=1).to_dict()
        condition_mean_df = pd.DataFrame({
            'drugName': list(mean_sim.keys()),
            'condition': condition,
            'mean_similarity': list(mean_sim.values())
        })

        condition_means.append(condition_mean_df)

    if condition_means:
        sim_df_all = pd.concat(condition_means)
        drug_scores = drug_scores.merge(sim_df_all, on=['condition', 'drugName'], how='left')
        drug_scores['mean_similarity'] = drug_scores['mean_similarity'].fillna(0)
        drug_scores['score'] += drug_scores['mean_similarity'] * 0.1

    # Clustering for exploration
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