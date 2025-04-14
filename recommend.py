from config import TOP_N
from logger import logger
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend(patient_df, model, patient_emb, tfidf, top_n=TOP_N):
    """Generate personalized drug recommendations"""
    logger.info("Generating recommendations...")
    recommendations = []

    drug_scores = model['drug_scores']
    drug_embeddings = model.get('drug_embeddings', None)
    drug_texts = model.get('drug_texts', None)

    for i, patient in patient_df.iterrows():
        condition = patient['condition']
        severity = patient['severity']
        patient_vector = patient_emb[i].reshape(1, -1)

        candidates = drug_scores[drug_scores['condition'] == condition]
        if candidates.empty:
            candidates = drug_scores.nlargest(100, 'count')

        scores = candidates['score'].copy()

        # Adjust for severity
        if severity == 'Severe':
            scores *= 1.2
        elif severity == 'Mild':
            scores *= 0.8

        # Similarity boost
        if drug_embeddings is not None and drug_texts is not None:
            try:
                drug_text_indices = [drug_texts.index(drug) for drug in candidates['drugName']]
                drug_vecs = drug_embeddings[drug_text_indices]
                sim = cosine_similarity(patient_vector, drug_vecs).flatten()
                scores += sim * 0.1  # Add small boost based on embedding similarity
            except Exception as e:
                logger.warning(f"Embedding similarity skipped due to: {e}")

        top_drugs = candidates.assign(score=scores).nlargest(top_n, 'score')

        recommendations.append({
            'patient_id': patient['patient_id'],
            'condition': condition,
            'recommendations': [
                {'drug': row['drugName'], 'score': round(row['score'], 3)}
                for _, row in top_drugs.iterrows()
            ]
        })

    return recommendations