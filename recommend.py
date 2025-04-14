from config import TOP_N
from logger import logger

def recommend(patient_df, model, patient_emb, tfidf, top_n=TOP_N):
    """Generate personalized drug recommendations"""
    logger.info("Generating recommendations...")
    recommendations = []

    for _, patient in patient_df.iterrows():
        condition = patient['condition']
        severity = patient['severity']

        candidates = model['drug_scores'][model['drug_scores']['condition'] == condition]
        if candidates.empty:
            candidates = model['drug_scores'].nlargest(100, 'count')

        scores = candidates['score'].copy()
        if severity == 'Severe':
            scores *= 1.2
        elif severity == 'Mild':
            scores *= 0.8

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