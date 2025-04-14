# collaborative.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from logger import logger

def build_user_item_matrix(df):
    logger.info("Building user-item matrix...")
    df = df.copy()
    df['user_id'] = df.groupby(['condition']).ngroup()
    df['item_id'] = df.groupby(['drugName']).ngroup()

    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values)
    return user_item_matrix, sparse_matrix

def collaborative_recommend(train_df, patient_df, top_n=5):
    logger.info("Running collaborative filtering...")
    user_item_matrix, sparse_matrix = build_user_item_matrix(train_df)
    similarity = cosine_similarity(sparse_matrix)

    recommendations = []
    for user_index in range(min(len(patient_df), similarity.shape[0])):
        sim_scores = similarity[user_index]
        top_users = sim_scores.argsort()[-top_n:][::-1]
        recommended_items = user_item_matrix.iloc[top_users].mean(axis=0).nlargest(top_n)
        drug_names = [user_item_matrix.columns[i] for i in recommended_items.index]

        recommendations.append({
            'patient_id': patient_df.iloc[user_index]['patient_id'],
            'condition': patient_df.iloc[user_index]['condition'],
            'recommendations': [{'drug': str(d), 'score': float(s)} for d, s in zip(drug_names, recommended_items.values)]
        })

    return recommendations