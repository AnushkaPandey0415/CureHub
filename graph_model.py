# graph_model.py
import networkx as nx
from logger import logger
from collections import defaultdict

def build_graph(df):
    logger.info("Building condition-drug graph...")
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['condition'], type='condition')
        G.add_node(row['drugName'], type='drug')
        G.add_edge(row['condition'], row['drugName'], weight=row['rating'])
    return G

def graph_recommend(train_df, patient_df, top_n=5):
    G = build_graph(train_df)
    recommendations = []

    for _, patient in patient_df.iterrows():
        condition = patient['condition']
        if condition not in G:
            continue
        neighbors = list(G.neighbors(condition))
        scores = defaultdict(float)

        for drug in neighbors:
            scores[drug] = G[condition][drug]['weight']

        top_drugs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations.append({
            'patient_id': patient['patient_id'],
            'condition': condition,
            'recommendations': [{'drug': d, 'score': round(s, 2)} for d, s in top_drugs]
        })

    return recommendations