from config import TOP_N
from logger import logger

def evaluate(test_df, model):
    """Evaluate model using Hit Rate and MRR"""
    logger.info("Evaluating model...")
    hits = 0
    mrr = 0
    n = min(1000, len(test_df))

    for _, row in test_df.sample(n, random_state=42).iterrows():
        condition = row['condition']
        actual_drug = row['drugName']

        candidates = model['drug_scores'][model['drug_scores']['condition'] == condition]
        if candidates.empty:
            continue

        top_drugs = candidates.nlargest(TOP_N, 'score')['drugName'].tolist()
        if actual_drug in top_drugs:
            hits += 1
            rank = top_drugs.index(actual_drug) + 1
            mrr += 1 / rank

    return {'hit_rate': hits / n, 'mrr': mrr / n, 'evaluated': n}