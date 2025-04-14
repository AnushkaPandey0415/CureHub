from model.config import TOP_N
from model.logger import logger

def evaluate(test_df, model):
    """Evaluate model using Hit Rate and MRR"""
    logger.info("Evaluating model...")
    hits = 0
    mrr = 0
    n = min(1000, len(test_df))

    for _, row in test_df.sample(n, random_state=42).iterrows():
        condition = row['condition'].strip().lower()
        actual_drug = str(row['drugName']).strip().lower()

        candidates = model['drug_scores']
        candidates = candidates[candidates['condition'].str.strip().str.lower() == condition]

        if candidates.empty:
            continue

        top_drugs = candidates.nlargest(TOP_N, 'score')['drugName'].str.strip().str.lower().tolist()
        if actual_drug in top_drugs:
            hits += 1
            rank = top_drugs.index(actual_drug) + 1
            mrr += 1 / rank

    return {
        'hit_rate': hits / n,
        'mrr': mrr / n,
        'evaluated': n
    }