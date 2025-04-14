import time
import json
import os
from logger import logger
from config import RESULTS_DIR
from data_loader import load_data
from embedding import create_embeddings
from model import train_model
from recommend import recommend
from evaluate import evaluate

def main():
    start = time.time()
    logger.info("Starting recommendation pipeline...")

    try:
        train_df, test_df, patient_df = load_data()
        train_emb, tfidf = create_embeddings(train_df)
        patient_emb, _ = create_embeddings(patient_df)
        model = train_model(train_df, train_emb)
        recommendations = recommend(patient_df, model, patient_emb, tfidf)
        metrics = evaluate(test_df, model)

        with open(os.path.join(RESULTS_DIR, "recommendations.json"), 'w') as f:
            json.dump(recommendations, f, indent=2)

        elapsed = time.time() - start
        logger.info(f"Pipeline completed in {elapsed:.2f}s")

        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'elapsed': elapsed
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    result = main()
    print(f"Time: {result['elapsed']:.2f}s")
    print(f"Hit Rate: {result['metrics']['hit_rate']:.4f}, MRR: {result['metrics']['mrr']:.4f}")