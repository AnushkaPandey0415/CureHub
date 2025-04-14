import time
import json
import os
from model.logger import logger
from model.config import RESULTS_DIR
from model.data_loader import load_data
from model.embedding import create_embeddings
from model.model import train_model
from model.recommend import recommend
from model.evaluate import evaluate
from model.collaborative import collaborative_recommend
from model.graph_model import graph_recommend
import numpy as np

def main():
    start = time.time()
    logger.info("Starting recommendation pipeline...")

    try:
        # Load and embed data
        train_df, test_df, patient_df = load_data()
        train_emb, tfidf = create_embeddings(train_df)
        patient_emb, _ = create_embeddings(patient_df)

        # Train model using embedded data
        model = train_model(train_df, train_emb)
        collab_recs = collaborative_recommend(train_df, patient_df)
        graph_recs = graph_recommend(train_df, patient_df)

        # Save them with custom serializer
        with open(os.path.join(RESULTS_DIR, "collab_recommendations.json"), 'w') as f:
            json.dump(collab_recs, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else str(o))
        with open(os.path.join(RESULTS_DIR, "graph_recommendations.json"), 'w') as f:
            json.dump(graph_recs, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else str(o))

        # Generate recommendations with updated embeddings
        recommendations = recommend(patient_df, model, patient_emb, tfidf)

        # Evaluate performance
        metrics = evaluate(test_df, model)

        # Save recommendations with serializer
        with open(os.path.join(RESULTS_DIR, "recommendations.json"), 'w') as f:
            json.dump(recommendations, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else str(o))

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