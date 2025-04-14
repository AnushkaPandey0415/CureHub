import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from model.config import EMBEDDINGS_DIR, MODEL_NAME, EMB_WEIGHT, TFIDF_WEIGHT
from model.logger import logger

def create_embeddings(df, column='text'):
    """Create sentence and TF-IDF embeddings"""
    cache_file = os.path.join(EMBEDDINGS_DIR, f"{column}_embeddings.pkl")
    if os.path.exists(cache_file):
        logger.info(f"Loading cached embeddings for {column}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    logger.info(f"Creating embeddings for {column} using model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    tfidf = TfidfVectorizer(max_features=1000)

    texts = df[column].tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    tfidf_matrix = tfidf.fit_transform(texts).toarray()

    combined = np.hstack([
        embeddings * EMB_WEIGHT,
        tfidf_matrix * TFIDF_WEIGHT
    ])

    with open(cache_file, 'wb') as f:
        pickle.dump((combined, tfidf), f)

    return combined, tfidf