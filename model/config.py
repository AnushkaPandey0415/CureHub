import os

# Directories and config
DATA_DIR = "data"
RESULTS_DIR = "results"
EMBEDDINGS_DIR = "embeddings"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_N = 5
SAMPLE_SIZE = 10000

# Embedding weights
TFIDF_WEIGHT = 0.3
TRANSFORMER_WEIGHT = 0.6
EMB_WEIGHT = 0.7

# Severity scaling
SEVERITY_SCALE = {
    'Mild': 0.85,
    'Moderate': 1.0,
    'Severe': 1.15
}

# Recency configuration
RECENCY_REFERENCE_DATE = "2025-01-01"
MAX_RECENCY_YEARS = 10

# Flags to toggle model modes
USE_COLLAB_FILTERING = True
USE_GRAPH_BASED = True

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)