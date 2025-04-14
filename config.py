import os

# Directories and config
DATA_DIR = "data"
RESULTS_DIR = "results"
EMBEDDINGS_DIR = "embeddings"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_N = 5
SAMPLE_SIZE = 10000

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)