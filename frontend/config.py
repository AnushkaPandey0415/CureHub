# Configuration settings for the frontend visualizations

import os

# Data directories
DATA_DIR = "data"
RESULTS_DIR = "results"

# Visualization settings
CHART_THEME = "dark_background"
COLOR_PALETTE = "Viridis"
COLOR_MAP = {
    'precision': '#1f77b4',
    'recall': '#ff7f0e',
    'f1_score': '#2ca02c',
    'accuracy': '#d62728'
}

# Default selections
DEFAULT_TOP_N = 5

# Chart display settings
CHART_HEIGHT = 400
CHART_WIDTH = 800

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)