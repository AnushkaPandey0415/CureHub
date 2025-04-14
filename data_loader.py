import os
import numpy as np
import pandas as pd
import pickle
from config import DATA_DIR, EMBEDDINGS_DIR, SAMPLE_SIZE
from logger import logger

def load_data():
    """Load and preprocess data"""
    cache_file = os.path.join(EMBEDDINGS_DIR, "data.pkl")
    train_path = os.path.join(DATA_DIR, "drugsComTrain_raw.csv")
    test_path = os.path.join(DATA_DIR, "drugsComTest_raw.csv")

    if os.path.exists(cache_file):
        logger.info("Loading cached data...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        logger.warning("Data files missing, creating synthetic data...")
        conditions = ["depression", "anxiety", "pain"]
        drugs = ["DrugA", "DrugB", "DrugC"]
        synthetic_data = {
            'condition': np.random.choice(conditions, SAMPLE_SIZE),
            'drugName': np.random.choice(drugs, SAMPLE_SIZE),
            'review': ["Sample review"] * SAMPLE_SIZE,
            'rating': np.random.randint(1, 11, SAMPLE_SIZE),
            'date': pd.date_range("2020-01-01", periods=SAMPLE_SIZE, freq="D")
        }
        train_df = pd.DataFrame(synthetic_data)
        test_df = train_df.sample(SAMPLE_SIZE // 10, random_state=42)
        patient_df = pd.DataFrame({
            'patient_id': range(100),
            'condition': np.random.choice(conditions, 100),
            'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], 100)
        })
    else:
        logger.info("Loading real data...")
        train_df = pd.read_csv(train_path).sample(SAMPLE_SIZE, random_state=42)
        test_df = pd.read_csv(test_path).sample(SAMPLE_SIZE // 10, random_state=42)
        patient_df = pd.DataFrame({
            'patient_id': range(100),
            'condition': np.random.choice(train_df['condition'].dropna().unique(), 100),
            'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], 100)
        })

    for df in [train_df, test_df]:
        df.fillna({'condition': '', 'review': '', 'drugName': ''}, inplace=True)
        df['text'] = (df['condition'] + " " + df['review']).str.lower().replace(r'[^a-z\s]', '', regex=True)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(5.0)
        df['date'] = pd.to_datetime(df['date'], format='%B %d, %Y', errors='coerce')
        df['recency'] = (pd.Timestamp('2025-01-01') - df['date']).dt.days.fillna(365).clip(0, 3650) / 3650

    patient_df['text'] = patient_df['condition'].str.lower().replace(r'[^a-z\s]', '', regex=True)

    with open(cache_file, 'wb') as f:
        pickle.dump((train_df, test_df, patient_df), f)

    return train_df, test_df, patient_df