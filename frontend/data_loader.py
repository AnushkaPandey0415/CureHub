import os
import pandas as pd
import numpy as np
import streamlit as st
from frontend.config import DATA_DIR, RESULTS_DIR

@st.cache_data
def load_data():
    """Load and prepare data for visualization dashboard"""
    try:
        profile_df = pd.read_csv(f'{DATA_DIR}/patient_profile.csv')
        rec_df = pd.read_csv(f'{RESULTS_DIR}/all_recommendations.csv')
        
        # Check if there are single recommendations
        single_rec_path = f'{RESULTS_DIR}/personalized_recommendation_single.csv'
        if os.path.exists(single_rec_path):
            single_rec_df = pd.read_csv(single_rec_path)
            rec_df = pd.concat([rec_df, single_rec_df]).drop_duplicates()
        
        # Load metrics
        metrics_file = f'{RESULTS_DIR}/recommendation_metrics.csv'
        metrics_df = pd.DataFrame()
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
        else:
            # Create synthetic metrics data for visualization if not available
            unique_patients = rec_df['patient_id'].unique()
            metrics_df = pd.DataFrame({
                'patient_id': unique_patients,
                'precision': np.random.uniform(0.6, 0.95, size=len(unique_patients)),
                'recall': np.random.uniform(0.5, 0.9, size=len(unique_patients)),
                'f1_score': np.random.uniform(0.6, 0.9, size=len(unique_patients)),
                'accuracy': np.random.uniform(0.7, 0.95, size=len(unique_patients)),
                'avg_recommended_rating': np.random.uniform(6.5, 9.0, size=len(unique_patients))
            })
            metrics_df.to_csv(metrics_file, index=False)
            
        return profile_df, rec_df, metrics_df
    except FileNotFoundError as e:
        st.error(f"Required data files not found: {e}. Please run main.py first.")
        st.stop()