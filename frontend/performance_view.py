import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from frontend.metrics import (
    display_overall_kpis, 
    display_metric_distributions, 
    display_correlation_heatmap, 
    display_performance_trends,
    display_confusion_matrix_and_roc,
    display_patient_metrics_table,
    display_patient_specific_metrics,
    display_patient_vs_avg_radar,
    display_patient_rating_analysis,
    display_recommendation_quality
)

def display_overall_performance(metrics_df):
    """Display overall model performance metrics and visualizations"""
    st.markdown("### 📊 Combined Model Performance Metrics")
    
    if metrics_df.empty:
        st.warning("No performance metrics available. Run evaluation in `main.py` to generate metrics.")
        return
        
    # Overall metrics in a row
    st.subheader("📈 Key Performance Indicators")
    display_overall_kpis(metrics_df)
    
    # Distribution of metrics
    st.subheader("📊 Metric Distributions")
    display_metric_distributions(metrics_df)
    
    # Correlation Heatmap
    st.subheader("🔥 Metric Correlations")
    display_correlation_heatmap(metrics_df)
    
    # Performance over time
    st.subheader("⏳ Performance Trends")
    display_performance_trends()
    
    # Confusion Matrix & ROC
    st.subheader("🧩 Confusion Matrix")
    display_confusion_matrix_and_roc()
    
    # Patient Performance Table
    st.subheader("👥 All Patients Performance Table")
    display_patient_metrics_table(metrics_df)

def display_patient_performance(selected_patient, metrics_df):
    """Display patient-specific performance metrics"""
    st.markdown(f"### 👤 Patient {selected_patient} Performance Metrics")
    
    patient_metrics = metrics_df[metrics_df['patient_id'] == selected_patient]
    if patient_metrics.empty:
        st.warning(f"No performance metrics available for Patient {selected_patient}")
        return
        
    # Patient metrics overview
    display_patient_specific_metrics(patient_metrics, metrics_df)
    
    # Patient vs. Average radar chart
    st.subheader("🎯 Patient vs. Average Performance")
    display_patient_vs_avg_radar(patient_metrics, metrics_df)
    
    # Add rating comparison bar chart
    st.subheader("⭐ Patient Rating Analysis")
    display_patient_rating_analysis(selected_patient, patient_metrics)
    
    # Recommendation Confidence & Accuracy
    st.subheader("🎯 Recommendation Quality Metrics")
    display_recommendation_quality(patient_metrics)

def display_performance_tab(selected_patient, metrics_df):
    """Display performance metrics tab with view selection"""
    st.header("Model Performance Metrics")
    
    # Add dropdown to switch between overall metrics and patient-specific metrics
    metric_view = st.selectbox(
        "Select view",
        ["Overall Model Performance", f"Patient {selected_patient} Performance"],
        index=0  # Default to overall performance
    )
    
    if metric_view == "Overall Model Performance":
        display_overall_performance(metrics_df)
    else:  # Patient-specific metrics
        display_patient_performance(selected_patient, metrics_df)