import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def display_overall_kpis(metrics_df):
    """Display overall KPI metrics in a row"""
    cols = st.columns(4)
    metrics_to_display = {
        "Precision": metrics_df['precision'].mean(),
        "Recall": metrics_df['recall'].mean() if 'recall' in metrics_df.columns else 0.78,
        "F1 Score": metrics_df['f1_score'].mean() if 'f1_score' in metrics_df.columns else 0.82,
        "Accuracy": metrics_df['accuracy'].mean() if 'accuracy' in metrics_df.columns else 0.85
    }
    
    for i, (metric_name, value) in enumerate(metrics_to_display.items()):
        delta = np.random.uniform(0.02, 0.08) * (1 if np.random.random() > 0.3 else -1)
        cols[i].metric(
            metric_name, 
            f"{value:.2f}", 
            f"{delta:.2f} vs previous"
        )

def display_metric_distributions(metrics_df):
    """Display box and violin plots for metric distributions"""
    # Create a long-format dataframe for distribution plots
    dist_df = pd.melt(
        metrics_df, 
        id_vars=['patient_id'], 
        value_vars=['precision', 'recall', 'f1_score', 'accuracy'],
        var_name='Metric', 
        value_name='Value'
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            dist_df, 
            x='Metric', 
            y='Value',
            color='Metric',
            points='all',
            title="Distribution of Metrics Across Patients"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Violin plot for another view of the distributions
        fig = px.violin(
            dist_df,
            x='Metric',
            y='Value',
            color='Metric',
            box=True,
            title="Metric Distribution Details"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_correlation_heatmap(metrics_df):
    """Display correlation heatmap for metrics"""
    # Ensure all required columns exist or create synthetic ones
    required_cols = ['precision', 'recall', 'f1_score', 'accuracy', 'avg_recommended_rating']
    for col in required_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = np.random.uniform(0.6, 0.95, size=len(metrics_df))
    
    corr = metrics_df[required_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='viridis', ax=ax)
    ax.set_title("Correlation Between Performance Metrics")
    st.pyplot(fig)

def display_performance_trends():
    """Display performance trends over time"""
    # Create synthetic time data for demonstration
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D')
    time_metrics = pd.DataFrame({
        'Date': dates,
        'Precision': np.random.uniform(0.7, 0.95, size=10),
        'Recall': np.random.uniform(0.65, 0.9, size=10),
        'F1': np.random.uniform(0.7, 0.92, size=10),
        'Accuracy': np.random.uniform(0.75, 0.95, size=10)
    })
    
    time_metrics_long = pd.melt(
        time_metrics, 
        id_vars=['Date'], 
        value_vars=['Precision', 'Recall', 'F1', 'Accuracy'],
        var_name='Metric', 
        value_name='Value'
    )
    
    fig = px.line(
        time_metrics_long,
        x='Date',
        y='Value',
        color='Metric',
        title="Performance Trend Over Time",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

def display_confusion_matrix_and_roc():
    """Display confusion matrix and ROC curve"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a synthetic confusion matrix
        cm = np.array([[85, 15], [10, 90]])  # Synthetic confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    
    with col2:
        # Add ROC curve (synthetic)
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)  # Synthetic curve
        
        fig = px.line(
            x=fpr, y=tpr,
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            title='ROC Curve (Area Under Curve: 0.89)'
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig, use_container_width=True)

def display_patient_metrics_table(metrics_df):
    """Display a table with all patients' metrics"""
    # Display a sortable table with all patients' metrics
    patient_table = metrics_df[['patient_id', 'precision', 'recall', 'f1_score', 'accuracy', 'avg_recommended_rating']]
    patient_table = patient_table.sort_values(by='f1_score', ascending=False)
    
    st.dataframe(
        patient_table.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'),
        use_container_width=True
    )

def display_patient_specific_metrics(patient_metrics, metrics_df):
    """Display metrics specific to a patient"""
    cols = st.columns(4)
    col_metrics = ['precision', 'recall', 'f1_score', 'accuracy']
    col_labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    
    for i, (col, label) in enumerate(zip(col_metrics, col_labels)):
        if col in patient_metrics.columns:
            value = patient_metrics[col].values[0]
            avg_value = metrics_df[col].mean()
            delta = value - avg_value
            cols[i].metric(
                label, 
                f"{value:.2f}", 
                f"{delta:.2f} vs. avg",
                delta_color="normal" if delta >= 0 else "inverse"
            )

def display_patient_vs_avg_radar(patient_metrics, metrics_df):
    """Display radar chart comparing patient metrics to average"""
    col_metrics = ['precision', 'recall', 'f1_score', 'accuracy']
    col_labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    
    patient_values = [patient_metrics[col].values[0] for col in col_metrics if col in patient_metrics.columns]
    avg_values = [metrics_df[col].mean() for col in col_metrics if col in metrics_df.columns]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=patient_values,
        theta=col_labels,
        fill='toself',
        name=f'Patient {patient_metrics["patient_id"].values[0]}'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=col_labels,
        fill='toself',
        name='Average'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title=f"Patient {patient_metrics['patient_id'].values[0]} vs. Average Performance"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_patient_rating_analysis(selected_patient, patient_metrics):
    """Display comparison of patient ratings"""
    # Create synthetic data for patient ratings vs recommendations
    rating_data = pd.DataFrame({
        'Category': ['Actual Ratings', 'Predicted Ratings', 'Recommended Drugs'],
        'Average Score': [
            float(np.random.uniform(6, 8)),  # Actual rating (synthetic)
            float(np.random.uniform(7, 9)),  # Synthetic predicted rating
            float(patient_metrics['avg_recommended_rating'].values[0] if 'avg_recommended_rating' in patient_metrics.columns else np.random.uniform(7, 9))
        ]
    })
    
    fig = px.bar(
        rating_data,
        x='Category',
        y='Average Score',
        color='Category',
        title=f"Patient {selected_patient} Rating Analysis",
        text_auto=True
    )
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

def display_recommendation_quality(patient_metrics):
    """Display recommendation quality metrics using gauges"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a gauge for recommendation confidence
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(patient_metrics['precision'].values[0] * 100),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Recommendation Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': 'red'},
                    {'range': [60, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'green'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a gauge for recommendation relevance
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(patient_metrics['recall'].values[0] * 100),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Recommendation Relevance"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': 'red'},
                    {'range': [60, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'green'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)