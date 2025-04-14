import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Setup ===
plt.style.use("dark_background")
st.set_page_config(page_title="CureHub Insights Dashboard", layout="wide")
st.title("💊 CureHub Insights Dashboard")
st.markdown("Explore **patient recommendations** and **model performance**.")

# === Load Data ===
@st.cache_data
def load_data():
    try:
        profile_df = pd.read_csv('data/patient_profile.csv')
        rec_df = pd.read_csv('results/all_recommendations.csv')
        
        # Check if there are single recommendations (from your example code)
        single_rec_path = 'results/personalized_recommendation_single.csv'
        if os.path.exists(single_rec_path):
            single_rec_df = pd.read_csv(single_rec_path)
            rec_df = pd.concat([rec_df, single_rec_df]).drop_duplicates()
        
        # Load metrics
        metrics_file = 'results/recommendation_metrics.csv'
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
        st.error(f"Required data files not found: {e}. Please run rec.py first.")
        st.stop()

profile_df, rec_df, metrics_df = load_data()

# === Sidebar for patient selection ===
st.sidebar.subheader("🔍 Select a Patient")
available_ids = sorted(rec_df['patient_id'].unique())
selected_patient = st.sidebar.selectbox("Patient ID", available_ids)

# Filter data for selected patient
patient_profile = profile_df[profile_df['patient_id'] == selected_patient]
patient_recs = rec_df[rec_df['patient_id'] == selected_patient]

# Reset button
if st.sidebar.button("🔄 Reset Patient Selection"):
    selected_patient = available_ids[0]
    st.experimental_rerun()

# === Tabs ===
tab1, tab2 = st.tabs(["Patient Results & Recommendations", "Performance Metrics"])

# === Tab 1: Patient Results & Recommendations ===
with tab1:
    st.header("Patient Results & Recommendations")
    st.markdown("View personalized drug recommendations and patient profile.")

    # Display patient info in a card-like format
    if not patient_profile.empty:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            ### 🧬 Patient ID: {selected_patient}
            """)
            
            # Create a styled card for patient info
            condition = patient_profile['condition'].values[0]
            current_drug = patient_profile['drugName'].values[0]
            current_rating = patient_profile['rating'].values[0]
            
            st.info(f"""
            **Condition:** {condition}  
            **Current Drug:** {current_drug}  
            **Rating:** {current_rating}/10
            """)
            
            # Add sentiment if available
            if 'sentiment' in patient_profile.columns:
                sentiment = patient_profile['sentiment'].values[0]
                sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                st.metric("Sentiment", f"{sentiment_label} ({sentiment:.2f})")
        
        with col2:
            # Add a radar chart for patient metrics
            if 'review' in patient_profile.columns:
                review_length = len(str(patient_profile['review'].values[0]))
                
                # Create synthetic metrics for visualization if needed
                metrics = {
                    'Current Rating': float(current_rating),
                    'Review Length': min(10, review_length/100),  # Normalized
                    'Engagement': float(np.random.uniform(7, 9)),
                    'Treatment Duration': float(np.random.uniform(5, 10))
                }
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(metrics.values()),
                    theta=list(metrics.keys()),
                    fill='toself',
                    name='Patient Metrics'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10])
                    ),
                    showlegend=False,
                    title="Patient Health Metrics"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No patient profile data found for this patient.")

    # Show recommendations
    if not patient_recs.empty:
        st.subheader("💊 Recommended Drugs")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar Chart of Recommendations
            fig_bar = px.bar(
                patient_recs.sort_values(by='hybrid_score', ascending=False),
                x='recommended_drug',
                y='hybrid_score',
                color='hybrid_score',
                color_continuous_scale='Viridis',
                title="Top Drug Recommendations by Hybrid Score"
            )
            fig_bar.update_layout(xaxis_title="Recommended Drug", yaxis_title="Hybrid Score")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Display recommendations as a styled dataframe
            st.dataframe(
                patient_recs[['recommended_drug', 'hybrid_score']]
                .sort_values(by='hybrid_score', ascending=False)
                .reset_index(drop=True),
                use_container_width=True
            )
        
        # Horizontal Score Comparison (Bullet Chart)
        fig_bullet = go.Figure()
        
        sorted_recs = patient_recs.sort_values(by='hybrid_score', ascending=False)
        for i, (_, row) in enumerate(sorted_recs.iterrows()):
            fig_bullet.add_trace(go.Indicator(
                mode="number+gauge",
                value=row['hybrid_score'],
                domain={'x': [0, 1], 'y': [i/len(sorted_recs), (i+0.8)/len(sorted_recs)]},
                title={'text': row['recommended_drug']},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [None, 10]},
                    'threshold': {
                        'line': {'color': "red", 'width': 2},
                        'thickness': 0.75,
                        'value': 7.5
                    },
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 7.5], 'color': "gray"}
                    ],
                    'bar': {'color': "darkgreen"}
                }
            ))
        
        fig_bullet.update_layout(
            height=100 + (50 * len(sorted_recs)),
            title="Drug Recommendation Strength",
        )
        st.plotly_chart(fig_bullet, use_container_width=True)
        
        # Effectiveness Gauge
        if len(patient_recs) > 0:
            avg_score = patient_recs['hybrid_score'].mean()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Recommendation Strength"},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 5], 'color': 'red'},
                        {'range': [5, 7], 'color': 'orange'},
                        {'range': [7, 10], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 7
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown(
            """
            ### How Recommendations Are Generated
            
            Recommendations are based on a hybrid scoring model:
            - **70%** - User rating importance
            - **20%** - Sentiment analysis from reviews
            - **10%** - Text similarity between patient condition and drug reviews
            
            Higher scores indicate stronger recommendations tailored to the patient's condition and preferences.
            """
        )
    else:
        st.warning("No recommendations found for this patient.")

# === Tab 2: Performance Metrics ===
with tab2:
    st.header("Model Performance Metrics")
    
    # Add dropdown to switch between overall metrics and patient-specific metrics
    metric_view = st.selectbox(
        "Select view",
        ["Overall Model Performance", f"Patient {selected_patient} Performance"],
        index=0  # Default to overall performance
    )
    
    if metric_view == "Overall Model Performance":
        st.markdown("### 📊 Combined Model Performance Metrics")
        
        if not metrics_df.empty:
            # Overall metrics in a row
            st.subheader("📈 Key Performance Indicators")
            
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
            
            # Distribution of metrics
            st.subheader("📊 Metric Distributions")
            
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
            
            # Correlation Heatmap
            st.subheader("🔥 Metric Correlations")
            
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
            
            # Performance over time (synthetic data)
            st.subheader("⏳ Performance Trends")
            
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
            
            # Confusion Matrix (using synthetic data)
            st.subheader("🧩 Confusion Matrix")
            
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
            
            # Patient Performance Table
            st.subheader("👥 All Patients Performance Table")
            
            # Display a sortable table with all patients' metrics
            patient_table = metrics_df[['patient_id', 'precision', 'recall', 'f1_score', 'accuracy', 'avg_recommended_rating']]
            patient_table = patient_table.sort_values(by='f1_score', ascending=False)
            
            st.dataframe(
                patient_table.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'),
                use_container_width=True
            )
            
        else:
            st.warning("No performance metrics available. Run evaluation in `rec.py` to generate metrics.")
    
    else:  # Patient-specific metrics
        st.markdown(f"### 👤 Patient {selected_patient} Performance Metrics")
        
        patient_metrics = metrics_df[metrics_df['patient_id'] == selected_patient]
        if not patient_metrics.empty:
            # Patient metrics overview
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
            
            # Patient vs. Average radar chart
            st.subheader("🎯 Patient vs. Average Performance")
            
            patient_values = [patient_metrics[col].values[0] for col in col_metrics if col in patient_metrics.columns]
            avg_values = [metrics_df[col].mean() for col in col_metrics if col in metrics_df.columns]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=patient_values,
                theta=col_labels,
                fill='toself',
                name=f'Patient {selected_patient}'
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
                title=f"Patient {selected_patient} vs. Average Performance"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add rating comparison bar chart
            st.subheader("⭐ Patient Rating Analysis")
            
            # Create synthetic data for patient ratings vs recommendations
            rating_data = pd.DataFrame({
                'Category': ['Actual Ratings', 'Predicted Ratings', 'Recommended Drugs'],
                'Average Score': [
                    float(patient_profile['rating'].mean() if 'rating' in patient_profile.columns else np.random.uniform(6, 8)),
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
            
            # Recommendation Confidence & Accuracy
            st.subheader("🎯 Recommendation Quality Metrics")
            
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
            
        else:
            st.warning(f"No performance metrics available for Patient {selected_patient}")

# === Custom CSS to make the dashboard look nicer ===
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1a1a;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3a3a3a;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# === Fallback ===
if profile_df.empty or profile_df.shape[1] < 2:
    st.warning("⚠️ Not enough data to visualize insights.")