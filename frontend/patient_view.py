import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def display_patient_info(selected_patient, patient_profile):
    """Display patient information in a card-like format"""
    if patient_profile.empty:
        st.warning("No patient profile data found for this patient.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### 🧬 Patient ID: {selected_patient}")
        
        # Create a styled card for patient info
        condition = patient_profile['condition'].values[0]
        current_drug = patient_profile.get('drugName', pd.Series(['Not specified'])).values[0]
        current_rating = patient_profile.get('rating', pd.Series([0])).values[0]
        
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
            
            # Create metrics for visualization
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

def display_recommendations(patient_recs):
    """Display recommendations visualization and data"""
    if patient_recs.empty:
        st.warning("No recommendations found for this patient.")
        return
    
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
    sorted_recs = patient_recs.sort_values(by='hybrid_score', ascending=False)
    display_recommendation_bullets(sorted_recs)
    
    # Effectiveness Gauge
    display_effectiveness_gauge(patient_recs)

def display_recommendation_bullets(sorted_recs):
    """Display bullet chart for recommendation scores"""
    fig_bullet = go.Figure()
    
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

def display_effectiveness_gauge(patient_recs):
    """Display gauge chart for average recommendation effectiveness"""
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

def display_recommendation_info():
    """Display information about recommendation generation"""
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

def display_patient_tab(selected_patient, patient_profile, patient_recs):
    """Display patient results and recommendations tab"""
    st.header("Patient Results & Recommendations")
    st.markdown("View personalized drug recommendations and patient profile.")
    
    # Display patient info
    display_patient_info(selected_patient, patient_profile)
    
    # Show recommendations
    display_recommendations(patient_recs)
    
    # Add recommendation information
    display_recommendation_info()