import streamlit as st
from frontend.patient_view import display_patient_tab
from frontend.performance_view import display_performance_tab

def setup_sidebar(rec_df):
    """Configure sidebar with patient selection controls"""
    st.sidebar.subheader("🔍 Select a Patient")
    available_ids = sorted(rec_df['patient_id'].unique())
    selected_patient = st.sidebar.selectbox("Patient ID", available_ids)
    
    # Reset button
    if st.sidebar.button("🔄 Reset Patient Selection"):
        selected_patient = available_ids[0]
        st.experimental_rerun()
        
    return selected_patient

def setup_dashboard(profile_df, rec_df, metrics_df):
    """Set up main dashboard with tabs and content"""
    # Setup sidebar
    selected_patient = setup_sidebar(rec_df)
    
    # Filter data for selected patient 
    patient_profile = profile_df[profile_df['patient_id'] == selected_patient]
    patient_recs = rec_df[rec_df['patient_id'] == selected_patient]
    
    # Create main tabs
    tab1, tab2 = st.tabs(["Patient Results & Recommendations", "Performance Metrics"])
    
    # Display content for each tab
    with tab1:
        display_patient_tab(selected_patient, patient_profile, patient_recs)
    
    with tab2:
        display_performance_tab(selected_patient, metrics_df)