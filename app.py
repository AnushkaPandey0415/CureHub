import streamlit as st
from frontend.dashboard import setup_dashboard
from frontend.data_loader import load_data
from frontend.styles import apply_dashboard_styles

def main():
    # Setup page configuration
    st.set_page_config(page_title="CureHub Insights Dashboard", layout="wide")
    
    # Apply custom CSS styles
    apply_dashboard_styles()
    
    # Set up dashboard header
    st.title("💊 CureHub Insights Dashboard")
    st.markdown("Explore **patient recommendations** and **model performance**.")
    
    # Load data
    profile_df, rec_df, metrics_df = load_data()
    
    # Check if data is loaded properly
    if profile_df.empty or profile_df.shape[1] < 2:
        st.warning("⚠️ Not enough data to visualize insights.")
        return
    
    # Setup the main dashboard
    setup_dashboard(profile_df, rec_df, metrics_df)

if __name__ == "__main__":
    main()