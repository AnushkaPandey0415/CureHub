import streamlit as st
from frontend.config import CHART_THEME, COLOR_PALETTE, COLOR_MAP

def apply_dashboard_styles():
    """Apply custom CSS styling to the dashboard"""
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
        
        /* Card styling */
        .css-1r6slb0, .css-12w0qpk {
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Metric styling */
        .css-1r6slb0.e1tzin5v3 {
            background-color: #0e1117;
            border: 1px solid #333;
            padding: 10px 15px;
            border-radius: 5px;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff;
        }
        h3 {
            border-bottom: 1px solid #333;
            padding-bottom: 0.5rem;
        }
        
        /* Info containers */
        .stAlert {
            background-color: rgba(25, 118, 210, 0.15);
            border-left-color: #1976D2;
        }
    </style>
    """, unsafe_allow_html=True)

def set_plot_theme():
    """Set the theme for matplotlib plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use(CHART_THEME)
    sns.set_palette(COLOR_PALETTE)
    
def get_color_scale(metric_name):
    """Get appropriate color scale for different metrics"""
    if metric_name in COLOR_MAP:
        return COLOR_MAP[metric_name]
    return COLOR_MAP['precision']  # Default

def create_gauge_steps(low=0, medium=60, high=80, max_val=100):
    """Create gauge chart steps with color coding"""
    return [
        {'range': [0, low], 'color': 'red'},
        {'range': [low, medium], 'color': 'orange'},
        {'range': [medium, max_val], 'color': 'green'}
    ]

def format_patient_card(patient_id, condition, drug, rating, sentiment=None):
    """Create a styled HTML card for patient information"""
    sentiment_html = ""
    if sentiment is not None:
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        sentiment_color = "#4CAF50" if sentiment > 0 else "#F44336" if sentiment < 0 else "#9E9E9E"
        sentiment_html = f"""
        <div style="margin-top:10px; padding:5px; background-color:rgba({sentiment_color.lstrip('#')}, 0.1); border-radius:5px;">
            <span style="color:{sentiment_color}">● {sentiment_label}</span>
            <span style="float:right; font-weight:bold;">{sentiment:.2f}</span>
        </div>
        """
        
    return f"""
    <div style="background-color:#2d2d2d; border-radius:10px; padding:15px; box-shadow:0 4px 8px rgba(0,0,0,0.2);">
        <h3 style="margin-top:0; border-bottom:1px solid #444; padding-bottom:10px;">🧬 Patient ID: {patient_id}</h3>
        <div style="margin:15px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <strong>Condition:</strong>
                <span>{condition}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <strong>Current Drug:</strong>
                <span>{drug}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <strong>Rating:</strong>
                <span>{rating}/10</span>
            </div>
        </div>
        {sentiment_html}
    </div>
    """