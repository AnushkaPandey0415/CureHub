import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from frontend.config import COLOR_PALETTE, COLOR_MAP, CHART_HEIGHT, CHART_WIDTH

def create_radar_chart(categories, values, title, max_value=10, name="Metrics"):
    """Create a radar chart from categories and values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_value])
        ),
        showlegend=False,
        title=title,
        height=CHART_HEIGHT
    )
    
    return fig

def create_multi_radar_chart(categories, data_dict, title, max_value=1):
    """Create a radar chart with multiple data series"""
    fig = go.Figure()
    
    for name, values in data_dict.items():
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_value])
        ),
        showlegend=True,
        title=title,
        height=CHART_HEIGHT
    )
    
    return fig

def create_bar_chart(df, x_col, y_col, title, color_col=None, color_continuous_scale=COLOR_PALETTE):
    """Create a bar chart from dataframe columns"""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col else y_col,
        color_continuous_scale=color_continuous_scale,
        title=title,
        text_auto=True
    )
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        height=CHART_HEIGHT
    )
    
    return fig

def create_bullet_charts(df, name_col, value_col, max_value=10, title="Comparison"):
    """Create bullet charts for comparing values"""
    sorted_df = df.sort_values(by=value_col, ascending=False)
    
    fig = go.Figure()
    
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=row[value_col],
            domain={'x': [0, 1], 'y': [i/len(sorted_df), (i+0.8)/len(sorted_df)]},
            title={'text': row[name_col]},
            gauge={
                'shape': "bullet",
                'axis': {'range': [None, max_value]},
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': max_value * 0.75
                },
                'steps': [
                    {'range': [0, max_value * 0.5], 'color': "lightgray"},
                    {'range': [max_value * 0.5, max_value * 0.75], 'color': "gray"}
                ],
                'bar': {'color': "darkgreen"}
            }
        ))
    
    fig.update_layout(
        height=100 + (50 * len(sorted_df)),
        title=title,
    )
    
    return fig

def create_gauge_chart(value, title, max_value=10, color="darkblue"):
    """Create a gauge chart for a single value"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'red'},
                {'range': [max_value * 0.5, max_value * 0.7], 'color': 'orange'},
                {'range': [max_value * 0.7, max_value], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.7
            }
        }
    ))
    
    fig.update_layout(height=CHART_HEIGHT - 100)
    return fig

def create_distribution_plots(df, id_col, value_cols, metric_name='Metric', value_name='Value'):
    """Create box and violin plots for distribution analysis"""
    # Create a long-format dataframe for distribution plots
    dist_df = pd.melt(
        df, 
        id_vars=[id_col], 
        value_vars=value_cols,
        var_name=metric_name, 
        value_name=value_name
    )
    
    # Box plot
    box_fig = px.box(
        dist_df, 
        x=metric_name, 
        y=value_name,
        color=metric_name,
        points='all',
        title=f"Distribution of {metric_name}s"
    )
    
    # Violin plot
    violin_fig = px.violin(
        dist_df,
        x=metric_name,
        y=value_name,
        color=metric_name,
        box=True,
        title=f"{metric_name} Distribution Details"
    )
    
    return box_fig, violin_fig

def create_correlation_heatmap(df, cols):
    """Create a correlation heatmap for specified columns"""
    # Ensure all required columns exist
    cols_present = [col for col in cols if col in df.columns]
    
    if len(cols_present) < 2:
        return None
    
    corr = df[cols_present].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='viridis', ax=ax)
    ax.set_title("Correlation Between Metrics")
    
    return fig

def create_line_chart(df, x_col, y_cols, title, markers=True):
    """Create a line chart for trend analysis"""
    # Convert to long format if needed
    if isinstance(y_cols, list) and len(y_cols) > 1:
        df_long = pd.melt(
            df, 
            id_vars=[x_col], 
            value_vars=y_cols,
            var_name='Metric', 
            value_name='Value'
        )
        
        fig = px.line(
            df_long,
            x=x_col,
            y='Value',
            color='Metric',
            title=title,
            markers=markers
        )
    else:
        y_col = y_cols[0] if isinstance(y_cols, list) else y_cols
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title,
            markers=markers
        )
    
    return fig

def create_confusion_matrix(cm, class_names=['Negative', 'Positive']):
    """Create a confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    return fig

def create_roc_curve(fpr, tpr, auc_score=None):
    """Create an ROC curve visualization"""
    if auc_score is None:
        auc_score = np.trapz(tpr, fpr)  # Approximate AUC
    
    fig = px.line(
        x=fpr, y=tpr,
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        title=f'ROC Curve (Area Under Curve: {auc_score:.2f})'
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    return fig

def create_styled_dataframe(df, highlight_cols=None):
    """Create a styled dataframe with highlights for min/max values"""
    if highlight_cols is None:
        highlight_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    styled_df = df.style
    
    for col in highlight_cols:
        if col in df.columns:
            styled_df = styled_df.highlight_max(subset=[col], color='lightgreen')
            styled_df = styled_df.highlight_min(subset=[col], color='lightcoral')
    
    return styled_df

def create_metrics_row(metrics_dict, delta_dict=None):
    """Create a row of metric displays with optional delta values"""
    cols = st.columns(len(metrics_dict))
    
    for i, (metric_name, value) in enumerate(metrics_dict.items()):
        delta = None
        delta_color = "normal"
        
        if delta_dict and metric_name in delta_dict:
            delta = delta_dict[metric_name]
            if isinstance(delta, (int, float)):
                delta = f"{delta:.2f}"
                delta_color = "normal" if float(delta) >= 0 else "inverse"
        
        cols[i].metric(
            metric_name, 
            f"{value:.2f}" if isinstance(value, (int, float)) else value, 
            delta,
            delta_color=delta_color
        )

def create_recommendation_info():
    """Create an informative text block about the recommendation process"""
    return st.markdown(
        """
        ### How Recommendations Are Generated
        
        Recommendations are based on a hybrid scoring model:
        - **70%** - User rating importance
        - **20%** - Sentiment analysis from reviews
        - **10%** - Text similarity between patient condition and drug reviews
        
        Higher scores indicate stronger recommendations tailored to the patient's condition and preferences.
        """
    )