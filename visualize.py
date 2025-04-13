import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import os

# === Setup ===
plt.style.use("dark_background")
st.set_page_config(page_title="CureHub Insights Dashboard", layout="wide")
st.title("💊 CureHub Insights Dashboard")
st.markdown("Explore **relationships and patterns** in CureHub datasets.")

# === Load Data ===
profile_df = pd.read_csv('data/patient_profile.csv')
rec_df = pd.read_csv('results/personalized_recommendation_single.csv')

# Sidebar or dropdown to select patient
st.sidebar.subheader("🔍 Select a Patient to View Recommendations")
available_ids = rec_df['patient_id'].unique()
selected_patient = st.sidebar.selectbox("Patient ID", available_ids)

# Filter data for selected patient
patient_profile = profile_df[profile_df['patient_id'] == selected_patient]
patient_recs = rec_df[rec_df['patient_id'] == selected_patient]

# Reset option
if st.sidebar.button("🔄 Reset Patient Selection"):
    selected_patient = available_ids[0]
    st.experimental_rerun()

# Show patient condition & current drug
if not patient_profile.empty:
    condition = patient_profile['condition'].values[0]
    current_drug = patient_profile['drugName'].values[0]
    current_rating = patient_profile['rating'].values[0]
    
    st.markdown(f"### 🧬 Patient ID: {selected_patient}")
    st.write(f"**Condition:** {condition}")
    st.write(f"**Current Drug:** {current_drug} (Rating: {current_rating}/10)")
else:
    st.warning("No patient profile data found.")

# Show recommendations
if not patient_recs.empty:
    st.markdown("### 💊 Recommended Drugs (Sorted by Hybrid Score)")
    st.dataframe(patient_recs[['recommended_drug', 'hybrid_score']].sort_values(by='hybrid_score', ascending=False), use_container_width=True)

    # --- Bar Plot of Recommendations ---
    fig_bar = px.bar(
        patient_recs.sort_values(by='hybrid_score', ascending=True),
        x='hybrid_score',
        y='recommended_drug',
        orientation='h',
        color='hybrid_score',
        color_continuous_scale='Viridis',
        title="Top Drug Recommendations by Hybrid Score"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Radar Chart of Scores ---
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=patient_recs['hybrid_score'],
        theta=patient_recs['recommended_drug'],
        fill='toself',
        name='Hybrid Score'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(patient_recs['hybrid_score']) + 1])
        ),
        showlegend=False,
        title="Recommendation Spread (Radar Chart)"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

else:
    st.warning("No recommendations found for this patient.")

# === Sentiment Analysis ===
# Ensure Sentiment Exists
if 'review' in profile_df.columns and 'sentiment' not in profile_df.columns:
    profile_df['sentiment'] = profile_df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Sentiment by Condition
if 'condition' in profile_df.columns and 'sentiment' in profile_df.columns:
    st.subheader("🧠 Average Sentiment per Condition")
    # Filter conditions to show top 10 based on the number of reviews
    top_conditions = profile_df['condition'].value_counts().head(10).index.tolist()

    # Searchable dropdown for selecting condition
    selected_condition = st.selectbox(
        "Select Condition:", 
        options=top_conditions, 
        index=0,
        help="Select a condition to see its average sentiment score."
    )

    # Filter data for selected condition and display its average sentiment
    condition_sent = profile_df[profile_df['condition'] == selected_condition]['sentiment'].mean()
    st.write(f"Average Sentiment for '{selected_condition}': {condition_sent:.2f}")

    # Display average sentiment for top 10 conditions
    condition_sent_all = profile_df[profile_df['condition'].isin(top_conditions)].groupby('condition')['sentiment'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=condition_sent_all.values, y=condition_sent_all.index, palette="coolwarm", ax=ax)
    ax.set_title("Average Sentiment by Condition (Top 10)")
    ax.set_xlabel("Average Sentiment")
    ax.set_ylabel("Condition")
    st.pyplot(fig)

    st.markdown(
        "This chart shows the average sentiment scores for the top 10 conditions. "
        "You can also select a specific condition from the dropdown to view its sentiment."
    )

# === Sentiment vs Rating Scatter ===
if 'rating' in profile_df.columns and 'sentiment' in profile_df.columns:
    st.subheader("💥 Rating vs Sentiment")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=profile_df, x='rating', y='sentiment', hue='rating', palette='Spectral', alpha=0.6)
    ax.set_title("Correlation Between Patient Ratings and Sentiment")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Sentiment")
    st.pyplot(fig)

    st.markdown(
        "This scatter plot shows the relationship between patient ratings and sentiment. "
        "Each point represents a review, and the color gradient corresponds to the rating."
    )

# === Line Graph: Sentiment Trend Over Time ===
if 'date' in profile_df.columns and 'sentiment' in profile_df.columns:
    st.subheader("📈 Sentiment Trend Over Time")

    # Ensure date is in datetime format
    profile_df['date'] = pd.to_datetime(profile_df['date'], errors='coerce')
    profile_df['year_month'] = profile_df['date'].dt.to_period('M')
    
    sentiment_trend = profile_df.groupby('year_month')['sentiment'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sentiment_trend.plot(kind='line', color='orange', marker='o', ax=ax)
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sentiment")
    st.pyplot(fig)

    st.markdown(
        "This line graph shows how sentiment has evolved over time. "
        "Each point represents the average sentiment for a given month."
    )

# === Boxplot: Rating vs Drug (Top 10 only for readability) ===
if 'drugName' in profile_df.columns and 'rating' in profile_df.columns:
    top_drugs = profile_df['drugName'].value_counts().head(10).index.tolist()
    top_df = profile_df[profile_df['drugName'].isin(top_drugs)]
    st.subheader("📦 Rating Distribution for Top 10 Drugs")
    
    fig, ax = plt.subplots(figsize=(8, 6))  # Smaller boxplot
    sns.boxplot(data=top_df, x='rating', y='drugName', palette='Set3', ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Drug Name")
    st.pyplot(fig)

    st.markdown(
        "This boxplot shows the distribution of ratings for the top 10 drugs. "
        "Each box represents the spread of ratings for a drug, with the median shown as a line."
    )

# === Age vs Rating/Sentiment ===
if 'age' in profile_df.columns:
    if 'rating' in profile_df.columns:
        st.subheader("👵 Rating Across Age Groups")
        
        profile_df['age_group'] = pd.cut(profile_df['age'], bins=[0, 20, 35, 50, 65, 100], labels=["0-20", "21-35", "36-50", "51-65", "65+"])

        fig, ax = plt.subplots(figsize=(8, 6))  # Smaller boxplot
        sns.boxplot(data=profile_df, x='age_group', y='rating', palette='viridis', ax=ax)
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Rating")
        st.pyplot(fig)

        st.markdown(
            "This boxplot shows how ratings are distributed across different age groups. "
            "The spread of ratings may vary depending on the group."
        )

    if 'sentiment' in profile_df.columns:
        st.subheader("🧠 Sentiment Across Age Groups")

        fig, ax = plt.subplots(figsize=(8, 6))  # Smaller boxplot
        sns.boxplot(data=profile_df, x='age_group', y='sentiment', palette='magma', ax=ax)
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Sentiment")
        st.pyplot(fig)

        st.markdown(
            "This boxplot shows sentiment distribution across different age groups. "
            "Age may influence how patients feel about treatments."
        )

# === Gender Differences in Sentiment ===
if 'gender' in profile_df.columns and 'sentiment' in profile_df.columns:
    st.subheader("⚖️ Sentiment Score by Gender")

    fig, ax = plt.subplots(figsize=(8, 6))  # Smaller boxplot
    sns.violinplot(data=profile_df, x='gender', y='sentiment', palette='cool', ax=ax)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Sentiment")
    st.pyplot(fig)

    st.markdown(
        "This violin plot shows the distribution of sentiment scores by gender. "
        "It helps visualize if sentiment varies by gender."
    )

# === Multi-dimensional Insights ===
if {'condition', 'gender', 'rating'}.issubset(profile_df.columns):
    st.subheader("📊 Average Rating by Gender and Condition (Top 5 Conditions)")
    top_conditions = profile_df['condition'].value_counts().head(5).index.tolist()
    filtered_df = profile_df[profile_df['condition'].isin(top_conditions)]

    fig, ax = plt.subplots(figsize=(10, 6))  # Smaller boxplot
    sns.barplot(data=filtered_df, x='condition', y='rating', hue='gender', palette='Set2', ax=ax)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

    st.markdown(
        "This bar chart compares the average rating by gender across the top 5 most common conditions. "
        "It helps identify if ratings differ based on gender for specific conditions."
    )

# === Fallback ===
if profile_df.empty or profile_df.shape[1] < 2:
    st.warning("⚠️ Not enough data to visualize insights.")
