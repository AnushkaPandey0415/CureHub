import streamlit as st
from metrics_loader import load_metrics, load_predictions
from plots import plot_confusion_matrix, plot_top_misclassifications

st.set_page_config(page_title="Drug Recommender Model Dashboard", layout="wide")

st.title("💊 Drug Recommendation Model Evaluation")

metrics = load_metrics()
y_true, y_pred = load_predictions()

st.header("📈 Performance Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    st.metric("Precision (weighted)", f"{metrics['precision']:.4f}")
with col2:
    st.metric("Recall (weighted)", f"{metrics['recall']:.4f}")
    st.metric("F1 Score (weighted)", f"{metrics['f1_score']:.4f}")

st.header("🔍 Confusion Matrix")
plot_confusion_matrix(y_true, y_pred)

st.header("❗ Top Misclassifications")
plot_top_misclassifications(y_true, y_pred)