import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter

def plot_confusion_matrix(y_true, y_pred, max_classes=10):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    
    if len(labels) > max_classes:
        top_labels = Counter(y_true).most_common(max_classes)
        top_indices = [labels.index(label) for label, _ in top_labels]
        cm = cm[top_indices][:, top_indices]
        labels = [label for label, _ in top_labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def plot_top_misclassifications(y_true, y_pred, top_n=10):
    df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
    misclassified = df[df['actual'] != df['predicted']]
    common_misclassifications = misclassified.groupby(['actual', 'predicted']).size().reset_index(name='count')
    top_misclassified = common_misclassifications.sort_values(by='count', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_misclassified, x="count", y="actual", hue="predicted", ax=ax)
    ax.set_title("Top Misclassifications")
    st.pyplot(fig)