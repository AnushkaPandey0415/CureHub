from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
from preprocess import load_and_preprocess

def train_diagnosis_model():
    print("📥 Loading and preprocessing training data...")
    df = load_and_preprocess("data/drugsComTrain_raw.csv")

    # Smart downsampling: use 30% for balance between speed + quality
    df = df.sample(frac=0.3, random_state=42)
    print(f"✅ Using {len(df)} rows for training.")

    X = df["review"]
    y = df["condition"]

    # Efficient vectorization: reduce size but retain bigrams
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    # Fast and scalable classifier
    model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    model.fit(X_vec, y)

    # Save model and vectorizer
    with open("models/diagnosis_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Diagnosis model trained and saved.")

if __name__ == "__main__":
    train_diagnosis_model()