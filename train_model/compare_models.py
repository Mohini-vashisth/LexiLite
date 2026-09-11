import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np

def load_data():
    df = pd.read_csv(os.path.join("data", "legal_clauses_labeled.csv")).rename(columns={"label": "risk"})
    df.drop_duplicates(subset=["clause"], inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df["clause"].tolist(), df["risk"].tolist()

def encode_with_sbert(clauses):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(clauses)

def encode_with_roberta(clauses, batch_size=8):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    model.eval()
    all_embeddings = []

    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings).numpy()

def evaluate_model(name, X, y, model_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = joblib.load(os.path.join(model_path, "clause_classifier.pkl"))
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, digits=3)
    print(f"\n📊 {name} Evaluation Report:")
    print(classification_report(y_test, y_pred, digits=3))
    return report

def plot_comparison(sbert_scores, roberta_scores):
    categories = ["precision", "recall", "f1-score"]
    classes = ["risky", "safe"]
    bar_width = 0.35
    x = np.arange(len(categories))

    for cls in classes:
        sbert_vals = [sbert_scores[cls][m] for m in categories]
        roberta_vals = [roberta_scores[cls][m] for m in categories]

        plt.figure(figsize=(8, 4))
        plt.bar(x - bar_width/2, sbert_vals, bar_width, label="Sentence-BERT")
        plt.bar(x + bar_width/2, roberta_vals, bar_width, label="RoBERTa")

        plt.xticks(x, categories)
        plt.ylim(0, 1.05)
        plt.title(f"Comparison for class: {cls}")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("📥 Loading data...")
    X_texts, y_labels = load_data()

    print("\n🔎 Encoding with Sentence-BERT...")
    sbert_embeddings = encode_with_sbert(X_texts)
    sbert_report = evaluate_model("Sentence-BERT", sbert_embeddings, y_labels, "saved_model")

    print("\n🔎 Encoding with RoBERTa...")
    roberta_embeddings = encode_with_roberta(X_texts)
    roberta_report = evaluate_model("RoBERTa", roberta_embeddings, y_labels, "saved_model_roberta")

    print("\n📊 Plotting comparison...")
    plot_comparison(sbert_report, roberta_report)