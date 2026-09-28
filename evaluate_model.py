"""
Evaluate the fine-tuned model on the validation set with real metrics
(accuracy, precision, recall, F1) — not just loss.
Run from project root: python evaluate_model.py
"""

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_DIR = "model_output/final_model"

# Rebuild the same validation split used during training (same random_state)
df = pd.read_csv("data/processed/balanced_sample.csv")
train_df, val_df = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict(texts, batch_size=32):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        predictions.extend(preds)
        if i % (batch_size * 5) == 0:
            print(f"  {i}/{len(texts)}")
    return predictions

print("Running predictions on validation set...")
val_texts = val_df["text"].tolist()
val_labels = val_df["label"].tolist()
predictions = predict(val_texts)

acc = accuracy_score(val_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(val_labels, predictions, average="binary")
cm = confusion_matrix(val_labels, predictions)

print(f"\n===== Validation Results ({len(val_labels)} examples) =====")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 score:  {f1*100:.2f}%")
print(f"\nConfusion matrix:")
print(f"                Predicted Neg   Predicted Pos")
print(f"Actual Neg      {cm[0][0]:<15} {cm[0][1]}")
print(f"Actual Pos      {cm[1][0]:<15} {cm[1][1]}")