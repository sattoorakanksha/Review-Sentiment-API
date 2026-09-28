"""
Build a genuinely independent test set (never touched during training OR
validation/model-selection) and evaluate the model on it.
Run from project root: python true_holdout_test.py
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_DIR = "model_output/final_model"

# Reload full raw data and the exact same balanced sample used for training,
# so we can exclude it entirely and draw from what's genuinely left over.
full_df = pd.read_csv("data/raw/Reviews.csv")
full_df = full_df[full_df["Score"] != 3].copy()
full_df["label"] = (full_df["Score"] >= 4).astype(int)

used_sample = pd.read_csv("data/processed/balanced_sample.csv")
# Exclude anything already used, matched on review text
holdout_pool = full_df[~full_df["Text"].isin(used_sample["text"])]

negative_pool = holdout_pool[holdout_pool["label"] == 0]
positive_pool = holdout_pool[holdout_pool["label"] == 1]

TEST_SIZE_PER_CLASS = 1000
neg_test = negative_pool.sample(n=TEST_SIZE_PER_CLASS, random_state=99)
pos_test = positive_pool.sample(n=TEST_SIZE_PER_CLASS, random_state=99)

test_df = pd.concat([neg_test, pos_test]).sample(frac=1, random_state=99)
test_df = test_df[["Text", "label"]].rename(columns={"Text": "text"})

test_df.to_csv("data/processed/true_holdout_test.csv", index=False)
print(f"Built true holdout set: {len(test_df)} reviews, never used in training or validation.")

# --- Evaluate ---
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
        predictions.extend(torch.argmax(outputs.logits, dim=1).tolist())
        if i % (batch_size * 5) == 0:
            print(f"  {i}/{len(texts)}")
    return predictions

print("\nRunning predictions on true holdout set...")
predictions = predict(test_df["text"].tolist())
labels = test_df["label"].tolist()

acc = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
cm = confusion_matrix(labels, predictions)

print(f"\n===== TRUE Holdout Results ({len(labels)} examples, never seen before) =====")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 score:  {f1*100:.2f}%")
print(f"\nConfusion matrix:")
print(f"                Predicted Neg   Predicted Pos")
print(f"Actual Neg      {cm[0][0]:<15} {cm[0][1]}")
print(f"Actual Pos      {cm[1][0]:<15} {cm[1][1]}")