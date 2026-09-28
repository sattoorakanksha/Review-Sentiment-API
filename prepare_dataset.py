"""
Build a balanced, binary-labeled sample for fine-tuning.
Run from project root: python prepare_dataset.py
"""

import pandas as pd

df = pd.read_csv("data/raw/Reviews.csv")

# Drop the ambiguous 3-star "neutral" reviews
df = df[df["Score"] != 3].copy()

# Binary label: 1 = positive (4-5 stars), 0 = negative (1-2 stars)
df["label"] = (df["Score"] >= 4).astype(int)

negative = df[df["label"] == 0]
positive = df[df["label"] == 1]

print(f"Available negative reviews: {len(negative)}")
print(f"Available positive reviews: {len(positive)}")

SAMPLE_SIZE_PER_CLASS = 5000

negative_sample = negative.sample(n=SAMPLE_SIZE_PER_CLASS, random_state=42)
positive_sample = positive.sample(n=SAMPLE_SIZE_PER_CLASS, random_state=42)

balanced = pd.concat([negative_sample, positive_sample]).sample(frac=1, random_state=42)  # shuffle
balanced = balanced[["Text", "label"]].rename(columns={"Text": "text"})

balanced.to_csv("data/processed/balanced_sample.csv", index=False)

print(f"\nSaved {len(balanced)} balanced, shuffled reviews to data/processed/balanced_sample.csv")
print(f"Label distribution:\n{balanced['label'].value_counts()}")