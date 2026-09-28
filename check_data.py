"""
Quick first look at the real data before building anything.
Run from project root: python check_data.py
"""

import pandas as pd

df = pd.read_csv("data/raw/Reviews.csv")

print(f"Total rows: {len(df)}\n")

print("Score distribution:")
print(df["Score"].value_counts().sort_index())
print()
print("Score distribution (as %):")
print((df["Score"].value_counts(normalize=True).sort_index() * 100).round(1))

print(f"\nMissing values per column:")
print(df.isnull().sum())

print(f"\nAverage review text length (characters): {df['Text'].str.len().mean():.0f}")
print(f"Longest review (characters): {df['Text'].str.len().max()}")