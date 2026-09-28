"""
Fine-tune DistilBERT for binary sentiment classification on the balanced
review sample. Run from project root: python train_model.py
"""

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "model_output"

# --- Load and split data ---
df = pd.read_csv("data/processed/balanced_sample.csv")

train_df, val_df = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=42
)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# --- Tokenize ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,  # covers the vast majority of reviews; long outliers get truncated
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# --- Load pretrained model, adapted for binary classification ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# --- Training setup ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    print(f"\nModel saved to {OUTPUT_DIR}/final_model")