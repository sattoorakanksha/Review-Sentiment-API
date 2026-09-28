"""
FastAPI service serving the fine-tuned sentiment model.
Run locally with: uvicorn app:app --reload
Then visit http://127.0.0.1:8000/docs for an interactive test UI.
"""

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "model_output/final_model"

app = FastAPI(title="FMCG Review Sentiment API")

# Load model and tokenizer ONCE at startup, not per-request — reloading
# a transformer model on every request would be extremely slow.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


class ReviewRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    confidence: float


@app.get("/")
def root():
    return {
        "message": "FMCG Review Sentiment API is running. POST to /predict to classify a review.",
        "note": "This model performs binary classification (positive/negative) only. "
                "It was trained exclusively on clearly polarized reviews (1-2 star vs. 4-5 star) "
                "and does not recognize neutral or mixed sentiment — ambiguous reviews will be "
                "forced into one of the two categories, sometimes with high confidence despite "
                "genuine ambiguity in the text.",
    }


@app.post("/predict", response_model=SentimentResponse)
def predict(review: ReviewRequest):
    inputs = tokenizer(
        review.text, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_class = torch.argmax(probs).item()
    confidence = probs[predicted_class].item()

    label = "positive" if predicted_class == 1 else "negative"

    return SentimentResponse(label=label, confidence=round(confidence, 4))