# FMCG Review Sentiment API

A fine-tuned transformer model that classifies customer product reviews as
positive or negative, served as a containerized REST API. Built to address
a specific limitation found in an earlier project: a model trained from
scratch (no pretrained language knowledge) failed to generalize to new data.
This project fine-tunes a genuinely pretrained model instead, and validates
that choice with a true independent holdout test.

## Problem

Companies — particularly FMCG/food & grocery brands — receive large volumes
of customer reviews. Manually reading through them to gauge sentiment
doesn't scale. This project builds a real, callable sentiment classification
service that could plug into a review-monitoring pipeline for exactly that
kind of business.

## Data

- **Source:** Amazon Fine Food Reviews (Kaggle / Stanford SNAP) — genuinely
  real, anonymized customer review data, not synthetic
- **Total available:** 568,454 reviews
- **Used for training:** a balanced sample of 10,000 reviews (5,000 positive,
  5,000 negative), deliberately scoped for practical CPU training time —
  see "Design Decisions" below
- **Label:** binary sentiment derived from star rating (1-2 stars = negative,
  4-5 stars = positive; 3-star/neutral reviews excluded — see limitations)

## Model & Approach

- **Base model:** `distilbert-base-uncased` (pretrained), fine-tuned for
  binary sequence classification
- **Training:** 3 epochs, 8,500 training examples, 1,500 validation examples
- **Why fine-tuning, not training from scratch:** a prior project's NER
  model, trained from a blank state with no pretrained embeddings, showed a
  20+ point accuracy gap between its validation split and a true independent
  test set. This project deliberately fine-tunes a pretrained model instead,
  to test whether that fixes the generalization problem.

## Results

### Validation (same split used during training/model selection)
| Metric | Score |
|--------|-------|
| Accuracy | 90.87% |
| Precision | 88.17% |
| Recall | 94.40% |
| F1 | 91.18% |

### True Holdout (2,000 reviews, zero role in training or validation —
drawn from the full 568K pool, excluding everything in the training sample)
| Metric | Score |
|--------|-------|
| Accuracy | 90.35% |
| Precision | 86.85% |
| Recall | 95.10% |
| F1 | 90.79% |

**Key finding:** only a 0.52-point accuracy gap between validation and true
holdout — a direct, measured contrast to the earlier NER project's 20+ point
gap. This is evidence that fine-tuning a pretrained model produces
meaningfully better generalization than training from scratch, not just a
theoretical claim.

## API

Built with FastAPI. Two endpoints:
- `GET /` — service info and an explicit limitation disclaimer
- `POST /predict` — takes `{"text": "..."}`, returns `{"label": "positive"|"negative", "confidence": 0.0-1.0}`

**Live demo:**(https://review-sentiment-api-f8dq.onrender.com/docs)

## Known Limitation

This model performs **binary classification only**. Since training data
excluded all neutral/mixed (3-star) reviews by design, the model has never
learned to recognize genuine ambiguity — it will force any review, however
mixed or lukewarm, into "positive" or "negative," sometimes with high
confidence despite real ambiguity in the text. Example: "The product was
fine but arrived a bit late, packaging was okay" was classified negative
at 96.6% confidence, despite reading as mixed/neutral to a human. This is
a deliberate scope limitation, disclosed directly in the API's root
response, not a hidden flaw.

## Design Decisions

- **10,000-review training sample, not the full 568K available:** fine-tuning
  a pretrained model needs far less data than training from scratch, since
  the model already understands language structure. This kept CPU training
  time practical without meaningfully limiting performance.
- **Binary sentiment, not 3-class:** 3-star reviews are genuinely ambiguous
  in text, making them a much harder, murkier target than clearly polarized
  reviews. Binary framing is a standard, defensible simplification.
- **CPU-only training and inference:** no GPU dependency, making this
  reproducible on ordinary hardware — at the cost of slower training
  (~3 hours for this fine-tuning run).

## Tech Stack
Python, PyTorch, Hugging Face Transformers, FastAPI, Docker, DistilBERT

## How to Run

**Locally:**
```bash
pip install -r requirements.txt
python train_model.py       # or skip if using the included fine-tuned model
uvicorn app:app --reload
```

**With Docker:**
```bash
docker build -t review-sentiment-api .
docker run -p 8000:8000 review-sentiment-api
```

Visit `http://127.0.0.1:8000/docs` for an interactive test UI.
