# Build Notes — FMCG Review Sentiment Classifier & API

Log real numbers and decisions here as they happen, not reconstructed later.

## Project Goal
Fine-tune a pretrained transformer (DistilBERT) for sentiment classification
on real customer food/grocery reviews, wrap it in a FastAPI service,
containerize with Docker, and deploy live. Directly addresses a limitation
found in the Receipt Intelligence project: a blank NER model (no pretrained
embeddings) failed to generalize on true held-out data. This project uses
a genuinely pretrained model instead, fine-tuned rather than trained from scratch.

## Data
- Source: Amazon Fine Food Reviews (Kaggle / Stanford SNAP)
- Real, publicly available dataset — not synthetic
- Total rows: 568,454
- Columns: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator,
  HelpfulnessDenominator, Score (1-5 stars), Time (Unix timestamp),
  Summary, Text
- Label source: Score (star rating) — Text is the model input

## Week 1 — Data Inspection & Preparation

### Score distribution (raw)
| Score | Count  | % |
|-------|--------|---|
| 1     | 52,268 | 9.2%  |
| 2     | 29,769 | 5.2%  |
| 3     | 42,640 | 7.5%  |
| 4     | 80,655 | 14.2% |
| 5     | 363,122| 63.9% |

- Severe class imbalance confirmed: 63.9% of all reviews are 5-star.
- A naive model predicting "positive" for everything would hit ~78%
  accuracy on a raw positive/negative split — meaningless without addressing
  the imbalance.

### Missing values
- ProfileName: 26 missing, Summary: 27 missing — neither column is used
  (Text and Score are the only relevant fields), so no cleanup needed.

### Text length
- Average: 436 characters (~80-100 words) — comfortably within DistilBERT's
  512-token limit
- Longest: 21,409 characters — real outlier tail; will need truncation
  handled explicitly in tokenization, not left to fail silently

### Key decisions made
1. **Dropped 3-star reviews (the "neutral" class).** Text sentiment at
   3 stars is genuinely ambiguous (could read positive, negative, or mixed)
   — going binary (positive/negative) is a cleaner, more defensible target
   than trying to also classify "neutral" from text alone.
2. **Binary label:** 1 = positive (Score 4-5), 0 = negative (Score 1-2)
3. **Balanced via undersampling, not class weighting.** With 568K rows
   available, took an equal random sample from each class rather than
   training on the raw imbalanced distribution.
4. **Sampled 5,000 reviews per class (10,000 total)**, not the full
   available pool (~82K negative reviews). Reasoning: practical training
   time on local hardware; 10K balanced, real reviews is a legitimate,
   sufficient sample for a fine-tuned model — documented here as a
   deliberate scoping choice, not a limitation to hide.

### Result
- Available before sampling: 82,037 negative reviews, 443,777 positive reviews
- Built data/processed/balanced_sample.csv: 10,000 rows, 5,000 per class,
  shuffled, columns [text, label]
- Confirmed balanced output: label 1 (positive) = 5,000, label 0 (negative) = 5,000



## Week 2 — (planned) Fine-tuning DistilBERT
## Week 2 — Fine-tuning Setup Note
- Initial time estimate for CPU fine-tuning was significantly off (expected
  20-60 min, actual progress bar estimated ~3.5 hours for 1596 steps /
  3 epochs on 8500 examples). CPU-only transformer fine-tuning is
  meaningfully slower than initially assumed — worth remembering for
  future project time estimates.
## Week 2 — Fine-tuning Results & Generalization Check

### Training
- Fine-tuned distilbert-base-uncased, 3 epochs, 8,500 training examples,
  1,500 validation examples (from the 10,000-review balanced sample)
- CPU-only training took significantly longer than initially estimated
  (~3+ hours, not the 20-60 min originally assumed)

### Validation results (same split used for model selection during training)
- Accuracy: 90.87% | Precision: 88.17% | Recall: 94.40% | F1: 91.18%

### True holdout results (2,000 fresh reviews, zero role in training OR
### validation/model-selection — drawn from the full 568K pool, excluding
### everything in the original 10K sample)
- Accuracy: 90.35% | Precision: 86.85% | Recall: 95.10% | F1: 90.79%

### Key finding
- Only a 0.52-point accuracy gap between validation and true holdout —
  essentially no overfitting detected. This is a direct, measured contrast
  to the Receipt Intelligence project's NER model, where a blank
  (non-pretrained) model showed a 20+ point gap between its dev split and
  true holdout data.
- Interpretation: fine-tuning a genuinely pretrained model (vs. training
  from scratch) produces meaningfully better generalization, exactly as
  the earlier project's finding predicted it should. This result is
  evidence for that lesson, not just a repeat of the same mistake.
## Week 3 — (planned) FastAPI service
## Week 3 — API Testing: A Real Limitation Found

- Tested /predict with 3 reviews: clear positive, ambiguous/mixed, short terse negative
- Clear cases (positive, short negative) classified correctly with high confidence
- Ambiguous case ("fine but arrived late, packaging was okay") was classified
  negative at 96.55% confidence — a human would likely call this mixed or
  mildly negative, not strongly negative
- Root cause: training data excluded all 3-star (neutral/mixed) reviews by
  design, so the model was never taught to recognize ambiguous sentiment —
  it can only choose between "clearly positive" and "clearly negative,"
  even when given text that's genuinely neither
- This is a known, deliberate scope limitation (binary classification was
  a conscious choice, documented in Week 1), not a bug — but worth stating
  explicitly: this model should not be trusted on mixed/lukewarm reviews,
  only on clearly polarized text
## Week 3 — CLOSED
- FastAPI service built and tested locally: GET / (info + limitation
  disclaimer) and POST /predict (sentiment classification) both confirmed
  working via Swagger UI
- Limitation note is embedded directly in the API's root response, not
  just documented separately — anyone calling the API sees the scope
  honestly before using /predict
- Ready for containerization (Week 4)
## Week 4 — (planned) Docker + deployment
## Week 4 — Docker: CLOSED
- Image built successfully after resolving two real issues: missing
  CPU-specific torch index (was downloading unnecessary multi-GB CUDA
  packages) and transient network instability (TLS handshake timeouts,
  fixed with retries + patience)
- Container run and verified via docker run -p 8000:8000 review-sentiment-api
- API confirmed working identically inside the container as it did locally
  (same /predict behavior, same limitation disclaimer on root endpoint)
- Project is now genuinely portable — runs the same way regardless of the
  host machine's Python/package setup
  ## Week 4 — Docker: Verified
- Tested /predict inside the container with the same "Absolutely loved
  this yogurt..." review used in the original local test
- Result: {"label": "positive", "confidence": 0.9876} — identical to the
  non-Docker result, confirming containerization changed nothing about
  model behavior, only how/where it runs
## Week 4 — GitHub Push: Bloated History Issue

- First push attempt failed (HTTP 408 timeout) trying to upload 2.02 GiB,
  despite currently-tracked files being clean and small
- Root cause: earlier commits in git history had included large files
  (likely training checkpoint folders) before .gitignore excluded them;
  .gitignore only prevents FUTURE commits from including files, it does
  NOT remove things already committed in history
- Fix: deleted the entire .git folder and reinitialized from scratch,
  since this is a personal project with no team history worth preserving
- Result: clean push, 268MB via Git LFS (the actual model) + 2.24 MiB for
  code/config — confirms the bloat was entirely leftover history, not
  anything currently needed
- Lesson: always check .gitignore is in place BEFORE the first commit,
  not after — cleaning up history later is possible (git filter-branch,
  BFG Repo-Cleaner) but a fresh start is simpler when no history needs saving
## Week 4 — CLOSED: Live Deployment

- Deployed to Render.com (free tier, 512MB RAM) after Hugging Face Spaces
  required a paid plan for Docker SDK
- No memory issues encountered despite the risk flagged beforehand —
  512MB was sufficient for DistilBERT + FastAPI + transformers in practice
- Live URL: https://review-sentiment-api-z97e.onrender.com
- Note: free tier spins down after ~15 min inactivity; first request after
  idle time will be slow (cold start, 30-60s) — expected, not a bug
- Full project now complete: fine-tuned pretrained model (validated via
  true holdout testing) -> FastAPI service -> Docker container -> live
  public deployment on GitHub 