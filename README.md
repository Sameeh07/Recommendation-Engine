# SHL Assessment Recommendation Engine

A hybrid recommendation system for SHL product assessments that combines rule-based filtering, text similarity, and machine learning to recommend relevant assessments based on hiring requirements.

##  Overview

This system recommends SHL assessments by analyzing job requirements (role, level, skills, hiring stage) and matching them against the SHL product catalog using a multi-stage pipeline.

**Data Source:** [SHL Product Catalog](https://www.shl.com/products/product-catalog/)  
**Catalog Size:** 24 SHL assessments with skills, job levels, categories, descriptions, and duration.

## How It Works

### 1. Data Preprocessing
- Load SHL catalog from JSON (nested structure: `shl_product_catalog.products`)
- Parse skills from comma-separated strings into lists
- Combine skills, description, and category into `combined_text` field for each assessment
- Normalize text (lowercase, remove punctuation, collapse whitespace)

### 2. Rule-Based Filtering
- Apply business logic from `config/rules.yaml`:
  - Job level matching (Entry-Level, Mid-Professional, etc.)
  - Duration penalties for screening stage (penalize tests >30 minutes)
  - Domain boosts (behavioral vs. technical skills)
- Output: `rule_score` and `rule_pass` boolean

### 3. Text Similarity
- **Algorithm:** TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Compute cosine similarity between job text and assessment `combined_text`
- Output: `similarity_score` (0-1, higher = better text match)

### 4. Feature Engineering
Generate 7 numerical features per job-assessment pair:
- Skill overlap count and ratio
- Similarity score
- Job level match (binary)
- Category match (binary)
- Duration penalty
- Rule score

### 5. ML Ranking
- **Model:** Gradient Boosted Trees (sklearn `GradientBoostingClassifier`)
- **Training:** Weak supervision using 7 synthetic job scenarios
  - Generate 168 examples (7 scenarios × 24 assessments)
  - Label top 20% as positive (1), rest as negative (0) based on rules + similarity
- **Output:** `ml_score` (0-1 relevance prediction)

### 6. Final Scoring
Combine scores with weighted formula:

final_score = 0.6 × ml_score + 0.4 × similarity_score

Rank assessments by `final_score` (descending) and return top N.

##  Evaluation

- **Metrics:** AUC (Area Under ROC Curve) on validation set
- **Results:** AUC = 1.0 (perfect separation on weak labels)
- **Scenario-based Testing:** 7 synthetic jobs (Account Manager, .NET Developer, Cashier, etc.)
- **Top Recommendations:** Evaluated for relevance (e.g., "Account Manager" → "Account Manager Solution")
- **Note:** Small dataset + weak labels may lead to overfitting; production use requires more data.




##  Usage

### Training
Train the ML model (run once before deploying):
```bash
python evaluation/evaluate.py

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training (weak supervision)
```
python evaluation/evaluate.py
```
This script:
1) Loads catalog and rules
2) Generates weak labels from synthetic scenarios
3) Trains GradientBoostingClassifier and saves models/ranking_model.pkl
4) Prints scenario-based top picks

## Run API
```
flask --app api/app.py run --host 0.0.0.0 --port 5000
```
or
```
python run_app_.py
```
## Run Streamlit UI
```
streamlit run ui/streamlit_app.py
```
or
```
python run_ui_.py
```

## Inference contract
Input example:
```
{
  "job_role": "Account Manager",
  "job_level": "Mid-Professional",
  "skills": ["client communication", "project coordination", "stakeholder management"],
  "hiring_stage": "Screening"
}
```
Output example (top-N list):
```
[
  {
    "assessment_name": "Account Manager Solution",
    "final_score": 0.92,
    "explanation": {
      "rule_pass": true,
      "rule_score": 0.9,
      "similarity_score": 0.88,
      "ml_score": 0.93,
      "job_level_score": 0.8,
      "duration_penalty": 0,
      "domain_boost": 0.2
    }
  }
]
```

## Notes
- Rules are externalized; adjust config/rules.yaml instead of changing code.
- Labels are heuristic; model is only as good as the scenarios and rules.
- Swap TF-IDF with Sentence Transformers in recommendation/similarity_engine.py if higher-quality embeddings are needed.
