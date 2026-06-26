# Credit Card Fraud Detection

End-to-end machine learning project detecting fraudulent credit card transactions using the ULB Credit Card Fraud dataset. Built as a data science portfolio project covering the full pipeline from raw data to deployed REST API.

**Best result:** Random Forest — 93.8% precision / 77.6% recall at optimized threshold, with only 13 false alarms across 56,962 test transactions.

---

## Business Problem

Credit card fraud costs the global financial industry billions annually. The core challenge is not just accuracy — it is catching real fraud (recall) without overwhelming operations teams with false alarms (precision), on a dataset where fraud represents only 0.173% of all transactions.

A naive model that labels everything as legitimate would be 99.83% accurate and completely useless. Every modeling decision in this project — SMOTE, stratified splits, threshold tuning — is a direct response to this imbalance.

---

## Dataset

**Source:** [ULB Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over 48 hours
- 492 fraud cases (0.173%)
- 30 features: V1–V28 (PCA-transformed, anonymized), Amount, Time
- No missing values

The dataset is downloaded automatically via `kagglehub` on first run and cached locally. It is excluded from version control.

---

## Project Structure

```
credit-card-fraud-detection/
├── notebook/
│   └── Credit_Card_Fraud_Detection.ipynb   ← full analysis
├── app/
│   ├── app.py                              ← Flask REST API
│   ├── model.pkl                           ← trained Random Forest
│   └── scaler.pkl                          ← fitted StandardScaler
├── images/                                 ← saved chart outputs
├── requirements.txt
└── README.md
```

---

## Key EDA Findings

**1 — Card Testing Behavior**
Median fraud amount is €9.25 vs €22.00 for legitimate transactions. A quarter of all fraud is €1 or less — consistent with fraudsters verifying stolen cards with micro-transactions before attempting larger ones.

**2 — Zero-Amount Transactions Are High Risk**
Authorization checks (€0 transactions) have a 1.48% fraud rate — 9x the dataset average. These should be flagged as elevated risk events in real monitoring systems.

**3 — V14 Is the Strongest Signal**
V14 shows the clearest separation between fraud and legitimate classes. An IQR outlier analysis found that extreme V14 values carry a 3.04% fraud rate vs 0.17% overall. These outliers are fraud signal, not noise — they were kept in the modeling data intentionally.

**4 — Both Models Independently Confirm V14**
Random Forest and XGBoost were trained separately and both ranked V14 as the #1 most important feature. XGBoost concentrates importance sharply on V14 (~0.58 of total), while RF distributes it more evenly (~0.23) — suggesting a strong non-linear fraud signal that boosting exploits more aggressively.

**5 — Overnight Fraud Spike**
Hourly trend analysis shows fraud rate increases during low-volume overnight hours, while transaction volume drops. Fraudsters appear to exploit periods of reduced monitoring.

**6 — Threshold Matters More Than Model Choice**
At the default 0.5 threshold, Logistic Regression generates 1,285 false alarms while missing only 9 frauds. Random Forest produces only 13 false alarms but misses 20. After threshold optimization, RF and XGBoost reach comparable F1 scores — model selection depends on the business cost of each error type.

---

## Modeling Approach

### Class Imbalance Handling
SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic fraud examples by interpolating between existing ones, balancing the training set to 50/50. Applied to training data only — never to the test set.

### Models Trained
| Model | Role |
|---|---|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Ensemble baseline — handles non-linear relationships |
| XGBoost | Sequential boosting — focuses on hard-to-classify examples |

### Evaluation Strategy
- Stratified 80/20 train/test split
- Stratified 5-fold cross-validation (SMOTE inside each fold via `imblearn.Pipeline`)
- Threshold optimization: F1 scanned across 0.1–0.9 range per model

---

## Results

### Single Split Performance (default threshold = 0.5)

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Fraud Precision | 0.065 | 0.857 | 0.748 |
| Fraud Recall | 0.908 | 0.796 | 0.847 |
| Fraud F1 | 0.121 | 0.825 | 0.794 |
| ROC-AUC | 0.970 | 0.963 | **0.977** |
| False Alarms (FP) | 1,285 | **13** | 28 |
| Missed Frauds (FN) | **9** | 20 | 15 |

### Cross-Validation (5-Fold, SMOTE inside each fold)

| Model | CV ROC-AUC | Std Dev |
|---|---|---|
| Logistic Regression | 0.9764 | ±0.0092 |
| **Random Forest** | **0.9781** | **±0.0063** |
| XGBoost | 0.9765 | ±0.0065 |

### After Threshold Optimization

| Model | Optimal Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.90 | 0.242 | 0.888 | 0.381 |
| **Random Forest** | **0.75** | **0.938** | **0.776** | **0.849** |
| XGBoost | 0.90 | 0.868 | 0.806 | 0.836 |

**Deployed model: Random Forest at threshold 0.75**
Best CV stability, highest optimized precision (93.8%), and fewest false alarms. XGBoost leads on single-split AUC (0.977) but requires a very high confidence threshold (0.90) before flagging fraud — a more hesitant operating point for a production system.

---

## REST API

The trained model is served as a Flask REST API.

### Run locally

```bash
cd app
python app.py
```

Server starts at `http://127.0.0.1:5000`

### Endpoints

**GET /health**
```bash
curl http://127.0.0.1:5000/health
```
```json
{"model": "RandomForest", "status": "ok"}
```

**POST /predict**

Accepts a JSON object with all 31 model features and returns a fraud probability and flag.

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.27, "V2": 2.46, ..., "Hour": 15.0, "Amount_scaled": -0.35, "Time_scaled": -0.79}'
```
```json
{
  "fraud_probability": 1.0,
  "is_fraud": true,
  "model": "RandomForest",
  "threshold_used": 0.75
}
```

**Input features:** V1–V28 (PCA-transformed), Hour (0–23), Amount_scaled, Time_scaled

**Validation:** returns HTTP 400 with a list of missing fields if any feature is absent, or if any value is non-numeric.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML | scikit-learn, xgboost, imbalanced-learn |
| Deployment | Flask, joblib |
| Dataset access | kagglehub |
| Environment | WSL Ubuntu, VS Code, virtualenv |

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Saurav-Subedi/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows WSL: same command

# Install dependencies
pip install -r requirements.txt

# Open notebook (dataset downloads automatically on first cell)
cd notebook
jupyter notebook Credit_Card_Fraud_Detection.ipynb

# Run the API (in a separate terminal)
cd app
python app.py
```

**Note:** `model.pkl` and `scaler.pkl` are included in the repository for convenience. In a production system these would be stored in cloud object storage or a model registry rather than committed to version control.

---

## Commit History

Each phase was committed separately to show progression:

- `Initial EDA + preprocessing + baseline models (LR, RF)`
- `Add correlation heatmap, hourly trends, V14 outlier analysis`
- `Add XGBoost model + feature importance comparison`
- `Add KFold cross-validation + threshold optimization + final comparison`
- `Add Flask API for fraud prediction deployment`
- `Clean notebook with section documentation`

---

## Author

**Saurav Subedi**  
IT Department Jyoti Bikash Bank ,  Data Scientist.  
[GitHub](https://github.com/Saurav-Subedi) 
