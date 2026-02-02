# Credit Card Fraud Detection using Machine Learning

This project focuses on building a machine learning system to detect fraudulent credit card transactions using an imbalanced dataset. The goal is to identify fraudulent activity accurately while minimizing false negatives, which is critical in real world financial systems.

---

## Project Overview

Credit card fraud detection is a classic imbalanced classification problem where fraudulent transactions represent a very small fraction of total transactions. In this project, I built an end to end machine learning pipeline that includes data preprocessing, model training, evaluation, and threshold tuning to improve fraud detection performance.

Key highlights:
- Handles highly imbalanced data
- Uses class weighting and threshold tuning
- Evaluates performance using ROC AUC, precision, recall, and confusion matrix
- Designed to be reproducible and production friendly

---

## Dataset

- Source: Kaggle Credit Card Fraud Dataset
- Dataset contains anonymized transaction features
- Target column: `Class`
  - 0 → Legitimate transaction
  - 1 → Fraudulent transaction

⚠️ The dataset is **not committed** to this repository due to size and best practices.  
Please download it separately and place it inside the `data/` folder as:


---

## Project Structure

```
fraud-detection-ml/
│
├── app.py # Training and evaluation script
├── README.md # Project documentation
├── requirements.txt # Minimal project dependencies
│
├── data/ # Dataset (not committed)
├── notebooks/ # EDA and experimentation notebooks
├── models/ # Saved trained models (ignored by git)
├── screenshots/ # Visual results and plots
└── src/ # Supporting scripts and utilities
```

---

## Machine Learning Approach

- Algorithm: Logistic Regression
- Handling imbalance:
  - Class weighting (`class_weight="balanced"`)
  - Custom probability threshold tuning
- Feature scaling using StandardScaler
- Stratified train test split

---

## Evaluation Metrics

- ROC AUC score
- Precision and recall for fraud class
- Confusion matrix
- Threshold based prediction instead of default 0.5

This approach prioritizes **catching fraudulent transactions** while balancing false positives.

---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
