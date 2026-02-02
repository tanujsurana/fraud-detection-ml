# Credit Card Fraud Detection using Machine Learning

## Overview
This project implements a machine learning based fraud detection system using a highly imbalanced credit card transaction dataset. The goal is to accurately detect fraudulent transactions while minimizing false positives.

## Problem
Fraud cases represent a very small fraction of total transactions, making accuracy an unreliable metric. This project focuses on recall, precision, F1 score, ROC AUC, and confusion matrix analysis.

## Approach
- Data preprocessing and feature scaling
- Handling class imbalance using class weighting
- Training multiple models including Logistic Regression and Random Forest
- Threshold tuning to balance fraud recall and false positives
- Evaluation using ROC AUC, precision, recall, and confusion matrix

## Results
At the selected threshold, the model:
- Correctly detected 83 out of 98 fraud cases
- Generated only 13 false positives
- Achieved ROC AUC of approximately 0.95

## Key Learnings
- Accuracy is misleading for imbalanced datasets
- Threshold tuning significantly improves fraud recall
- Confusion matrix analysis is critical for business driven ML decisions

## Technologies
Python, Scikit learn, Pandas, NumPy, Matplotlib, Seaborn
