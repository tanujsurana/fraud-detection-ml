import argparse
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download the dataset and place it as data/creditcard.csv\n"
            "Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud"
        )
    return pd.read_csv(path)


def train_and_evaluate(df: pd.DataFrame, threshold: float):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print(f"ROC AUC: {roc_auc_score(y_test, y_probs):.4f}")
    print(f"Threshold: {threshold}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training Script")
    parser.add_argument("--data", type=str, default="data/creditcard.csv", help="Path to dataset CSV")
    parser.add_argument("--threshold", type=float, default=0.2, help="Decision threshold for fraud class")
    parser.add_argument("--out", type=str, default="models/fraud_model.joblib", help="Output path to save model")
    args = parser.parse_args()

    df = load_data(args.data)
    model, scaler = train_and_evaluate(df, threshold=args.threshold)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, args.out)
    print(f"Saved model to: {args.out}")


if __name__ == "__main__":
    main()
EOF

