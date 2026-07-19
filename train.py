"""
Train and compare classification models on the Wisconsin Breast Cancer dataset.
Saves the best model + scaler + feature list to models/ for the API to serve.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

def load_data(path="data/breast-cancer.csv"):
    df = pd.read_csv(path)
    # Drop id and any stray unnamed columns
    df = df.drop(columns=[c for c in df.columns if c.lower() == "id" or "unnamed" in c.lower()], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y

def main():
    X, y = load_data()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print(f"{'Model':<22}{'CV-Acc':>10}{'Test-Acc':>10}{'Precision':>11}{'Recall':>10}{'F1':>8}{'ROC-AUC':>10}")
    print("-" * 85)

    best_name, best_model, best_f1 = None, None, -1

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = dict(cv_accuracy=cv_scores.mean(), test_accuracy=acc,
                              precision=prec, recall=rec, f1=f1, roc_auc=auc)

        print(f"{name:<22}{cv_scores.mean():>10.4f}{acc:>10.4f}{prec:>11.4f}{rec:>10.4f}{f1:>8.4f}{auc:>10.4f}")

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    print("\nBest model:", best_name)
    print("\nClassification report for best model:")
    y_pred_best = best_model.predict(X_test_s)
    print(classification_report(y_test, y_pred_best, target_names=["Benign", "Malignant"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_best))

    # Persist artifacts
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open("models/model_info.json", "w") as f:
        json.dump({"best_model_name": best_name, "results": results}, f, indent=2)

    print("\nSaved: models/best_model.joblib, models/scaler.joblib, models/feature_names.json")

if __name__ == "__main__":
    main()
