import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # ==========================
    # File path dataset
    # ==========================
    X_train_path = sys.argv[1] if len(sys.argv) > 1 else "telco_churn_preprocessing/X_train.csv"
    X_test_path  = sys.argv[2] if len(sys.argv) > 2 else "telco_churn_preprocessing/X_test.csv"
    y_train_path = sys.argv[3] if len(sys.argv) > 3 else "telco_churn_preprocessing/y_train.csv"
    y_test_path  = sys.argv[4] if len(sys.argv) > 4 else "telco_churn_preprocessing/y_test.csv"

    # Load dataset
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Input example untuk MLflow
    input_example = X_train.head(5)

    # ==========================
    # Best params (fixed)
    # ==========================
    best_params = {
        "solver": "liblinear",
        "penalty": "l2",
        "max_iter": 2000,
        "C": 29.763514416313193
    }

    # MLflow run
    mlflow.set_experiment("Telco Churn - Logistic Regression Best Params")

    with mlflow.start_run():
        model = LogisticRegression(
            solver=best_params["solver"],
            penalty=best_params["penalty"],
            max_iter=best_params["max_iter"],
            C=best_params["C"],
            random_state=42
        )
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        # Metrics
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        # --- Confusion matrix ---
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r,c]), va='center', ha='center')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close(fig)

        # --- Prediksi CSV ---
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba})
        pred_path = "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        # --- Metrics JSON ---
        json_path = "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # Log model + input example
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)

        # Log metrics
        mlflow.log_metrics(metrics_dict)

        # Log artefak
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(pred_path)
        mlflow.log_artifact(json_path)

