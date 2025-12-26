import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    X_train_path = sys.argv[1]
    X_test_path  = sys.argv[2]
    y_train_path = sys.argv[3]
    y_test_path  = sys.argv[4]

    max_iter = int(sys.argv[5])
    solver   = sys.argv[6]
    penalty  = sys.argv[7]
    C        = float(sys.argv[8])

    X_train = pd.read_csv(X_train_path)
    X_test  = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test  = pd.read_csv(y_test_path).values.ravel()

    input_example = X_train.head(5)

    mlflow.log_params({
        "solver": solver,
        "penalty": penalty,
        "max_iter": max_iter,
        "C": C
    })

    model = LogisticRegression(
        solver=solver,
        penalty=penalty,
        max_iter=max_iter,
        C=C,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    mlflow.log_metrics(metrics)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba
    }).to_csv("predictions.csv", index=False)

    mlflow.log_artifact("predictions.csv")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    mlflow.log_artifact("metrics.json")

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=input_example
    )

