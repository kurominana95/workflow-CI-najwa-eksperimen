import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ==========================
# MLflow setup
# ==========================
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Telco Churn - Logistic Regression")
mlflow.sklearn.autolog()

# ==========================
# Load preprocessed data
# ==========================
X_train = pd.read_csv("preprocessing/telco_churn_preprocessing/X_train.csv")
X_test = pd.read_csv("preprocessing/telco_churn_preprocessing/X_test.csv")
y_train = pd.read_csv("preprocessing/telco_churn_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("preprocessing/telco_churn_preprocessing/y_test.csv").values.ravel()

# ==========================
# Train model
# ==========================
with mlflow.start_run():

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")

