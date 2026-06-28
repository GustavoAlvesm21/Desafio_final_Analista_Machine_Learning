import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PATH = "/home/gu/Documentos/desafio_final_xpe/data/Credit.csv"

def load_data():

    data = pd.read_csv(PATH)

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("category").cat.codes

    return data

def prepare_data(data):

    X = data.drop("class", axis=1)
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    param_grid = {

        "C": [0.01, 0.1, 1, 10, 100],

        "max_iter": [100, 300, 500, 1000],

        "solver": [
            "liblinear",
            "lbfgs"
        ],

        "penalty": [
            "l2"
        ]
    }

    model = LogisticRegression()

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    with mlflow.start_run(run_name="LogisticRegression_GridSearch"):

        mlflow.log_params(grid.best_params_)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        mlflow.log_metric("cv_best_score", grid.best_score_)

        mlflow.sklearn.log_model(best_model, "best_model")

    print("Melhores parâmetros:")
    print(grid.best_params_)

    print()

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

def main():

    data = load_data()

    prepare_data(data)

if __name__ == "__main__":
    main()