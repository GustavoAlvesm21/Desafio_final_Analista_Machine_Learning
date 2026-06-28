import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PATH = "/home/gu/Documentos/desafio_final_xpe/data/Credit.csv"

def load_data():
    
    data = pd.read_csv(PATH)

    return data

def prepare_data(data):
    
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category').cat.codes

    X = data.drop('class', axis=1)
    y = data['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    return X_train, X_test, y_train, y_test

def train(X_train, X_test, y_train, y_test):    
    with mlflow.start_run(run_name="SVM"):

        model = SVC(
            kernel="rbf",
            C=1.0
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("C", 1.0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(model, "model_SVM")

    print("Treinamento concluído")

def main():
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    train(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()