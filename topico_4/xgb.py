import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PATH = "/home/gu/Documentos/desafio_final_xpe/data/Credit.csv"

def load_data():
    
    data = pd.read_csv(PATH)
    
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category').cat.codes
    
    return data

def prepare_data(data):
    
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=24)

    return X_train, X_test, y_train, y_test
                
def train(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=f"XGBClassifier") as run:
        
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300]
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
        model = XGBClassifier(eval_metric="logloss")
    
        grid = GridSearchCV(
            model, 
            param_grid,
            cv=skf,
            scoring="f1"
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_score = grid.best_score_
        best_params = grid.best_params_

        y_pred = best_model.predict(X_test)

        mlflow.log_param("best_params", best_params)
        mlflow.log_metric("f1_score", best_score)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))

        mlflow.xgboost.log_model(best_model, "xgboost_model")

        print(f"Best Parameters: {best_params}")
        print(f"Best Score: {best_score}")

def main():
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    train(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()