import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / 'data'
OUTPUT_DIR = ROOT.parent / 'topico_3' / 'deep_ml' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PATH = DATA_ROOT / 'Credit.csv'

LR = 0.001
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepNN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def load_data():

    data = pd.read_csv(PATH)

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category').cat.codes

    return data


def prepare_data(data):

    X = data.drop("class", axis=1)
    y = data["class"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=24
    )

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):

    model = DeepNN(X_train.shape[1]).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    best_loss = float("inf")
    best_path = OUTPUT_DIR / "best_model.pth"

    for epoch in range(EPOCHS):

        model.train()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    print(f"Final Loss: {loss.item():.4f}")

    model.load_state_dict(torch.load(best_path))
    model.eval()

    with torch.no_grad():

        prob = model(X_test)

        pred = (prob > 0.5).float()

        y_true = y_test.cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = prob.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    with mlflow.start_run(run_name="Deep_NN_Run"):

        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("activation", "ReLU")
        mlflow.log_param("dropout", 0.3)
        mlflow.log_param("hidden_layers", "128-64-32-16")

        mlflow.log_metric("final_loss", loss.item())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.pytorch.log_model(
            model,
            "Deep_NN_Model",
            serialization_format="pickle"
        )


def main():

    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    train(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()