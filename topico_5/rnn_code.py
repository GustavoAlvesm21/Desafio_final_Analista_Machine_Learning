import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import mlflow
import mlflow.pytorch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
OUTPUT_DIR = ROOT.parent / "topico_5" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATH = DATA_ROOT / "Credit.csv"

LR = 0.001
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CreditRNN(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):

        output, hidden = self.rnn(x)

        out = self.fc(hidden[-1])

        return torch.sigmoid(out)


def load_data():

    data = pd.read_csv(PATH)

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("category").cat.codes

    return data


def prepare_data(data):

    X = data.drop("class", axis=1)
    y = data["class"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=24
    )

    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)

    y_train = torch.FloatTensor(y_train.values).view(-1, 1)
    y_test = torch.FloatTensor(y_test.values).view(-1, 1)

    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):

    X_train = X_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_train = y_train.to(DEVICE)
    y_test = y_test.to(DEVICE)

    model = CreditRNN(
        input_size=X_train.shape[2]
    ).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")
    best_path = OUTPUT_DIR / "best_model.pth"

    with mlflow.start_run(run_name="RNN"):

        for epoch in range(EPOCHS):
            
            model.train()

            pred = model(X_train)
            loss = criterion(pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred = (pred > 0.5).float()
            train_acc = (train_pred == y_train).float().mean()

            model.eval()
            with torch.no_grad():

                pred_val = model(X_test)
                val_loss = criterion(pred_val, y_test)

                val_pred = (pred_val > 0.5).float()
                val_acc = (val_pred == y_test).float().mean()

            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                torch.save(model.state_dict(), best_path)

            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            mlflow.log_metric("val_loss", val_loss.item(), step=epoch)
            mlflow.log_metric("train_accuracy", train_acc.item(), step=epoch)
            mlflow.log_metric("val_accuracy", val_acc.item(), step=epoch)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss.item():.4f} | "
                    f"Train Acc: {train_acc.item():.4f} | "
                    f"Val Acc: {val_acc.item():.4f}"
                )

        model.load_state_dict(torch.load(best_path))
        model.eval()

        with torch.no_grad():
            pred = model(X_test)
            pred = (pred > 0.5).float()
            acc = (pred == y_test).float().mean()

        print(f"Final Accuracy: {acc.item():.4f}")

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("hidden_size", 32)
        mlflow.log_param("num_layers", 2)

        mlflow.log_metric("best_val_loss", best_loss)
        mlflow.log_metric("final_accuracy", acc.item())

        mlflow.pytorch.log_model(
            model,
            "RNN_model",
            serialization_format="pickle"
        )

def main():

    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    train(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()