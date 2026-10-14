import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


data = load_breast_cancer()


x_train, x_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=data.target)


scaler = StandardScaler().fit(x_train)
x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)


class BreastCancerMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=30, out_features=16)
        self.layer_2 = nn.Linear(in_features=16, out_features=8)
        self.layer_3 = nn.Linear(in_features=8, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model = BreastCancerMLP().to(device)

def accuracy_fn(y_true, y_pred):

    return torch.eq(y_true, y_pred).sum().item() / len(y_true) * 100

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

history = {"train_loss": [], "test_loss": []}

for epoch in range(200):

    model.train()
    logits = model(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(logits))

    loss = loss_fn(logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    history["train_loss"].append(loss.item())
    history["test_loss"].append(test_loss.item())

    if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

print(f"\nFinal | Train acc: {acc:.2f}% | Test acc: {test_acc:.2f}%\n")

y_true = y_test.cpu().numpy().astype(int)
final_pred = test_pred.cpu().numpy().astype(int)
print(confusion_matrix(y_true, final_pred))
print(classification_report(y_true, final_pred, target_names=data.target_names))

baseline = LogisticRegression(max_iter=1000).fit(x_train, y_train)
print(f"Baseline LogisticRegression test acc: {baseline.score(x_test, y_test) * 100:.2f}%")

Path("images").mkdir(exist_ok=True)
plt.figure(figsize=(7, 4))
plt.plot(history["train_loss"], label="train")
plt.plot(history["test_loss"], label="test")
plt.xlabel("Epoch")
plt.ylabel("BCE loss")
plt.title("Breast cancer - loss curves")
plt.legend()
plt.tight_layout()
plt.savefig("images/04_breast_cancer_loss.png", dpi=150)
plt.show()