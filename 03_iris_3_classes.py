import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path                   


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=data.target)

x_train = torch.from_numpy(x_train).type(torch.float)
x_test = torch.from_numpy(x_test).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.long)
y_test = torch.from_numpy(y_test).type(torch.long)


x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

# Training the module
class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=4, out_features=16)
        self.layer_2 = nn.Linear(in_features=16, out_features=16)
        self.layer_3 = nn.Linear(in_features=16, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model = IrisModel().to(device)

def accuracy_fn(y_true, y_pred):

    return torch.eq(y_true, y_pred).sum().item() / len(y_true) * 100

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

history = {"train_loss": [], "test_loss": []}

for epoch in range(200):

    model.train()
    logits = model(x_train)
    y_pred = logits.argmax(dim=1)
    loss = loss_fn(logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test)
        test_pred = test_logits.argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    history["train_loss"].append(loss.item())
    history["test_loss"].append(test_loss.item())

    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")


print(f"\nFinal | Train acc: {acc:.2f}% | Test acc: {test_acc:.2f}%")

Path("images").mkdir(exist_ok=True)
plt.figure(figsize=(7, 4))
plt.plot(history["train_loss"], label="train")
plt.plot(history["test_loss"], label="test")
plt.xlabel("Epoch")
plt.ylabel("CrossEntropyLoss loss")
plt.title("Iris - loss curves")
plt.legend()
plt.tight_layout()
plt.savefig("images/03_iris_loss.png", dpi=150)
plt.show()

