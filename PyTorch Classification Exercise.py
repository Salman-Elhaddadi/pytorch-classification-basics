import torch
from torch import nn
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# ---------- 1. DATA ----------
n_samples = 1000

x_1, y = make_moons(n_samples, noise=0.03, random_state=42)


moons = pd.DataFrame({"X1" : x_1[:, 0],
                      "X2" : x_1[:, 1],
                      "lable" : y})
#print(moons.head(10))

x2 = torch.from_numpy(x_1).type(torch.float)
y2 = torch.from_numpy(y).type(torch.float)

x_train, x_test, y_train, y_test = train_test_split(x2,
                                                    y2,
                                                    test_size=0.2,
                                                    random_state=42)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- 2. MODEL ----------
class MoonModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


torch.manual_seed(42)
model_3 = MoonModelV0().to(device)
#print(model_3)

# ---------- 3. TOOLS ----------
def accuracy_fn(y_true, y_pred):

    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_3.parameters(), lr=0.01)


# ---------- 4. TRAINING ----------
epochs = 500

x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

for epoch in range(epochs):

    model_3.train()
    y_logits = model_3(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_3.eval()
    with torch.inference_mode():

        test_logits = model_3(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")


x_min, x_max = x_1[:, 0].min() -0.1, x_1[:, 0].max() + 0.1
y_min, y_max = x_1[:, 0].min() -0.1, x_1[:, 0].max() + 0.1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.float).to(device)


model_3.eval()
with torch.inference_mode():
    grid_logits = model_3(grid).squeeze()

grid_pred = torch.round(torch.sigmoid(grid_logits))
zz = grid_pred.cpu().numpy().reshape(xx.shape)

plt.contourf(xx, yy, zz, cmap=plt.cm.RdYlBu, alpha=0.7)
plt.scatter(x_1[:, 0], x_1[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.title("Decision boundary")
plt.show()
