import torch
from torch import nn
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

torch.manual_seed(42)

num_classes = 4
x_1, y = make_blobs(n_samples=1000, centers=num_classes, n_features=2, random_state=42,
                    cluster_std=0.6)

x_1 = torch.from_numpy(x_1).type(torch.float)
y = torch.from_numpy(y).type(torch.long)

x_train, x_test, y_train , y_test = train_test_split(x_1,
                                                     y,
                                                     test_size=0.2,
                                                     random_state=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

class BlobModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=num_classes)
        self.act = nn.Tanh()

    def forward(self, x):

        return self.layer_3(self.act(self.layer_2(self.act(self.layer_1(x)))))


model_4 = BlobModel().to(device)

def accuracy_fn(y_true, y_pred):

    return torch.eq(y_true, y_pred).sum().item() / len(y_true) * 100

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_4.parameters(), lr=0.001)

for epoch in range(200):
    
    model_4.train()
    logits = model_4(x_train)
    y_pred = logits.argmax(dim=1)
    loss = loss_fn(logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(x_test)
        test_pred = test_logits.argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")


@torch.inference_mode()
def predict(model, data):
    model.eval()
    return model(data.to(device)).argmax(dim=1).cpu().numpy()


X_np, y_np = x_1.numpy(), y.numpy()
x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]   # 4 clearly different colors
cmap = ListedColormap(colors)

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),    # finer grid
                     np.linspace(y_min, y_max, 400))
grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.float).to(device)

zz = predict(model_4, grid).reshape(xx.shape)

Xtr, ytr = x_train.cpu().numpy(), y_train.cpu().numpy()
Xte, yte = x_test.cpu().numpy(), y_test.cpu().numpy()
wrong_train = predict(model_4, x_train) != ytr          # misclassified train points
wrong_test = predict(model_4, x_test) != yte            # misclassified test points

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, zz, levels=np.arange(-0.5, num_classes), colors=colors, alpha=0.25)
plt.contour(xx, yy, zz, levels=np.arange(0.5, num_classes - 1), colors="k", linewidths=0.8)
plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cmap, s=15,
            edgecolors="white", linewidths=0.3)                          # train = dots
plt.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cmap, s=30, marker="^",
            edgecolors="white", linewidths=0.3)                          # test = triangles
plt.scatter(Xtr[wrong_train, 0], Xtr[wrong_train, 1], s=90, facecolors="none",
            edgecolors="k", linewidths=1.5, label=f"misclassified train ({wrong_train.sum()})")
plt.scatter(Xte[wrong_test, 0], Xte[wrong_test, 1], s=110, facecolors="none",
            edgecolors="red", linewidths=1.5, label=f"misclassified test ({wrong_test.sum()})")
plt.legend()
plt.title("Decision boundary (4 classes) - dots: train, triangles: test")
plt.show()