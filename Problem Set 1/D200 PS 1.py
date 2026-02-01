import torch

##################################################
#################### Problem 1 ###################
##################################################

## """ 1a """
print("*** Answers for Problem 1a ***")

# 1. Create a tensor a containing the values [1.0, 2.0, 3.0, 4.0, 5.0]
a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print("a =", a)

# 2. Create a 3x3 tensor B filled with ones
B = torch.ones(3, 3)
print(f"\n")
print("B =\n", B)

# 3. Reshape a to a 5x1 column vector
a_col = a.view(5, 1)
print(f"\n")
print("a reshaped (5x1) =\n", a_col)

# 4. Compute the element-wise square of a
a_square = a ** 2
print(f"\n")
print("element-wise square of a =", a_square)

# 5. Compute the matrix product of B with itself
B_product = torch.matmul(B, B)
print(f"\n")
print("B @ B =\n", B_product)


## """ 1b """
print(f"\n")
print("*** Answers for Problem 1b ***")

# 1.
# f(x) = x^2 + 3x + 1
# df/dx = 2x + 3
# x = 2 -> df/dx = 7
derivative = 2*2 + 3

# 2.
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1

# 3.
y.backward()
print(x.grad)

# 4.
print("Analytical gradient:", derivative)
print("Pytorch autograd gradient:", x.grad.item())


## """ 1c """
print(f"\n")
print("*** Answers for Problem 1c ***")

# 1.
# g(x, y) = x^2 y + y^3
# ∂g/∂x = 2xy
# ∂g/∂y = x^2 + 3y^2
x_m, y_m = 1, 2
derivative_x = 2 * x_m * y_m
derivative_y = x_m**2 + 3 * y_m**2

# 2.
x_a = torch.tensor(1.0, requires_grad=True)
y_a = torch.tensor(2.0, requires_grad=True)
g = x_a**2 * y_a + y_a**3
g.backward()

# 3.
print("Analytical ∂g/∂x:", derivative_x)
print("Pytorch ∂g/∂x:", x_a.grad.item())
print("Analytical ∂g/∂y:", derivative_y)
print("Pytorch ∂g/∂y:", y_a.grad.item())


# %%
##################################################
#################### Problem 2 ###################
##################################################

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# Generate synthetic data for linear regression
torch.manual_seed(0) # control randomness

n_samples = 100
true_weight = 3.5
true_bias = 1.2
X = torch.randn(n_samples, 1)
noise = 0.3 * torch.randn(n_samples, 1)
y = true_weight * X + true_bias + noise


## """ 2a """

# 1.
plt.figure()
plt.scatter(X.numpy(), y.numpy(), s=15)
plt.title("Synthetic data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

model = nn.Linear(in_features=1, out_features=1) # y_hat = Wx + b

# 2.
print("Initial weight:", model.weight.item())
print("Initial bias:", model.bias.item())

# 3.
with torch.no_grad():
    y_pred_init = model(X)

plt.figure()
plt.scatter(X.numpy(), y.numpy(), s=15, label='data')
plt.scatter(X.numpy(), y_pred_init.numpy(), s=15, label='initial predication')
plt.title("Initial predictions vs True data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


## """ 2b """
model_sgd = nn.Linear(in_features=1, out_features=1)
criterion = nn.MSELoss()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.1)

epochs = 100
loss_history_sgd = []

for epoch in range(epochs):
    y_pred_sgd = model_sgd(X)               # 1. Forward pass
    loss_sgd = criterion(y_pred_sgd, y) # 2. Compute loss
    optimizer_sgd.zero_grad()           # 3. Zero gradients
    loss_sgd.backward()                 # 4. Backward pass
    optimizer_sgd.step()                # 5. Update parameters

    loss_history_sgd.append(loss_sgd.item()) # Store loss

plt.figure()
plt.plot(loss_history_sgd)
plt.title("Loss curve (SGD)")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.show()


## """ 2c """

# 1.
learned_weight = model_sgd.weight.item()
learned_bias = model_sgd.bias.item()
print(f"Learned weight (SGD): {learned_weight:.6f}")
print(f"Learned bias (SGD): {learned_bias:.6f}")

# 2.
X_design = torch.cat([X, torch.ones(n_samples, 1)], dim=1)
beta_hat = torch.linalg.solve(X_design.T @ X_design, X_design.T @ y)

ols_weight = beta_hat[0].item()
old_bias = beta_hat[1].item()
print(f"\n")
print(f"OLS weight: {ols_weight:.6f}")
print(f"OLS bias: {old_bias:.6f}")

# 3.
table = pd.DataFrame({
    "Weight": [learned_weight, ols_weight, true_weight],
    "Bias":   [learned_bias, old_bias, true_bias]
}, index=["SGD", "OLS", "True"])

print(f"\n")
print(table)

# 4.
with torch.no_grad():
    y_pred_trained = model_sgd(X)

X_sorted = torch.argsort(X[:, 0])

plt.figure()
plt.scatter(X.numpy(), y.numpy(), s=15, label="data")
plt.plot(X[X_sorted].numpy(), y_pred_trained[X_sorted].numpy(), label="SGD fitted line")
plt.title("Learned regression line (SGD)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


## """ 2d """
model_adam = nn.Linear(in_features=1, out_features=1)
criterion = nn.MSELoss()
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.1)

epochs = 100
loss_history_adam = []

for epoch in range(epochs):
    y_pred_adam = model_adam(X)           # 1. Forward pass
    loss_adam = criterion(y_pred_adam, y) # 2. Compute loss
    optimizer_adam.zero_grad()            # 3. Zero gradients
    loss_adam.backward()                  # 4. Backward pass
    optimizer_adam.step()                 # 5. Update parameters

    loss_history_adam.append(loss_adam.item()) # Store loss

# Figurization
plt.figure()
plt.plot(loss_history_sgd, label="SGD")
plt.plot(loss_history_adam, label="Adam")
plt.title("Loss curves: SGD vs Adam")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.show()

# Analyze
print("Loss at epoch 10:")
print(f"SGD: {loss_history_sgd[9]:.6f}")
print(f"Adam: {loss_history_adam[9]:.6f}")
"""
From the loss curves, SGD converges faster than Adam in 
this experiment, as SGD reaches a much lower MSE within 
the first 10 epochs, while Adam decreases more slowly.
"""


# %%
##################################################
#################### Problem 3 ###################
##################################################

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Basic setup
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Visualize some examples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")


## """ 3a """
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10) 
).to(device)

print("\nModel architecture:\n", model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params: {total_params}")
print(f"Trainable params: {trainable_params}")


## """ 3b """
def compute_accuracy(model, data_loader, device):
    model.eval() # Sets evaluation mode (disables dropout/batchnorm training behavior)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

criterion = nn.CrossEntropyLoss() # Include LogSoftmax and Negative Log Likelihood
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 5
train_loss_history = []
train_acc_history = []
test_acc_history = []

for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)           # 1. Forward pass
        loss = criterion(outputs, labels) # 2. Compute loss
        optimizer.zero_grad()             # 3. Zero gradients
        loss.backward()                   # 4. Backward pass
        optimizer.step()                  # 5. Update parameters

        running_loss += loss.item()  # record current loss in this batch

    avg_train_loss = running_loss / len(train_loader)

    train_acc = compute_accuracy(model, train_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)

    train_loss_history.append(avg_train_loss)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f"Epoch {epoch:>2}/{n_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train Accuracy: {train_acc:.2f}% | "
          f"Test Accuracy: {test_acc:.2f}%")


## """ 3c """

# 1. Plot the training loss and accuracies over epochs
plt.figure()
plt.plot(train_loss_history)
plt.title("Training Loss over Epochs")
plt.xlabel("epoch")
plt.ylabel("CrossEntropy loss")
plt.show()

plt.figure()
plt.plot(train_acc_history, label="Train Acc")
plt.plot(test_acc_history, label="Test Acc")
plt.title("Accuracy over Epochs")
plt.xlabel("epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# 2.
model.eval()
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)

# Display 10 test images with their predicted labels
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = images[i].cpu().squeeze()
    true_label = labels[i].cpu().item()
    pred_label = preds[i].cpu().item()

    ax.imshow(img, cmap="gray")
    
    # Mark incorrect predictions in red
    color = "red" if pred_label != true_label else "black"
    ax.set_title(f"Pred: {pred_label} / True: {true_label}", color=color)
    ax.axis("off")

plt.tight_layout()
plt.show()

# 3.
final_test_acc = test_acc_history[-1]
print(f"\nFinal test accuracy: {final_test_acc:.2f}% (random guess ~10%)")
