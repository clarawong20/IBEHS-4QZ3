import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# PARAMETERS
# ------------------------------
learning_rate = 1
num_epochs = 300
AX = (0, 1.5, -0.5, 0.5)  # plot window (xmin, xmax, ymin, ymax)

# ------------------------------
# LOAD DATA
# ------------------------------
data = pd.read_csv("deliverable1_activities.csv")

# Filter to only sitting and walking
data = data[data["activity"].isin(["sitting", "walking"])]

# Feature selection (you can adjust which accelerometer axes to use)
X = data[["Ax", "Ay"]].values
y = np.where(data["activity"] == "walking", 1, -1)  # walking = +1, sitting = -1

# Combine into single list of tuples
dataset = list(zip(X, y))

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def predict(x1, x2, b, w1, w2):
    s = b + w1 * x1 + w2 * x2
    return (1 if s > 0 else -1), s

def plot_points(ax):
    x_n, y_n, x_p, y_p = [], [], [], []
    for (x, label) in dataset:
        if label == 1:
            x_p.append(x[0]); y_p.append(x[1])
        else:
            x_n.append(x[0]); y_n.append(x[1])
    ax.scatter(x_n, y_n, c='r', marker='x', label='Sitting (-1)')
    ax.scatter(x_p, y_p, c='g', marker='o', label='Walking (+1)')

def plot_boundary(ax, b, w1, w2):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    eps = 1e-12
    # If w2 is not (near) zero, plot as y = -(b + w1*x)/w2
    if abs(w2) > eps:
        x_vals = np.linspace(x_min, x_max, 2)
        y_vals = -(b + w1 * x_vals) / w2
        ax.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    # If w2 is (near) zero but w1 isn't, the boundary is vertical: x = -b/w1
    elif abs(w1) > eps:
        x0 = -b / w1
        ax.vlines(x0, y_min, y_max, linestyles='dashed', colors='k', label='Decision Boundary')
    else:
        # both weights are (near) zero — cannot draw a meaningful boundary
        ax.text(0.02, 0.98, 'No decision boundary (w≈0)', transform=ax.transAxes,
                verticalalignment='top', fontsize=8, color='k')

def accuracy(data, b, w1, w2):
    correct = sum(1 for (x, y) in data if predict(x[0], x[1], b, w1, w2)[0] == y)
    return correct / len(data)

# ------------------------------
# POCKET PERCEPTRON ALGORITHM
# ------------------------------
def pocket_perceptron(data, learning_rate, num_epochs):
    b, w1, w2 = 0.0, 0.0, 0.0
    best_b, best_w1, best_w2 = b, w1, w2
    best_acc = accuracy(data, b, w1, w2)
    mistakes_per_epoch = []

    for epoch in range(num_epochs):
        mistakes = 0
        for (x, y) in data:
            yhat, _ = predict(x[0], x[1], b, w1, w2)
            if yhat != y:
                # Update weights
                b += learning_rate * y
                w1 += learning_rate * y * x[0]
                w2 += learning_rate * y * x[1]
                mistakes += 1

                # Pocket check
                current_acc = accuracy(data, b, w1, w2)
                if current_acc > best_acc:
                    best_acc = current_acc
                    best_b, best_w1, best_w2 = b, w1, w2
        
        mistakes_per_epoch.append(mistakes)
        print(f"Epoch {epoch+1}: {mistakes} misclassified")
        if mistakes == 0:
            break

    print(f"\nFinal Pocket accuracy: {best_acc*100:.2f}%")
    print(f"Best params: b={best_b:.3f}, w1={best_w1:.3f}, w2={best_w2:.3f}")
    return best_b, best_w1, best_w2, mistakes_per_epoch

# ------------------------------
# TRAIN MODEL
# ------------------------------
b, w1, w2, mistakes = pocket_perceptron(dataset, learning_rate, num_epochs)

# ------------------------------
# PLOT RESULTS
# ------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(AX[0], AX[1]); ax.set_ylim(AX[2], AX[3])
plot_points(ax)
plot_boundary(ax, b, w1, w2)
ax.set_xlabel("Ax")
ax.set_ylabel("Ay")
ax.legend()
ax.set_title("Pocket Perceptron: Sitting vs Walking")
plt.show()

# ------------------------------
# PLOT MISTAKES PER EPOCH
# ------------------------------
plt.figure(figsize=(6,4))
plt.plot(range(1, len(mistakes)+1), mistakes, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassified Samples')
plt.title('Training Progress')
plt.show()
