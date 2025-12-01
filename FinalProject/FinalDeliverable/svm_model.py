import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import joblib

###### 1. Load data ######
data = pd.read_csv('training_data.csv')

# Filter for the activities
data = data[data['activity'].isin(['sit', 'walk', 'run', 'turn CW'])]

###### 2. Prepare features (windowing) ######
WINDOW_SIZE = 10
def compute_window_features(ax, ay, az):
    ax = np.array(ax)
    ay = np.array(ay)
    az = np.array(az)
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    return [np.mean(ax), np.mean(ay), np.mean(az), np.mean(a_mag), 
            np.std(ax), np.std(ay), np.std(az), np.std(a_mag), 
            np.var(a_mag)]

windows_features = []
windows_labels = []
windows_activities = []  # Track activities in each window

ax_win, ay_win, az_win = [], [], []
activities_win = []

for i, row in data.iterrows():
    ax_win.append(row['Ax'])
    ay_win.append(row['Ay'])
    az_win.append(row['Az'])
    activities_win.append(row['activity'])
    
    if len(ax_win) == WINDOW_SIZE:
        features = compute_window_features(ax_win, ay_win, az_win)
        windows_features.append(features)
        # Label window with the most common activity
        most_common_activity = max(set(activities_win), key=activities_win.count)
        windows_labels.append(most_common_activity)
        windows_activities.append(activities_win.copy())
        
        ax_win.pop(0)
        ay_win.pop(0)
        az_win.pop(0)
        activities_win.pop(0)

X = np.array(windows_features)
y = np.array(windows_labels)

###### 3. Split data ######
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

###### 4. Scale features ######
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

###### 5. Hyperparameter tuning with GridSearchCV ######
param_grid = {
    'estimator__C': [1, 10, 100],
    'estimator__kernel': ['linear', 'rbf'],
    'estimator__probability': [True] 
}

grid = GridSearchCV(
    OneVsOneClassifier(SVC(probability=True)),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train_scaled, y_train)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

###### 6. Train final model ######
model = OneVsOneClassifier(SVC(kernel=grid.best_params_['estimator__kernel'], C=grid.best_params_['estimator__C'], probability=True))
model.fit(X_train_scaled, y_train)

###### 7. SVM Model Validation ######
# Validate with 10-fold cross-validation
# use k = 10 since smaller dataset
scores = cross_val_score(
    OneVsOneClassifier(SVC(kernel=grid.best_params_['estimator__kernel'], C=grid.best_params_['estimator__C'], probability=True)),
    X_train_scaled,
    y_train,
    cv=10,
    scoring='accuracy'
)

print("Cross-validation accuracy scores:", scores)
print("Mean CV accuracy:", scores.mean())

# Plot cross-validation scores
plt.figure(figsize=(6, 5))
plt.boxplot(scores, vert=True)

# Also plot individual points
plt.scatter(
    [1]*len(scores), scores, 
    color='black', s=40, alpha=0.7
)

plt.title("10-fold Cross-Validation Accuracy")
plt.ylabel("Accuracy")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.xticks([1], ['Accuracy'])
plt.show()

###### 8. Evaluate on test set ######
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))

##### 9. Visualize Data ######
# a) confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualize Confusion Matrix
plt.figure(figsize=(7,6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

classes = ["sit", "walk", "run", "turn CW"]

# Axis labels
plt.xticks(np.arange(4), classes)
plt.yticks(np.arange(4), classes)
# label squares
for i in range(4):
    for j in range(4):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# b) PCA Visualization of Feature Space
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
# Use a fixed activity order and explicit color mapping so plots are consistent
activity_list = ['sit', 'walk', 'run', 'turn CW']
color_map = {'sit': 'blue', 'walk': 'green', 'run': 'red', 'turn CW': 'yellow'}

for act in activity_list:
    col = color_map.get(act, 'gray')
    idx = (y_train == act)
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=act,
        color=col,
        s=50,
        alpha=0.8
    )

plt.title("PCA Projection of Accelerometer Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

# feature distribution per activity using A_Mag only
plt.figure(figsize=(8, 6))

for act in activity_list:
    values = data[data['activity'] == act]['A_mag']
    plt.violinplot(values, positions=[activity_list.index(act)], showmeans=True)

plt.title("Distribution of A_mag for Each Activity")
plt.xlabel("Activity")
plt.ylabel("A_mag")
plt.xticks(range(len(activity_list)), activity_list)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()

# feature distribution using mean A_mag per window
plt.figure(figsize=(8, 6))
mean_a_mag_per_window = []

for features in windows_features:
    mean_a_mag_per_window.append(features[3])  # mean A_mag is the 4th feature
mean_a_mag_per_window = np.array(mean_a_mag_per_window)
for act in activity_list:
    values = mean_a_mag_per_window[np.array(windows_labels) == act]
    plt.violinplot(values, positions=[activity_list.index(act)], showmeans=True)

plt.title("Distribution of Mean A_mag per Window for Each Activity")
plt.xlabel("Activity")
plt.ylabel("Mean A_mag")
plt.xticks(range(len(activity_list)), activity_list)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()

# feature distribution using std of A_mag per window
plt.figure(figsize=(8, 6))
std_a_mag_per_window = []

for features in windows_features:
    std_a_mag_per_window.append(features[7])  # std A_mag is the 8th feature
std_a_mag_per_window = np.array(std_a_mag_per_window)
for act in activity_list:
    values = std_a_mag_per_window[np.array(windows_labels) == act]
    plt.violinplot(values, positions=[activity_list.index(act)], showmeans=True)

plt.title("Distribution of Std A_mag per Window for Each Activity")
plt.xlabel("Activity")
plt.ylabel("Std A_mag")
plt.xticks(range(len(activity_list)), activity_list)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()

# feature distribution using var A_mag per window
plt.figure(figsize=(8, 6))
var_a_mag_per_window = []

for features in windows_features:
    var_a_mag_per_window.append(features[8])  # var A_mag is the 9th feature
var_a_mag_per_window = np.array(var_a_mag_per_window)
for act in activity_list:
    values = var_a_mag_per_window[np.array(windows_labels) == act]
    plt.violinplot(values, positions=[activity_list.index(act)], showmeans=True)

plt.title("Distribution of Var A_mag per Window for Each Activity")
plt.xlabel("Activity")
plt.ylabel("Var A_mag")
plt.xticks(range(len(activity_list)), activity_list)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()

# Save model and scaler
joblib.dump({'model': model, 'scaler': scaler}, 'svm_model.pkl')
print("Model and scaler saved to 'svm_model.pkl'")
