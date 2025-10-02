import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        continue

features = pd.read_csv('diabetes.csv')

''' 

1. Data Exploration: Report and plot descriptive statistics of your available data. Descriptive statistics are the
first step in exploring a dataset and at a minimum should include mean, median, mode, standard deviation, and
range. Additional information should include skewness, kurtosis, and completeness

'''

cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Summary table 
summary = pd.DataFrame({
    "Mean": features[cols].mean(),
    "Median": features[cols].median(),
    "Mode": features[cols].mode().iloc[0],   # mode() can return multiple rows; take first
    "Standard Deviation": features[cols].std(),
    "Range": features[cols].max() - features[cols].min(),
    "Skewness": features[cols].skew(),
    "Kurtosis": features[cols].kurtosis(),
    "Completeness": (features[cols] != 0).mean()
})

print("Descriptive Statistics")
print(summary)

# grid of box plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(cols):
    axes[i].boxplot(features[col].dropna())
    axes[i].set_title(col)
    
    # Compute stats
    data = features[col].dropna()
    mean = data.mean()
    median = data.median()
    mode = data.mode().iloc[0]
    std = data.std()
    data_range = data.max() - data.min()
    skew = data.skew()
    kurt = data.kurtosis()
    completeness = (features[col] != 0).mean() * 100
    
    # Annotate on plot
    stats_text = (f"Mean: {mean:.2f}\nMedian: {median:.2f}\nMode: {mode}\n"
                  f"Std: {std:.2f}\nRange: {data_range:.2f}\n"
                  f"Skew: {skew:.2f}\nKurt: {kurt:.2f}\nCompl: {completeness:.1f}%")
    axes[i].text(1.05, 0.5, stats_text, transform=axes[i].transAxes, fontsize=9,
                 verticalalignment='center')

plt.tight_layout()
plt.show()

# grid of histograms
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(cols):
    data = features[col].dropna()
    axes[i].hist(data, bins=20, color='skyblue', edgecolor='black')
    
    # Stats
    mean = data.mean()
    median = data.median()
    mode = data.mode().iloc[0]
    std = data.std()
    data_range = data.max() - data.min()
    skew = data.skew()
    kurt = data.kurtosis()
    completeness = (features[col] != 0).mean() * 100
    
    # Overlay mean, median, mode lines
    axes[i].axvline(mean, color='red', linestyle='--', label='Mean')
    axes[i].axvline(median, color='green', linestyle='--', label='Median')
    axes[i].axvline(mode, color='orange', linestyle='--', label='Mode')
    
    # Annotate remaining stats
    stats_text = (f"Std: {std:.2f}\nRange: {data_range:.2f}\n"
                  f"Skew: {skew:.2f}\nKurt: {kurt:.2f}\nCompl: {completeness:.1f}%")
    axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                 verticalalignment='top', horizontalalignment='right', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    axes[i].set_title(col)
    axes[i].legend()

plt.tight_layout()
plt.show()


''' 

2. Feature Selection

'''
num_df = features.select_dtypes(include=[np.number])
corr = num_df.corr(numeric_only=True)
# Display correlations with Outcome
outcome_corr = corr["Outcome"].drop("Outcome").sort_values(ascending=False)
print('\n Correlation matrix\n', outcome_corr)
# Correlation heat map
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# rank the features by correlation with the outcome
# choose top feature and second feature with correlation < 0.7 to first
feat_rank = outcome_corr
feat1 = feat_rank.index[0]
feat2 = None
for cand in feat_rank.index[1:]:
    inter = abs(corr.loc[feat1, cand])
    if inter < 0.7:
        feat2 = cand
        break
    if feat2 is None:
        feat2 = feat_rank.index[1]
print("Chosen features:", feat1, "&", feat2)

# remove features with 0 values
cleaned_features = features[(features[feat1] != 0) & (features[feat2] != 0)]
# save cleaned data to txt file
cleaned_features.to_csv('cleaned_features.txt', sep='\t', index=False)

'''

3 & 4
Domain and Label Set

'''
X = cleaned_features[[feat1, feat2]].copy()
y = cleaned_features["Outcome"].copy()
print(X.head())
print(y.head())

''' 

5. Train/Test Data

'''

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)
plt.scatter(X_train[feat1], X_train[feat2], alpha=0.6, label="Train")
plt.scatter(X_test[feat1], X_test[feat2], marker="x", label="Test")
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.legend()
plt.title("Train vs Test Split")
plt.show()

'''

6. Initial Measure of Success (h₁)

'''
# Plot training data, colored by outcome
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='bwr', alpha=0.7)
plt.xlabel(X_train.columns[0])
plt.ylabel(X_train.columns[1])
plt.title("Training Data by Outcome")
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='No Diabetes', markerfacecolor='blue', markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Diabetes', markerfacecolor='red', markeredgecolor='k', markersize=8)
]
plt.legend(handles=legend_elements, title='Outcome')
plt.show()

# logistic regression model for h1
model = LogisticRegression()
model.fit(X_train, y_train)

# graph with decision boundary
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Visualize decision boundary on training data
x_min, x_max = X[feat1].min() - 1, X[feat1].max() + 1
y_min, y_max = X[feat2].min() - 1, X[feat2].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train[feat1], X_train[feat2], c=y_train, edgecolor='k', alpha=0.7)
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title("Linear Decision Boundary (Training)")
plt.tight_layout()
plt.show()

w1, w2 = clf.coef_[0]; b = clf.intercept_[0]
print(f"h1(x) = sign({w1:.3f}*{feat1} + {w2:.3f}*{feat2} + {b:.3f})")


# confusion matrix
y_pred_train = clf.predict(X_train)
cm_train = confusion_matrix(y_train, y_pred_train)
acc_train = accuracy_score(y_train, y_pred_train)
err_train = 1 - acc_train

print("Training accuracy:", round(acc_train,3), "| Empirical error:", round(err_train,3))
ConfusionMatrixDisplay(cm_train).plot(values_format='d')
plt.title("Training Confusion Matrix")
plt.tight_layout()
plt.show()