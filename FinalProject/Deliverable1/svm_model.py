import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load accelerometer data
data = pd.read_csv('test2.csv')

# Filter only sitting vs walking
data = data[data['activity'].isin(['sit', 'walk'])]

# Features and labels
X = data[['Ax', 'Ay', 'Az', 'A_mag']]
y = data['activity']

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump({'model': model, 'scaler': scaler}, 'deliverable1_svm_model.pkl')
print("Model and scaler saved to 'deliverable1_svm_model.pkl'")
