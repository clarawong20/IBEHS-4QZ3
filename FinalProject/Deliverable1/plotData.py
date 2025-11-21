import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # enables 3D plotting

# Load accelerometer data
data = pd.read_csv('demo_test.csv')
data = data[data['activity'].isin(['sit', 'walk'])]

# Extract relevant columns
Ax = data['Ax']
Ay = data['Ay']
Az = data['Az']
activity = data['activity']

# Assign colors for each activity type
colors = {
    'walk': 'green',
    'sit': 'blue',
}

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each activity with a different color
for act in data['activity'].unique():
    subset = data[data['activity'] == act]
    ax.scatter(
        subset['Ax'], subset['Ay'], subset['Az'],
        label=act,
        color=colors.get(act, 'gray'),
        s=40,
        alpha=0.7
    )

# Labels, legend, and title
ax.set_xlabel('Ax')
ax.set_ylabel('Ay')
ax.set_zlabel('Az')
ax.set_title('3D Accelerometer Data by Activity')
ax.legend(title='Activity')
plt.tight_layout()
plt.show()