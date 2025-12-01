from sense_hat import SenseHat
import joblib
import numpy as np
import time
import math
from collections import deque

# Load trained SVM and scaler
bundle = joblib.load('svm_model.pkl')
model = bundle['model']
scaler = bundle['scaler']

sense = SenseHat()
sense.clear()

# LED colors for each activity
colors = {
    "sit": [0, 0, 255],
    "walk": [0, 255, 0],
    "run": [255, 0, 0],
    "turn CW": [255, 255, 0]
}

WINDOW_SIZE = 10
window_ax = deque(maxlen=WINDOW_SIZE)
window_ay = deque(maxlen=WINDOW_SIZE)
window_az = deque(maxlen=WINDOW_SIZE)
window_mag = deque(maxlen=WINDOW_SIZE)

# Feature extraction
def compute_features():
    """Compute statistics from the rolling window."""
    ax = np.array(window_ax)
    ay = np.array(window_ay)
    az = np.array(window_az)
    mag = np.array(window_mag)

    features = [
        np.mean(ax), np.mean(ay), np.mean(az), np.mean(mag),
        np.std(ax), np.std(ay), np.std(az), np.std(mag),
        np.var(mag)
    ]

    return np.array(features).reshape(1, -1)

# get accelerometer data
def get_features():
    a = sense.get_accelerometer_raw()
    Ax, Ay, Az = a['x'], a['y'], a['z']
    A_mag = math.sqrt(Ax**2 + Ay**2 + Az**2)
    return Ax, Ay, Az, A_mag

# Startup message
sense.show_message("READY", text_colour=[255, 255, 255])
time.sleep(1)
sense.clear()

print("Starting real-time HAR...")

last_prediction = None  # Track last activity

while True:
    Ax, Ay, Az, A_mag = get_features()

    # Add to rolling windows
    window_ax.append(Ax)
    window_ay.append(Ay)
    window_az.append(Az)
    window_mag.append(A_mag)

    # Prediction only after window is full
    if len(window_ax) == WINDOW_SIZE:
        X = compute_features()
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        # Print to terminal
        print("Activity:", prediction)

        # change LED color only if prediction changed
        if prediction != last_prediction:
            color = colors.get(prediction, [255, 255, 255])
            sense.clear(color)
            last_prediction = prediction

    # Joystick stop
    for e in sense.stick.get_events():
        if e.direction == "middle" and e.action == "pressed":
            sense.show_message("END", text_colour=[255, 255, 255])
            print("Measurement stopped")
            sense.clear()
            exit()

    time.sleep(0.02)
