# RASPBERRY PI SCRIPT FOR DATA COLLECTION
from sense_hat import SenseHat
from datetime import datetime
import joblib
import numpy as np 
import csv, time, math

sense = SenseHat()
sense.clear()
#filename = "deliverable1_log_" + datetime.now().strftime("%H%M%S") + ".csv"
filename = "final_training_data.csv"
button_labels = {"up": "walk", "down": "sit", "left": "run", "right": "turn CW"}
activity = None
active = False
msg = ""
color_map = { "sit": [0, 0, 255],
			  "walk": [0, 255, 0],
			  "run": [255, 0, 0],
			  "turn CW": [255, 255, 0]
			  }
color = [255, 255, 255]


with open(filename, 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(["time", "Ax", "Ay", "Az", "A_mag", "active", "activity"])
	sense.show_message("START", text_colour=[255, 255, 255], scroll_speed=0.1)
	
	while True:
		for e in sense.stick.get_events():
			if e.direction == "up" and e.action == "pressed" or e.direction == "down" and e.action == "pressed" or e.direction == "left" and e.action == "pressed" or e.direction == "right" and e.action == "pressed":
				active = True
				activity = button_labels [ e.direction ]
				#msg = "sit" if activity == "sit" else "walk"
				msg = activity
				color = color_map.get(activity, [255, 255, 255])
				print("activity:", activity)
				sense.show_message ( msg , text_colour = color , scroll_speed =0.05)
				
			
			elif e.direction == "middle" and e.action == "pressed":
				sense.show_message("END", text_colour=[255, 255, 255], scroll_speed=0.05)
				active = False
				print("recording stopped")
				sense.clear()
				f.close()
				exit()
		
		a = sense.get_accelerometer_raw ()
		Ax , Ay , Az = a ['x'] , a ['y'] , a ['z']
		A_mag = math . sqrt ( Ax **2 + Ay **2 + Az **2)
		writer.writerow ([ datetime.now().isoformat() , Ax , Ay , Az , A_mag, active, activity ])
		f.flush()
		# collect sample at 50Hz
		time.sleep (0.02)
