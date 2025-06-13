# Import necessary packages
from __future__ import print_function
from image_research.face_recognition import FaceDetector, FaceRecognizer
from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2
import csv
import os
#import serial  # Import serial for Arduino communication

# Initialize Serial Communication
#SerialObj = serial.Serial('COM3')  # Change COM port if needed
#SerialObj.baudrate = 9600  # Set Baud rate to 9600
#SerialObj.bytesize = 8     # Number of data bits = 8
#SerialObj.parity = 'N'     # No parity
#SerialObj.stopbits = 1     # Number of Stop bits = 1

# Hardcoded paths
FACE_CASCADE_PATH = "D:/Autism/cascades/haarcascade_frontalface_default.xml"
CLASSIFIER_PATH = "D:/Autism/output/classifier"
CONFIDENCE_THRESHOLD = 50.0  # Adjusted threshold
USE_PICAMERA = False  # Set to True if using Raspberry Pi camera
CSV_FILENAME = "Security_Record.csv"
ENTRY_COOLDOWN = 50  # Time before logging again

# Initialize the video stream
print("[INFO] Warming up camera...")
vs = VideoStream(usePiCamera=USE_PICAMERA).start()
time.sleep(2.0)  # Give the camera some time to warm up

# Initialize the face detector and recognizer
fd = FaceDetector(FACE_CASCADE_PATH)
fr = FaceRecognizer.load(CLASSIFIER_PATH)
fr.setConfidenceThreshold(CONFIDENCE_THRESHOLD)

# Ensure the CSV file has headers
if not os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Entry", "Name"])

# Dictionary to track last logged entries
last_entries = {}
face_detected_once = False  # Ensures '1' is sent only once

while True:
    frame = vs.read()
    intruder = False

    # Resize frame for faster processing
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Get current timestamp
    timestamp = datetime.datetime.now()
    date = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")

    for (x, y, w, h) in faceRects:
        # Extract face and predict identity
        face = gray[y:y + h, x:x + w]
        prediction, confidence = fr.predict(face)
        match_score = 100 - confidence  # Convert confidence for better readability

        # Determine if it's an intruder
        if confidence > CONFIDENCE_THRESHOLD:
            color = (0, 0, 255)  # Red for intruder
            intruder = True
            prediction = "Normal"
        else:
            color = (0, 255, 0)  # Green for recognized face

        # Display recognition details
        text = "{}: {:.2f}%".format(prediction.replace("faces\\", ""), match_score)
        cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Log only if the same person hasn’t been logged recently
        if prediction not in last_entries or (timestamp - last_entries[prediction]).seconds >= ENTRY_COOLDOWN:
            with open(CSV_FILENAME, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([date, time_str, "OUT", prediction])
            print(f"[LOG] {prediction} recorded at {time_str}")
            last_entries[prediction] = timestamp  # Update last logged time

        # Send '1' to Arduino only once after first recognition
        if not face_detected_once:
#            SerialObj.write(b'1\n')  # Send '1' to Arduino
          #  print("[INFO] '1' sent to Arduino")
            face_detected_once = True  # Ensure it's sent only once

    # Show the output frame
    cv2.imshow("Atutism Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit when 'q' is pressed
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
SerialObj.close()  # Close the serial connection
