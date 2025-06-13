# python 3.py --face-cascade cascades/haarcascade_frontalface_default.xml --classifier output/classifier  --conf conf/alerts.json
# import the necessary packages
from __future__ import print_function
from image_research.face_recognition import FaceDetector
from image_research.face_recognition import FaceRecognizer
from image_research.utils import Conf
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2


# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-m", "--classifier", required=True, help="path to the classifier")
ap.add_argument("-t", "--confidence", type=float, default=50.0,  # Adjusted threshold
	help="minimum confidence threshold for positive face identification")
ap.add_argument("-n", "--consec-frames", type=int, default=90,
	help="# of consecutive frames containing an unknown face before sending alert")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream
print("[INFO] Warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()

# initialize the face detector and recognizer
fd = FaceDetector(args["face_cascade"])
fr = FaceRecognizer.load(args["classifier"])

# Set confidence threshold (lower confidence means better match)
fr.setConfidenceThreshold(args["confidence"])

# Tracking consecutive frames
consec = None
color = (0, 255, 0)
lastSent = None

while True:
	# grab the next frame from the stream
	frame = vs.read()
	intruder = False

	# resize the frame, convert to grayscale, and detect faces
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

	# draw timestamp on the frame
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# loop over detected faces
	for (i, (x, y, w, h)) in enumerate(faceRects):
		# grab the face region and predict
		face = gray[y:y + h, x:x + w]
		(prediction, confidence) = fr.predict(face)

		# Debugging output
		#print(f"Raw Prediction: {prediction}, Confidence: {confidence}")

		# Convert confidence score (100 - confidence for better readability)
		match_score = 100 - confidence

		# Handle "Unknown" case
		if prediction.lower() == "unknown" or confidence == 0:
			prediction = "Normal"  # Change "Unknown" to "Normal"
			confidence = 100  # Set confidence to 100% for unknown faces

		# if first frame or different person detected, reset consecutive frames count
		if consec is None or prediction != consec[0]:
			consec = [prediction, 1]
			color = (0, 255, 0)  # Green for recognized faces

		elif prediction == consec[0]:
			consec[1] += 1

		# Intruder detection logic (higher confidence means an unknown face)
		if confidence > args["confidence"]:
			color = (0, 0, 255)  # Red for intruder
			intruder = True

		# Display text with flipped confidence
		text = "{}: {:.2f}%".format(prediction.replace("faces\\", ""), match_score)
		cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

		# Send an alert if an intruder is detected
		if intruder and (lastSent is None or (timestamp - lastSent).seconds >= 10):
			print("[ALERT] Intruder detected at {}".format(timestamp))
			lastSent = timestamp  # Update last sent time

	# Show the frame
	cv2.imshow("Face Recognition System", frame)
	key = cv2.waitKey(1) & 0xFF

	# Quit when 'q' is pressed
	if key == ord("q"):
		break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
