# USAGE
# python gather_selfies.py --face-cascade cascades/haarcascade_frontalface_default.xml \
# --output output/faces/ajay.txt

# import the necessary packages
import cv2
from image_research.face_recognition import FaceDetector
from imutils.video import VideoStream
from imutils import encodings
import argparse
import imutils
import time

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
# ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")
ap.add_argument("-p", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# initialize the face detector
fd = FaceDetector(r"D:\Autism\cascades\haarcascade_frontalface_default.xml")
captureMode = False
color = (0, 255, 0)

# open the output file for writing
f = open(r"output/faces/Normal.txt", "w+")
total = 0

# loop over the frames from the video stream
while True:
    # grab the next frame from the stream
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=9, minSize=(100, 100))

    # if at least one face was detected
    if len(faceRects) > 0:
        # select the largest face (based on area)
        (x, y, w, h) = max(faceRects, key=lambda b: (b[2] * b[3]))

        # if in capture mode, save the face region
        if captureMode:
            face = gray[y:y + h, x:x + w].copy(order="C")
            f.write("{}\n".format(encodings.base64_encode_image(face)))
            total += 1

        # draw a rectangle around the face and instructions
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame = cv2.putText(frame, 'Press C to capture', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, 'Press Q to Exit', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)

    # show the frame
    cv2.imshow("Frame", frame)

    # read key press
    key = cv2.waitKey(1) & 0xFF

    # toggle capture mode if 'c' is pressed
    if key == ord("c"):
        if not captureMode:
            captureMode = True
            color = (0, 0, 255)
        else:
            captureMode = False
            color = (0, 255, 0)

    # exit if 'q' is pressed
    elif key == ord("q"):
        break

# clean up
print(f"[INFO] wrote {total} frames to file")
f.close()
cv2.destroyAllWindows()
vs.stop()
