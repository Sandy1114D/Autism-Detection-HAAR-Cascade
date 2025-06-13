# USAGE
# python Register.py --face-cascade cascades/haarcascade_frontalface_default.xml --output output/faces/autism.txt

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
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")
ap.add_argument("-p", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# initialize the face detector and output file
fd = FaceDetector(args["face_cascade"])
f = open(args["output"], args["write_mode"])
total = 0
captureMode = False
color = (0, 255, 0)

# loop over the frames of the video
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, scaleFactor=1.2, minNeighbors=8, minSize=(100, 100))

    # process detected faces
    for (x, y, w, h) in faceRects:
        if captureMode:
            face = gray[y:y + h, x:x + w].copy(order="C")
            f.write("{}\n".format(encodings.base64_encode_image(face)))
            total += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # toggle capture mode
    if key == ord("c"):
        captureMode = not captureMode
        color = (0, 0, 255) if captureMode else (0, 255, 0)
    elif key == ord("q"):
        break

print("[INFO] wrote {} frames to file".format(total))
f.close()
cv2.destroyAllWindows()
vs.stop()