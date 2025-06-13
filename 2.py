# USAGE
# python 2.py --selfies output/faces --classifier output/classifier --sample-size 200

# import the necessary packages
from __future__ import print_function
from image_research.face_recognition import FaceRecognizer
from imutils import encodings
import numpy as np
import argparse
import imutils
import random
import glob
import cv2
import os

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--selfies", required=True, help="path to the selfies directory")
ap.add_argument("-c", "--classifier", required=True, help="path to the output classifier directory")
ap.add_argument("-n", "--sample-size", type=int, default=200, help="maximum sample size for each face")
args = vars(ap.parse_args())

# Initialize LBPH Face Recognizer with optimized parameters
if imutils.is_cv2():
    fr = FaceRecognizer(cv2.createLBPHFaceRecognizer(radius=2, neighbors=5, grid_x=8, grid_y=8))
else:
    fr = FaceRecognizer(cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=5, grid_x=8, grid_y=8))

# Initialize the list of labels
labels = []

# Loop over the input faces for training
for (i, path) in enumerate(glob.glob(os.path.join(args["selfies"], "*.txt"))):
    # Extract the person's name from the file name
    name = os.path.basename(path).replace(".txt", "")
    print(f"[INFO] Training on '{name}'...")

    # Load the faces file, sample it, and initialize the list of faces
    sample = open(path).read().strip().split("\n")
    sample_size = min(len(sample), args["sample_size"])  # Adjust sample size dynamically
    sample = random.sample(sample, sample_size)
    faces = []

    # Loop over the faces in the sample
    for face in sample:
        # Decode the face image
        img = encodings.base64_decode_image(face)

        # Preprocessing steps
        img = cv2.equalizeHist(img)  # Histogram Equalization
        img = img.astype("float32") / 255.0  # Normalize pixel values

        # Append to training set
        faces.append(img)

    # Train the recognizer with improved dataset
    fr.train(faces, np.array([i] * len(faces)))
    labels.append(name)

# Save the trained model
fr.setLabels(labels)
fr.save(args["classifier"])
print("[INFO] Training complete. Model saved!")
