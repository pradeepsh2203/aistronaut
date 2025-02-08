from deepface import DeepFace
import cv2
import json
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

img_path = "img/img_10.png"
img = cv2.imread(img_path)

objs = DeepFace.analyze(img)
print(objs)