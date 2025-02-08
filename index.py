from deepface import DeepFace
import cv2
import json
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
def emotion_check(img_array):
    # img_path = img_path
    # img = cv2.imread(img_path)
    objs = DeepFace.analyze(img_array)
    return objs
   