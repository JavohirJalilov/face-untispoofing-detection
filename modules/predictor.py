import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np
import math

class FaceDetector:
    def __init__(self):
        path = "/Users/javohirjalilov/github/cvProject/ProjectsPR/Face/face-untispoofing-detection/models/mediapipe/face_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def bbox_face(self, landmarks, image):
        for face_landmark in landmarks:
            h, w, _ = image.shape
            
            x1 = 2000
            y1 = 2000
            x2 = y2 = 0
            for id, lm in enumerate(face_landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x1:
                    x1 = x
                if x > x2:
                    x2 = x

                if y < y1:
                    y1 = y
                if y > y2:
                    y2 = y
        
        w = x2 - x1
        h = y2 - y1

        if w > h:
            d = w - h
            y1 -= d//2
            y2 = y1 + w
        if h > w:
            d = h - w
            x1 -= d//2
            x2 = x1 + h

        return x1, y1, x2, y2
    
    def detecFace(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = self.detector.detect(image=mp_image)
        face_landmarks = results.face_landmarks
        x1, y1, x2, y2 = self.bbox_face(face_landmarks, image)
        return x1, y1, x2, y2


if __name__ == "__main__":
    face_detector = FaceDetector()
