import cv2
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from modules import FaceDetector
from modules.utils import CropImage
from time import time

model = tf.keras.models.load_model("models/untispoofing")
face_detector = FaceDetector()
crop_image = CropImage()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    start = time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        x1, y1, x2, y2 = face_detector.detecFace(img)

        crop = crop_image.crop(img, (x1, y1, x2 - x1, y2 - y1))
        
        batch_img = np.expand_dims(crop, axis=0)
        pred = model(batch_img)
        softmax_data = softmax(pred.numpy())
        idx = softmax_data.argmax()

        label = "Real" if idx == 1 else "Fake"
        # Display the resulting frame
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        img = cv2.putText(img, label, (x1, y1 -10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 246, 0), 2)

    except:
        pass

    end = time()
    FPS = round(1/(end - start), 1)

    img = cv2.putText(img, f"FPS: {FPS}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()