import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("livenes_models/model")
classes = ['fake', 'real']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    in_img = cv2.resize(img, (32, 32))
    in_img = in_img.astype('float') / 255.0
    in_img = tf.keras.preprocessing.image.img_to_array(in_img)
    
    in_img = np.expand_dims(in_img, axis=0)
    pred = model(in_img)
    idx = np.argmax(pred.numpy())

    print(pred)
    # Display the resulting frame
    cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()