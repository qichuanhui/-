import cv2
import numpy as np
import tensorflow as tf

size = 4
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'without_mask', 1: 'with_mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
webcam = cv2.VideoCapture()  # Use camera 0
webcam.open(0,cv2.CAP_DSHOW)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=tf.keras.models.load_model("mask.h5")
while True:
    (rval, im) = webcam.read()
    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = im[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (128, 128))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 128, 128, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        # print(result)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the image
    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()