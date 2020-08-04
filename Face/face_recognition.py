import dlib
import cv2
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('model.h5')
lst = []
cam = cv2.VideoCapture(0)
cam.set(3, 400)
cam.set(4, 400)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
SCALE = 1
i = 0
print(dlib.cuda.get_device())
for name in os.listdir('data'):
    lst.append(name)
lst.append('Guest')
while True:
    ret, frame = cam.read()
    temp = cv2.resize(frame, (0, 0), fx=1/SCALE, fy=1/SCALE)
    dets = detector(temp)
    if dets:
        max_a = (dets[0].bottom() - dets[0].top()) * (dets[0].right() - dets[0].left())
        d = dets[0]
        if len(dets) > 1:
            for dz in dets[1:]:
                h = (dz.bottom() - dz.top()) * (dz.right() - dz.left())
                if h > max_a:
                    d = dz
        left = d.left()*SCALE
        right = d.right()*SCALE
        top = d.top()*SCALE
        bottom = d.bottom()*SCALE
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0))
        shape = predictor(temp, d)
        face = facerec.compute_face_descriptor(temp, shape)
        res = model.predict(np.asarray(face).reshape(-1, 1, 128))
        res = res.flatten()
        if np.any(res > 0.9):
            name = lst[res.argmax()] + f': {round(res[res.argmax()] * 100 , 3)}'
        else:
            name = lst[-1]
        cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()