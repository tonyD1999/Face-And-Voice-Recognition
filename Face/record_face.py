import dlib
import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-n',
                '--name',
                required=True,
                help='name of person to train on'
                )
ap.add_argument('-c',
                '--count',
                required=True,
                help='number of images'
                )
args = vars(ap.parse_args())


cam = cv2.VideoCapture(0)
cam.set(3, 400)
cam.set(4, 400)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
SCALE = 1
i = 0
flag = True
if not os.path.exists(f'data/{args["name"]}'):
    os.makedirs(f'data/{args["name"]}')
while flag:
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
        face = temp[d.top():d.bottom(), d.left():d.right()]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0))
        if i == int(args['count']):
            flag = False
        cv2.imwrite(f'data/{args["name"]}/test{i}.jpg', face)
        i = i + 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()
