import dlib
from imutils import paths
import numpy as np
import os
import pickle


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

dataset = {}
for name in os.listdir('data'):
    dataset[name] = []

imagePaths = list(paths.list_images('data'))
num_of_images = len(imagePaths)
for(i, data) in enumerate(imagePaths):
    print(data)
    print(f'Processing image {i}/{num_of_images}')
    label = data.split('\\')[-2]
    img = dlib.load_rgb_image(data)
    dets = detector(img)
    for d in dets:
        shape = predictor(img, d)
        face = facerec.compute_face_descriptor(img, shape)
        face = np.array(face).reshape(1, 128)
        dataset[label].append(face)

dataset['code'] = list(dataset.keys())
with open('encoded_faces_with_labels.pickle', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)