import snowboydecoder
import sys
import wave
import pyaudio
import cv2
import requests
import json
import pickle
from queue import Queue
import numpy as np
server_url = 'http://172.20.10.3:5001/'
model_file = './computer.umdl'
queue = Queue()
p = pyaudio.PyAudio()
sensitivity = 0.7
lst = ['1752015', '1752259', '1752041', 'NOT INDENTIFIED']
detection = snowboydecoder.HotwordDetector(model_file, sensitivity=sensitivity)


def callback(in_data, frame_count, time_info, status):
    queue.put(in_data)
    return (in_data, pyaudio.paContinue)


stream = p.open(
                rate=16000,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=512,
                input_device_index=11,
                stream_callback=callback
)
stream.start_stream()


def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
while True:
    _, frame = cam.read()
    cv2.imshow('frame', frame)
    # encapsulate_face = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
    # face_response = request.post(server_url+'face', data=encapsulate_face).json()['data']
    if queue.qsize() == 0:
        continue
    while queue.qsize() > 32:
        queue.get()
    buff = []
    if queue.qsize() >= 32:
        while queue.qsize() > 0:
            buff.append(queue.get())
    ans = detection.detector.RunDetection(b''.join(buff))
    # print(ans)
    if ans == 1:
        print("yes")
        # _, frame = cam.read()
        # cv2.imshow('frame', frame)
        encapsulate_face = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
        face_response = requests.post(server_url+'face', data=encapsulate_face).json()
        if face_response['data']:
            face_result = np.array(face_response['data'])
            left, right, top, bottom = tuple(face_response['box'])
            # stream.close()
            p.terminate()
            p1 = pyaudio.PyAudio()
            stream1 = p1.open(
                    rate=16000,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=512,
                    input_device_index=11,
            )
            frames = []
            for i in range(0, int(16000/512*3)):
                # _, frame = cam.read()
                # cv2.imshow('frame', frame)
                frames.append(stream1.read(512))
            stream1.close()
            p1.terminate()
            encapsulate_voice = pickle.dumps(frames, protocol=pickle.HIGHEST_PROTOCOL)
            voice_response = requests.post(server_url+'voice', data=encapsulate_voice).json()['data']
            voice_response = np.array(voice_response)
            result = np.add(face_result*0.5, voice_response*0.5)
            if np.any(result > 0.8):
                name = f'{lst[result.argmax()]}:{max(result)}'
            else:
                name = lst[-1]
            p = pyaudio.PyAudio()                                                                                                                                                                                                                                                                                                                                                                                                                
            stream = p.open(
                    rate=16000,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=512,
                    input_device_index=11,
                    stream_callback=callback
            )
            cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.imshow('result', frame)
            stream.start_stream()
    if cv2.waitKey(1) == 27:
        break
