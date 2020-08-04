import pyaudio
import wave
import argparse
import os
from io import BytesIO
import numpy as np
import pickle
import librosa
import resemblyzer

voice = resemblyzer.VoiceEncoder('cpu')


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 2
count = 0
data_encode = []
while count < 1:

    wave_output_filename = f"check1.wav"
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    buff = BytesIO()
    wf = wave.open(buff, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    wf = wave.open(wave_output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    encoded_data = np.frombuffer(buff.getvalue(), dtype=np.int16)
    audio = encoded_data.astype(np.float32, order='C') / 32768.0
    print(audio)
    w = librosa.resample(audio, 44100*2, 44100)
    print('Wav')
    print(w)
    print('Lib')
    y, _ = librosa.load('check1.wav', 44100)
    print(y)
    librosa.output.write_wav('check2.wav', w, 44100)
    print('Lib2')
    y, _ = librosa.load('check2.wav', 44100)
    print(y)
    wav = resemblyzer.preprocess_wav(audio)
    embed = voice.embed_utterance(wav)
    print(embed)
    count += 1


