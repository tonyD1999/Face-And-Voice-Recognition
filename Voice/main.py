import pyaudio
import numpy as np
import tensorflow
import wave
from resemblyzer import VoiceEncoder, preprocess_wav, normalize_volume, trim_long_silences, hparams
from io import BytesIO
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--mic', required=False, help='Input mic id', default=None)
args = vars(ap.parse_args())

if args['mic']:
    MIC_ID = int(args['mic'])
else:
    MIC_ID = None

CHANNELS = 1
RATE = 16000
tf_model = tensorflow.keras.models.load_model('model.h5')
encoder = VoiceEncoder('cpu')

p = pyaudio.PyAudio()
with open('total.pickle', 'rb') as handle:
    data = pickle.load(handle)
label = data['code']


while True:

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=512,
                    input_device_index=MIC_ID
                    )

    print("* recording")

    frames = []

    for i in range(0, int(RATE / 512 * 3)):
        data = stream.read(512)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    print(len(frames))
    buff = BytesIO()
    wf = wave.open('check.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.writeframes(b''.join(frames))
    wf.close()


    wav = preprocess_wav('check.wav')
    embed = encoder.embed_utterance(wav)
    embed = np.array(embed).reshape(-1, 1, 256)
    res = tf_model.predict(embed)
    res = res.flatten()
    if res.max() > 0.8:
        result = label[res.argmax()]
    else:
        result = 'Noise'
    print(f'{result} - {res}')
