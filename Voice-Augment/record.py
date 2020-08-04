import pyaudio
import wave
import argparse
import os
from io import BytesIO

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name', required=True, help='Input your name')
ap.add_argument('-c', '--count', required=True, help='Input times')
ap.add_argument('-m', '--mic', required=False, help='Input mic id', default=None)
args = vars(ap.parse_args())

if args['mic']:
    MIC_ID = int(args['mic'])
else:
    MIC_ID = None

if not os.path.exists(f'data/{args["name"]}'):
    os.mkdir(f'data/{args["name"]}')

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3
count = 0
data_encode = []
while count < int(args['count']):

    wave_output_filename = f"data/{args['name']}/{count}.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=MIC_ID
                    )

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
    wf = wave.open(wave_output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.writeframes(b''.join(frames))
    wf.close()
    # encoded_data = resemblyzer.preprocess_wav(wave_output_filename)
    # data_encode.append(encoded_data)
    count += 1


# with open(f'data/{args["name"]}_encoded_wav.pickle', 'wb') as handle:
#     pickle.dump(data_encode, handle, protocol=pickle.HIGHEST_PROTOCOL)