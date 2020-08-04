import pickle
import glob
import numpy as np
from resemblyzer import VoiceEncoder
import os

encoder = VoiceEncoder('cpu')
data = {}
classes = {}
i = 0
path = 'data'
for name in os.listdir(path):
    if os.path.isdir(os.path.join(path, name)):
        classes[name] = i
        i += 1


def preprocess(name, embedding):
    global data
    embedded = encoder.embed_utterance(embedding)
    embedded = np.array(embedded).reshape(1, 256)
    if name not in data:
        data[name] = [embedded]
    else:
        data[name].append(embedded)


for filepath in glob.glob('*.pickle'):
    name = filepath.split('_')[0]
    with open(filepath, 'rb') as handle:
        embedding = pickle.load(handle)
        for em in embedding:
            preprocess(name, em)
data['code'] = list(data.keys())
data['label'] = classes
with open('total.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)