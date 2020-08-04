import glob
import pickle
import numpy as np

for filepath in glob.glob('data/*.pickle'):
    name = filepath.split('\\')[1].split('_')[0]
    print('Processing', name)
    with open(filepath, 'rb') as handle:
        embedding = pickle.load(handle)
        data = []
        for i in range(len(embedding)):
            current = embedding[i]
            for j in range(len(embedding)):
                data.append(np.hstack((current, embedding[j])))
                current = np.hstack((embedding[j], current))
                print(current.shape)
                data.append(current)
        data.extend(embedding)
    with open(f'{name}_encoded_wav.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)