import pickle
with open('total.pickle', 'rb') as handle:
    data = pickle.load(handle)

print(data)