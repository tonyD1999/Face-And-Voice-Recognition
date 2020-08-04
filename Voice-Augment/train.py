import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, LSTM, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

with open('total.pickle', 'rb') as handle:
    data = pickle.load(handle)


def transformToXy(data):
    X, y = [], []
    for name in data['code']:
        X.extend([i for i in data[name]])
        y.extend([data['label'][name]] * len(data[name]))

    return X, y

def encodeXy(X, y):
    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes=len(data['code']))
    return X, y


X, y = transformToXy(data)
X, y = encodeXy(X, y)
print(X.shape)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y.reshape(-1, 1, len(data['code'])),
                                                    test_size=0.2,
                                                    random_state=42
                                                    )
print(X_train.shape)
print(y_train.shape)
print(len(data['code']))
model = Sequential([
    Dense(2048, input_shape=(1, 256), activation='relu'),
    Dropout(0.5),
    Dense(len(data['code']), activation='softmax')
])
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=35, validation_split=0.2)
print(model.predict(X[0].reshape(-1, 1, 256)), y[0])
print(model.evaluate(X_test, y_test))
model.save('model.h5')