import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

with open('encoded_faces_with_labels.pickle', 'rb') as handle:
    data = pickle.load(handle)


def transformToXy(data):
    X, y = [], []
    for name in data['code']:
        X.extend([i for i in data[name]])
        y.extend([name] * len(data[name]))

    return X, y


def encodeXy(X, y):
    X = np.array(X)
    y = np.array(y)

    labelEncoder, oneHotEncoder = LabelEncoder(), OneHotEncoder()

    y = labelEncoder.fit_transform(y).reshape(-1, 1)

    y = oneHotEncoder.fit_transform(y).toarray().reshape(
        -1, 1, len(data['code']))

    return X, y


X, y = transformToXy(data)
X, y = encodeXy(X, y)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
print(X_train.shape)
print(y.shape)
model = Sequential([
    Dense(1024, input_shape=(1, 128), activation='relu'),
    Dense(len(data['code']), activation='softmax')
])

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)
print(model.predict(X_train[0].reshape(-1, 1, 128)), y_train[0])
print(history.history.keys())
acc = history.history['accuracy']
val = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epoch = range(len(acc))
plt.plot(epoch, acc, 'r', label='ACc')
plt.plot(epoch, val, 'b', label='Val_acc')
plt.legend()
plt.show()
plt.plot(epoch, loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.plot(epoch, val_loss)
plt.xlabel('epoch')
plt.ylabel('val_loss')
plt.show()
model.save('model.h5')
