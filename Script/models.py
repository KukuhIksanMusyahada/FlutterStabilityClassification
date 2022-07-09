import numpy as np

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense


def model(X, y, max_epoch):
    model = Sequential([
        Dense(300, activation='relu', input_dim = 2),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(X,y, epochs= max_epoch, verbose=0)

    return model