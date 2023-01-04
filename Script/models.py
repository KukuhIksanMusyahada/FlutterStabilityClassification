import os
import numpy as np
import datetime
import pickle

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

from Script import path_handler as ph


def model(X_train, X_val, y_train, y_val, max_epoch= 300):
    model = Sequential([
        Dense(300, activation='relu', input_dim = 2),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['Accuracy'])

    history = model.fit(X_train,y_train, epochs= max_epoch, 
                        validation_data=(X_val, y_val),verbose=0)

    return model, history

def savemodel(model, history, optional_path: str=None):
    """Save both model and history"""
    nomor_model = str(len(os.listdir(ph.get_models_data()))+1)
    folder_name = "ModelFlutterClassification" + nomor_model
    if optional_path != None:
        model_directory = os.path.join (optional_path, folder_name)
    else:
        model_directory = os.path.join (ph.get_models_data(), folder_name)
    os.makedirs(model_directory)
    history_file = os.path.join(model_directory, 'history.pkl')


    model.save(model_directory)
    print ("\nModel saved to {}".format(model_directory))

    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print ("Model history saved to {}".format(history_file))


def loadmodel(path_to_model):
    """Load Model and optionally it's history as well"""
    history_file = os.path.join(path_to_model, 'history.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("\nmodel loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")

    return model, history


def predict(model, mach= None, vf=None):
    input = [[mach, vf]]
    pred = model.predict(input)
    pred = np.round(pred[0][0])

    return pred