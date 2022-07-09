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

def SaveModel(model, history, optional_path: str=None):
    """Save both model and history"""
    now = datetime.datetime.now()
    time_now = now.strftime('%Y%m%d%H%M%S')
    folder_name = "ANN " + time_now
    if optional_path != None:
        model_directory = os.path.join (optional_path, folder_name)
    else:
        model_directory = os.path.join (ph.get_models_data(), folder_name)
    os.makedirs(model_directory)
    history_file = os.path.join(model_directory, 'history.pkl')


    model.save(model_directory)
    print ("\nModel saved to {}".format(model_directory))

    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
    print ("Model history saved to {}".format(history_file))


def LoadModel(path_to_model):
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