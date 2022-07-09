import numpy as np


from Script import data_processing as dp
from Script import models


def run_trainer():
    data = dp.scan()
    label = data[:,2]
    input = data[:,:2]
    X_train, X_val, y_train, y_val = dp.train_val_split(input, label)
    model, history = models.model(X_train, X_val, y_train, y_val, max_epoch=300)
    models.savemodel(model, history)
    print('----------------------training model is done----------------------')

    return model

