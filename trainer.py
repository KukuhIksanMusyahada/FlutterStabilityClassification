import numpy as np
import datetime

from Script import data_processing as dp
from Script import models


def run_trainer():
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING CLASSIFICATION FLUTTER START AT {time_now}')
    data = dp.scan()
    label = data[:,2]
    input = data[:,:2]
    X_train, X_val, y_train, y_val = dp.train_val_split(input, label)
    model, history = models.model(X_train, X_val, y_train, y_val, max_epoch=300)
    models.savemodel(model, history)
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING CLASSIFICATION DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO TRAIN CLASSIFICATION MODEL IS {delta_time}')
    print('----------------------training model is done----------------------')

    return model

