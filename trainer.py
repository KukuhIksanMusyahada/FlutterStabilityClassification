import numpy as np


from Script import data_processing as dp
from Script import models


def run_trainer():
    data = dp.scan()
    label = data[:,2]
    input = data[:,:2]
    model, history = models.model(input, label, max_epoch=300)
    models.savemodel(model, history)
    print('----------------------training model is done----------------------')

    return model

