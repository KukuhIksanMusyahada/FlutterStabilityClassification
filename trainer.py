import numpy as np


from Script import data_processing as dp
from Script import models


def trainer():
    data = dp.scan()
    label = data[:,2]
    input = data[:,:2]
    model = models.model(input, label, max_epoch=300)

    return model


if __name__=='__main__':
    model = trainer()
    x = model.predict([[0.63, 2.3]])
    y = np.round(x[0][0])
    print(x)
    print(y)