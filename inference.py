import os
import numpy as np

import trainer

from Script import models
from Script import path_handler as ph

def inference():
    mach = float(input('please input mach number: '))
    vf = float(input('please input Flutter Speed: '))
    total_models = len(os.listdir(ph.get_models_data()))
    if total_models == 0:
        trainer.run_trainer()
    total_models = len(os.listdir(ph.get_models_data()))
    model_number = input(f'Total Models Detected: {total_models} \n Input which model will be used to inference')
    model_name = f'ModelFlutterClassification{model_number}'
    path = os.path.join(ph.get_models_data(), model_name)
    model, history = models.loadmodel(path)
    prediction = models.predict(model, mach, vf)
    if prediction == 0.0:
        output_string ='Airfoil does not experience flutter phenomena'
    else:
        output_string ='Airfoil experience flutter phenomena'
    print(output_string)


if __name__=='__main__':
    inference()





