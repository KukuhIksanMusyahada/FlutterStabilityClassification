import os
import numpy as np
import datetime

import trainer

from Script import models
from Script import path_handler as ph
from Script import data_processing as dp

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
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'PREDICTION USING CLASSIFICATION FLUTTER MODEL START AT {time_now}')
    model, _ = models.loadmodel(path)
    prediction = models.predict(model, mach, vf)
    if prediction == 0.0:
        output_string ='Airfoil does not experience flutter phenomena'
    else:
        output_string ='Airfoil experience flutter phenomena'
    print(output_string)
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'PREDICTION DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO DO ONE PREDICTION IS {delta_time}')

def multiple_inference():
    mach_min = float(input('please input mach number minimum: '))
    mach_max = float(input('please input mach number maximum: '))
    vf_min = float(input('please input Flutter Speed minimum: '))
    vf_max = float(input('please input Flutter Speed maximum: '))
    total_models = len(os.listdir(ph.get_models_data()))
    if total_models == 0:
        trainer.run_trainer()
    total_models = len(os.listdir(ph.get_models_data()))
    model_number = input(f'Total Models Detected: {total_models} \n Input which model will be used to inference ')
    if int(model_number) == 0:
        trainer.run_trainer()
        total_models = len(os.listdir(ph.get_models_data()))
        model_number = input(f'Total Models Detected: {total_models} \n Input which model will be used to inference ')
    model_name = f'ModelFlutterClassification{model_number}'
    path = os.path.join(ph.get_models_data(), model_name)
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'PREDICTION USING CLASSIFICATION FLUTTER MODEL START AT {time_now}')
    model, _ = models.loadmodel(path)
    Mach = []
    Vf = []
    Stability = []
    for mach in np.arange(mach_min,mach_max,step=0.01):
        for vf in np.arange(vf_min, vf_max,step=0.1):
            prediction = models.predict(model, mach, vf)
            Mach.append(mach)
            Vf.append(vf)
            Stability.append(prediction)
            if prediction == 0.0:
                output_string ='Airfoil does not experience flutter phenomena'
            else:
                output_string ='Airfoil experience flutter phenomena'
            print(output_string)
    pred_arr = np.array([Mach, Vf, Stability])
    dp.save_processed_data(pred_arr, path=ph.get_results_data(), names='Prediction.csv')
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'PREDICTION DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO COMPLETE ALL PREDICTION IS {delta_time}')


if __name__=='__main__':
    # inference() # comment if want to predict multiple times 
    multiple_inference() # comment if want to predict single times 





