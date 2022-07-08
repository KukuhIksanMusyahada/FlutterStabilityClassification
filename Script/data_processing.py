import os
import re
import numpy as np
import pandas as pd


from Script import path_handler as ph




def read_df(path):
    try:
        df = pd.read_csv(path, usecols=['plunge(airfoil)']).to_numpy()
    except ValueError:
        df = pd.read_csv(path, usecols=['plunge_airfoil']).to_numpy()

    return df


def gradien(array):
    ''' Calculate the gradien of an array on each point. Assuming that each 
    horizontal axis have the same step so the gradien is just the delta of its current value
    with the  next step'''
    grad = []
    for row in range(array.shape[0]-1):
        delta = array[row+1]- array[row]
        grad.append(delta)
    return grad

def find_turn_point(grad):
    before_sign = -1
    pos_index = []
    for index, elem in enumerate(grad):
        if elem !=0:
            sign = elem/ abs(elem)
        else:
            sign = before_sign
        if sign < before_sign:
            pos_index.append(index)
        before_sign = sign

    return pos_index

def divergence_test(array, index, wait =3):
    count = 0
    val_before= array[index[0]]
    divergen = 0
    for i in index:
        value = array[i]
        if value > val_before:
            count +=1
        if count == wait:
            divergen= 1
        val_before = value
    return divergen



def extract_mach_and_vf(file: str):
    pattern = r'M_([0-9\.]*)_VF_([0-9\.]*).csv'
    result  = re.match(pattern, file)

    return float(result.group(1)), float(result.group(2))

def scan(path=ph.get_raw_data()):
    mach = []
    vf = []
    flutter = []
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                result = extract_mach_and_vf(file)
                mach.append(result[0])
                vf.append(result[1])
                print(file)
                df = read_df(os.path.join(dir_path, file))
                grad= gradien(df)
                turn_point = find_turn_point(grad)
                divergen = divergence_test(df, turn_point)
                flutter.append(divergen)
    list = [mach, vf, flutter]
    return np.array(list).T


if __name__=='__main__':
    scan()