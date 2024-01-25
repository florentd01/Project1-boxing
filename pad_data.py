import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import mediapipe as mp
import csv
import os
import pandas as pd
import pickle


view = 'side' # side
quality = ''
#quality = '1.5_'

total_data = []

for participant in range(1,17):
    with open(f'Landmarks/{view}_{quality}/{participant}{view}_{quality}landmarks.pkl', 'rb') as file:
        data = pickle.load(file)
        total_data.extend(data)
        print(participant)

def pad_data(data):
    max_size = max(len(lst) for lst in data)
    print(max_size)
    
    for i in range(len(data)):
        for j in range(max_size - len(data[i])):
            padding_list = [-10] * len(data[i][0])
            data[i].append(padding_list)
    
    return data


padded_data = pad_data(total_data)

with open(f'Landmarks/{view}_{quality}/{view}_{quality}final_landmarks.pkl', 'wb') as file:
    pickle.dump(padded_data, file)
    print("DONE")