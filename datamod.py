
# coding: utf-8

# In[ ]:


import csv
import sys
path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if path in sys.path:
    print('Existing interference cleared')
    sys.path.remove(path)
import cv2
import numpy as np
import scipy.misc
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


BATCH_SIZE = 64
DELTA_STEER = 0.25

def extract():
    lines = []
    with open ('./data/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        
    shuffle(lines)
    shuffle(lines)
    
    X_train, X_test = train_test_split(lines, test_size = 0.2)
    X_train_nos = len(X_train)
    X_test_nos = len(X_test)
    
    print("Number of training samples:", X_train_nos)
    print("Number of testing samples:", X_test_nos)
    
    return X_train, X_train_nos, X_test, X_test_nos 

def flip(img, angle):
    return np.fliplr(img), -1*angle

def crop(img, crop_factor = 0.4):
    return img[np.ceil(crop_factor * img.shape[0]):, :]

def reshape(img, shape = (64, 64,3)):
    return scipy.misc.imresize(img, shape)

def augment(sample, X_data, y_data):
    center = cv2.imread(sample[0])
    left = cv2.imread(sample[1])
    right = cv2.imread(sample[2])
    steering = float(sample[3])
    
    leftSteer = steering - DELTA_STEER
    rightSteer = steering + DELTA_STEER
    
    X_data.append(center)
    y_data.append([steering])
    
    X_data.append(left)
    y_data.append([steering + DELTA_STEER])
    
    X_data.append(center)
    y_data.append([steering - DELTA_STEER])
    
    leftFlip, leftAngle = flip(left, leftSteer)
    rightFlip, rightAngle = flip(right, rightSteer)
    
    X_data.append(leftFlip)
    X_data.append([leftAngle])
    
    X_data.append(rightFlip)
    X_data.append([rightAngle])
    
X_train, X_train_nos, X_test, X_test_nos = extract()

def get_data(data_type = 'train'):
    while True:
        if data_type == 'train':
            data, samples = X_train, X_train_nos
        else:
            data, samples = X_test, X_test_nos
            
        data = shuffle(data)
        for i in range(0, samples, BATCH_SIZE):
            X_data = []
            y_data = []
            
            for sample in data[i:i + BATCH_SIZE]:
                augment(sample, X_data, y_data)
                
            print(len(X_data), len(y_data ))
            X_data = np.array(X_data)
            y_data = np.array(y_data)
            
            yield shuffle(X_data, y_data)
            
if __name__ == '__main__':
    print('./data/data/driving_log.csv')
            
    
    
    
    

