# -*- coding: utf-8 -*-

import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from time import time
from time import sleep
from scipy import ndimage
from tqdm import tqdm
import cv2 #REQUIRES "conda install opencv"
from random import shuffle 
import keras 
from keras.models import model_from_json
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.layers import BatchNormalization
from keras.optimizers import *
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, KFold
from keras.callbacks import ModelCheckpoint, EarlyStopping 
from pathlib import Path
from subprocess import check_output
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from keras import backend as K

#Used for running the program on GPU via CUDA. 
K.tensorflow_backend._get_available_gpus()


#The following hash table stores the names of questions and 
#their corresponsing image size for training (For instance, 
#if the image size is 64x64, then the value to the key is 64.)
#Input data files are available in the "../input/" directory.
#In the loop, the program will read folders by the name of their key value 
#inside the directory that this file is contained in. 
Questions ={
  "Q1": 64,
  "Q2": 128,
  "Q3": 64,
  "Q4": 64,
  "Q7": 128,
  "Q8": 128,
  "Q9": 128,
  "Q17": 128,
  "Q18": 64
}


#these variables are reset after each loop iteration. 
num_true = 0
num_false = 0

#This method will read each image for training and split it 
#based on the filename to assign a label.
# An image called 'n_1.png' for example is a training
#image that indicates an incorrect response. 
def label_correct_answers(img):
    global num_true
    global num_false
    prefix = img.split('_')[0]
    if prefix == 'y':
        num_true += 1
        label = np.array([0,1])
    elif prefix == 'n':
        num_false += 1
        label = np.array([1,0])
    return label

#This method reads each file in a folder and reads it as an array of 
#values. cv2.imread with setting '0' will read the image as grayscale. 
#image is then resized to the specified size and labeled using the method above. 
    #datapath: path of the folder containing image files. 
    #files: An array of files in the directory
    #size: Specified size from the hash set 'Questions'
def labeledData(datapath, files, size):
    images = []
    for i in files:
        path = os.path.join(datapath, i)
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (size, size))
        images.append([np.array(img), label_correct_answers(i)])
    return images


#!Important	
#Path to data folder here: 
data_folder = "C:/Users/MinearLab-Ninja/Desktop/Topo/"


##Begin Creating Models
for key in Questions:
    QuestionData = data_folder + "AllData/" + key + "/"
    all_files = [f for f in os.listdir(QuestionData) if not f.startswith(".")]

    sz = Questions[key]
    training_images = labeledData(QuestionData, all_files, sz)
    tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,sz,sz,1)
    tr_lbl_data = np.array([i[1] for i in training_images])

    #Define model variables 
    kernel = 3
    strides = 2
    seed = 5
    
    print("Working with {0} images".format(len(all_files)))
    
    ##Display photos
    print("Image examples: ")
    for i in range(1, 9):
        print(training_images[i])
        display(_Imgdis(filename=QuestionData + all_files[i], width=240, height=320))
    
 
    #Use Keras Sequential and add convolution layers
    model = Sequential()
  

    model.add(Conv2D(filters=32, kernel_size=kernel, strides = strides, padding = 'same', activation = 'relu', input_shape=[sz,sz,1]))
    model.add(MaxPool2D(pool_size=5,padding='same'))
  
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=kernel, strides = strides, padding = 'same', activation = 'relu', input_shape=[sz,sz,1]))
    model.add(MaxPool2D(pool_size=5, padding = 'same'))
  
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=kernel, strides = strides, padding = 'same', activation = 'relu', input_shape=[sz,sz,1]))
    model.add(MaxPool2D(pool_size=5, padding = 'same'))

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu', input_shape=[sz,sz,1]))
    model.add(Dropout(rate=0.25))
    model.add(Dense(2,activation='sigmoid')) #2 labels, so sigmoid rather than softmax

  
    #Adam with defaults. Refer to Keras documentaiton for 
    #parameter tuning and other optimizer options. 
    optimizer = Adam()
    #DEFAULTS @ADAM -> lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False

    #Monitor epochs and stop to prevent overfitting
    earlyStop = EarlyStopping(monitor='loss', patience=35, verbose=0) 
    
    #Monitor checkpoint to grab the best model
    #checkpoint = ModelCheckpoint('cp.h5', verbose=0, save_best_only=True)
    
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    
    #Set Verbose to 0 in order to reduce text output if creating many models. 
    model.fit(x=tr_img_data,y=tr_lbl_data, epochs=200,batch_size=num_false+num_true, callbacks=[earlyStop], verbose = 1)
    model.summary()
    
    num_true = 0
    num_false = 0
    
    #FOR SAVING 
    # serialize model to JSON
    model_json = model.to_json()
    with open(data_folder + "/newModels/" + key + ".json", "w") as json_file:
        json_file.write(model_json)
        ## serialize weights to HDF5
        model.save_weights(data_folder + "newModels/" + key + ".h5")
        print("Saved model to disk")

    
