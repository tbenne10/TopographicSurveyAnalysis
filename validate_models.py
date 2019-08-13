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


#Path to directory containing folder keys here: 
data_folder = ""



##---K-Folds Cross Validation---##
##------------------------------##
for key in Questions:
    print("====Question:" + key + "====")

    ## specify data in path in loop
    data = Path(data_folder + key + "/")
    ##
    
    files = [f for f in os.listdir(data) if not f.startswith(".")]
    

    ##
    sz = Questions[key]
    ##sz = 64 #Reshape size 
    
    #These are to print how many of each label image is in the folder
    num_true = 0
    num_false = 0
    
     
    ##reshaping it again to -1*size*size*1. 1 represents the color code as grayscale
    #########
    images = labeledData(data, files, sz)
    shuffle(images)
    img_data = np.array([i[0] for i in images]).reshape(-1,sz,sz,1)
    lbl_data = np.array([i[1] for i in images])
    #########
    
     
    kernel = 3
    strides = 2
    seed = 5
    
    #Split k-fold into desired number n_splits. 
    kfold = KFold(n_splits=12, shuffle=True, random_state=seed)
    kfold.get_n_splits(img_data)
    
    #Store-variables
    scores = []
    losses = []
    probabilities = []
    classes = []
    label_data_1D = []
    
    #Run validation
    #IMPORTANT NOTE: Validation will take a very long time. 
    #It may appear frozen but is in fact processing. 
    for train, test in kfold.split(img_data):
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
    
      
        optimizer = Adam()
        #DEFAULTS @ADAM -> lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
      
        #Monitor epochs and stop to prevent overfitting
        earlyStop = EarlyStopping(monitor='loss', patience=35, verbose=0) 
        
        #Monitor checkpoint to grab the best model
        #checkpoint = ModelCheckpoint('cp.h5', verbose=0, save_best_only=True)
        
        model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        ##
        model.fit(x=img_data[train],y=lbl_data[train],epochs=200,batch_size=num_false+num_true, callbacks=[earlyStop], verbose = 0)
        #model.summary()
        score = model.evaluate(img_data[test], lbl_data[test], verbose = 0)
        
        
        new_probabilities = model.predict(img_data[test], verbose=0)
        new_classes = model.predict_classes(img_data[test], verbose=0)
        new_probabilities = new_probabilities[:, 0]
        probabilities.extend(new_probabilities)

        #Used to flip classifactions as they are backwards
        for i, elem in enumerate(new_classes):
            new_classes[i] = (1, 0)[new_classes[i] == 1]
        classes.extend(new_classes)

        new_lbl_data_1D = lbl_data[test]
        new_lbl_data_1D = new_lbl_data_1D[:,0]
        label_data_1D.extend(new_lbl_data_1D)

        scores.append(score[1]*100)
        losses.append(score[0])
    #Accuracy: 
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
    print(np.mean(losses), np.std(losses))
    precision = precision_score(label_data_1D,classes)
    recall = recall_score(label_data_1D,classes)
    f1 = f1_score(label_data_1D,classes)
    ck = cohen_kappa_score(label_data_1D,classes)
    print("Precision: %.4f%%" % precision)
    print("Recall: %.4f%%" % recall)
    print("F1: %.4f%%" % f1)
    print("Cohen's Kappa: %.4f%%" % ck)



##---Leave One Out Cross Validation---##
##------------------------------------##
for key in Questions:
print("====Question:" + key + "====")

    ## specify data in path in loop
    data = Path(data_folder + key + "/")
    ##
    
    files = [f for f in os.listdir(data) if not f.startswith(".")]
    

    ##
    sz = Questions[key]
    ##sz = 64 #Reshape size 
    
    #These are to print how many of each label image is in the folder
    num_true = 0
    num_false = 0
    
     
    ##reshaping it again to -1*size*size*1. 1 represents the color code as grayscale
    #########
    images = labeledData(data, files, sz)
    shuffle(images)
    img_data = np.array([i[0] for i in images]).reshape(-1,sz,sz,1)
    lbl_data = np.array([i[1] for i in images])
    #########
    
     
    kernel = 3
    strides = 2
    seed = 5
    
    #Split k-fold into desired number n_splits. 
    kfold = KFold(n_splits=12, shuffle=True, random_state=seed)
    kfold.get_n_splits(img_data)
    
    #Store-variables
    scores = []
    losses = []
    probabilities = []
    classes = []
    label_data_1D = []
    
    #Run Cross validation
    #Note: This will take way longer than K-folds cross validation. 
    #Expect this program to run for at least a day on a 
    ## sample set of 500 images 
    for train, test in loo.split(img_data):
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

        optimizer = Adam()
      
        #Monitor epochs and stop to prevent overfitting
        earlyStop = EarlyStopping(monitor='loss', patience=35, verbose=0) 
        
        #Monitor checkpoint to grab the best model
        #checkpoint = ModelCheckpoint('cp.h5', verbose=0, save_best_only=True)
        
        model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        ##
        model.fit(x=img_data[train],y=lbl_data[train],epochs=200,batch_size=num_false+num_true, callbacks=[earlyStop], verbose = 0)
        #model.summary()
        score = model.evaluate(img_data[test], lbl_data[test], verbose = 0)
        
        new_probabilities = model.predict(img_data[test], verbose=0)
        new_classes = model.predict_classes(img_data[test], verbose=0)
        new_probabilities = new_probabilities[:, 0]
        probabilities.extend(new_probabilities)

        for i, elem in enumerate(new_classes):
            new_classes[i] = ("1", "0")[new_classes[i] == 1]
        classes.extend(new_classes)

        new_lbl_data_1D = lbl_data[test]
        new_lbl_data_1D = new_lbl_data_1D[:,0]
        label_data_1D.extend(new_lbl_data_1D)
        
        scores.append(score[1]*100)
        losses.append(score[0])
    #Accuracy
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
    #Loss
    print(np.mean(losses), np.std(losses))
    precision = precision_score(label_data_1D,classes)
    recall = recall_score(label_data_1D,classes)
    f1 = f1_score(label_data_1D,classes)
    ck = cohen_kappa_score(label_data_1D,classes)
    print("Precision: %.4f%%" % precision)
    print("Recall: %.4f%%" % recall)
    print("F1: %.4f%%"  % f1)
    print("Cohen's Kappa: %.4f%%" % ck)

