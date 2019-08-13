#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#This script scores results from each student
#Drawn images are downloaded from a .csv file, converted from string base64 encoding,
#and scored against machine learning models saved to disk


import csv
import os
#import file
import cv2
import re
import base64
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import cohen_kappa_score
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter import simpledialog

import sys
import os.path



#Specify max size due to large size of Base64 images
csv.field_size_limit(sys.maxsize)
    
#Specify which questions are drawn images. Their associated value is the 
#size of the image used in data preprocessing for the machine learning model. 
drawn_images ={
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

#init variables
filename = ""
filedir = ""
modeldir = ""
prefix = ""

##Retrieve the CSV file to read image data
def getCSVfile():
    global filename 
    global filedir
    filename = askopenfilename()
    filedir = os.path.abspath(os.path.join(filename, os.pardir))
    filedir += "/"
    print(filedir)
    
#Select the directory containing H5 and JSON model files. 
def getModelDir():
    global modeldir
    modeldir = askdirectory() 
    modeldir += "/"

#Select a prefix to read only specific records starting with the prefix. 
def getPrefix():
    global prefix
    prefix = simpledialog.askstring("input string", "Enter an ID prefix:")


#Run program and create two response CSV files. 
def Start():
    #for indexing
    drawn_images_list = list(drawn_images)
    
    #Load models: 
    models = []
    print("Loading models... This may take a moment")
    for key in drawn_images: 
        json_file_path = modeldir + key + ".json"
        weight_file_path = modeldir + key + ".h5"
        json_file = open(json_file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weight_file_path)
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        models.append(loaded_model)
        print(f"Loaded model {key}...")
    print("Done loading models")
    
    #Function to process each individual image
    #Returns a prediction score of 1 or 0.
    def process_image(Qnum, uri):
        print(f"Processing image: {Qnum}")
        #Ensure value exists
        if(uri == None): return 0
        #Grab value to resize image
        size = drawn_images[Qnum]
        #create image file as temporary
        path = modeldir + "temp.png"
        img = open(path, "wb")
        img.write(base64.b64decode(uri))
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (size, size))
        img_reshape = np.array(img).reshape(-1,size,size,1)
        #Run image against model
        print("Acc: ")
        print (models[drawn_images_list.index(Qnum)].predict(img_reshape))
        pred = models[drawn_images_list.index(Qnum)].predict_classes(img_reshape)[0] 
        #This flips the class as the prediction score is on the opposite entry. 
        pred = ("1", "0")[pred == 0]
        pred_array = models[drawn_images_list.index(Qnum)].predict(img_reshape)
        #Remove the image to make room for another
        os.remove(modeldir + "temp.png")
        eps = .15 #Min. acceptable criterion
        if(1-np.amax(pred_array) > eps):
            return 'f'
        return pred
    
    #Open two files, one for response scores and the other for written
    #question responses. Each file name is appended with a prefix if 
    #a prefix is give. 
    data = open(filename, 'r')
    responses = open(filedir + 'responses_pref' + prefix + '.csv', 'w') 
    Wresponses = open(filedir + 'Wresponses_pref' + prefix + '.csv', 'w')     
    read_data = csv.reader(data, delimiter=',')
    write_responses = csv.writer(responses, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_ALL)
    write_Wresponses = csv.writer(Wresponses, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_ALL)
    line_count = 0
    for row in read_data:
        if row[0].startswith(prefix, 0, len(prefix)):
            if line_count == 0:
                line_count += 1
                write_responses.writerow(['Number','Participant', 'Q1_drawn', 'Q2_drawn', 
                             'Q3_drawn', 'Q4_drawn', 'Q7_drawn', 'Q8_drawn', 
                             'Q9_drawn', 'Q17_drawn', 'Q18_drawn', 'Q5_response', 
                             'Q5_correct_response', 'Q5_accuracy','Q6_response', 
                             'Q6_correct_response', 'Q6_accuracy','Q10_1_response', 
                             'Q10_1_correct_response','Q10_1_accuracy', 'Q10_2_response', 
                             'Q10_2_correct_response', 'Q10_2_accuracy', 'Q11_response', 
                             'Q11_correct_response', 'Q11_accuracy', 'Q12_response', 
                             'Q12_correct_response','Q12_accuracy', 'Q13_response', 
                             'Q13_correct_response', 'Q13_accuracy', 'Q14_1_response', 
                             'Q14_1_correct_response', 'Q14_1_accuracy', 'Q14_2_response',
                             'Q14_2_correct_response','Q14_2_accuracy', 'Q15_AB_response', 
                             'Q15_AB_correct_response','Q15_AB_accuracy', 'Q15_AD_response',
                             'Q15_AD_correct_response','Q15_AD_accuracy', 'Q15_BC_response',
                             'Q15_BC_correct_response','Q15_BC_accuracy', 'Q15_CD_response',
                             'Q15_CD_correct_response','Q15_CD_accuracy','Q15_BD_response',
                             'Q15_BD_correct_response','Q15_BD_accuracy', 'Total', 'Date Submitted']) 
                write_Wresponses.writerow(['Number','Participant','Q2_written', 'Q7_written', 'Q8_written', 
                            'Q9_written', 'Q14_2_written', 'Q17_written', 'Q18_written', 'Date Submitted'])      
            else: 
                #Resp used for responses, respW for written reponses
                resp = []
                respW = []
                count = 0
                ##logic here
                #append number and name
                resp.append(line_count)
                resp.append(row[0])
                respW.append(line_count)
                respW.append(row[0])
                #append drawn images
                for x in drawn_images: 
                    y = row[drawn_images_list.index(x) + 2].split(',')
                    if(len(y) > 1):
                        resp.append(process_image(x, y[1]))
                    else: resp.append("N/A")
                    #print(row[drawn_images_list.index(x) + 2])
                ##Q5
                ##TODO: find locations of data in row
                resp.append(row[23])
                resp.append("A")
                resp.append(("0", "1")[row[23] == "A"])
                #Q6
                resp.append(row[24])
                resp.append("A")
                resp.append(("0", "1")[row[24] == "A"])
                #Q10_1
                resp.append(row[15])
                resp.append("Josh")
                resp.append(("0", "1")["josh" in row[15].lower()])
                #Q10_2
                resp.append(row[18])
                resp.append("josh")
                resp.append(("0", "1")["josh" in row[18].lower()])
                #Q11
                resp.append(row[25])
                resp.append("B")
                resp.append(("0", "1")[row[25] == "B"])
                #Q12
                resp.append(row[26])
                resp.append("B")
                resp.append(("0", "1")[row[26] == "B"])
                #Q13
                resp.append(row[17])
                resp.append("40")
                resp.append(("0", "1")["40" in row[19]])
                #Q14_1
                resp.append(row[18])
                resp.append("Josh")
                resp.append(("0", "1")["josh" in row[18].lower()])
                #Q15
                ##Refer to re library for digit extraction
                resp.append(row[20])
                resp.append("7040-7080")
                val = re.findall("\d+", row[20]) 
                if(len(val) > 0): 
                    resp.append(("0", "1")[int(val[0]) >= 7040 and int(val[0]) <= 7080])
                else: resp.append("0")
                #Q16: 
                resp.append(row[27])
                resp.append("yes")
                resp.append(("0", "1")[row[27] == "yes"])
                resp.append(row[28])
                resp.append("yes")
                resp.append(("0", "1")[row[28] == "yes"])
                resp.append(row[29])
                resp.append("yes")
                resp.append(("0", "1")[row[29] == "yes"])
                resp.append(row[30])
                resp.append("no")
                resp.append(("0", "1")[row[30] == "no"])
                resp.append(row[31])
                resp.append("yes")
                resp.append(("0", "1")[row[31] == "yes"])
                ##WRITE ALL THE WRITTEN RESPONSES HERE
                respW.append(row[11])
                respW.append(row[12])
                respW.append(row[13])
                respW.append(row[14])
                respW.append(row[16])
                respW.append(row[19])
                respW.append(row[21])
                respW.append(row[22])
                #Total
                sum = 0
                for x in resp:
                    if x == "1":
                        sum += 1
                resp.append(sum)
                #Dates
                resp.append(row[32])
                respW.append(row[32])
                #Write rows
                write_responses.writerow(resp)
                write_Wresponses.writerow(respW)
                line_count += 1
    print(f"Finished, {line_count} rows read: ")
    data.close()
    responses.close()


##Run GUI 
root = tk.Tk()
root.wm_title("Run Participant Data")
selectCsv = tk.Button(root, text='Select CSV file', width=25, command=getCSVfile)
selectCsv.pack()
selectDirectory = tk.Button(root, text='Select model directory', width=25, command=getModelDir)
selectDirectory.pack()
selectPrefix = tk.Button(root, text='Select an ID prefix', width=25, command=getPrefix)
selectPrefix.pack()
startButton = tk.Button(root, text='Start', width=25, command=Start)
startButton.pack()

root.mainloop()