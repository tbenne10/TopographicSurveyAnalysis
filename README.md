# TopographicSurveyAnalysis
CNN classification for scoring topographic images to measure spatial awareness
![alt text](https://github.com/tbenne10/TopographicSurveyAnalysis/blob/master/Assessment_page/q1_1.png)
![alt text](https://github.com/tbenne10/TopographicSurveyAnalysis/blob/master/Assessment_page/q1_2.png)
This repo is designed to provide an example of how to implement a custom topographic survey setup that scores submitted responses using a convolutional neural network. Updated code and information is pending. The assessment page was combined into a single page to give a visual of the entire survey. 

*Please refer to the paper for notes regarding PHP configuration on web hosting providers as some hosts do not allow large transfer sizes by default*. 

These scripts will not work by default without configuration, specifically in choosing directories to operate on. Only run_participants.py contains a GUI to allow directory selection. This is because the other files are intended to be run in an IDE (preferably Spyder) with adjustments made as needed. Additionally, run_participants.py will rely on the structure of the CSV file that it is made to run. 

Assessment_page includes the survey itself and scripting to transfer drawn responses. to a mysql database. 
Retrieve_page includes a page to retrieve individual student responses for review or hand scoring. 

create_models.py - Create a model for each question given a set of images and save files to .h5 and .json. 

run_participants.py - Load a CSV file containing string URIs of responses and score each individual response, produce an output file of results. Contains a GUI to specify the input/output directories and the prefix of responses to read (I.E, if only responses begining with "1" are needed to score, then score only these entries.)

validate_models.py - K-fold and LOOC scoring of models on the same image sets used for create_models.py. Used in evalutating model performance. 
