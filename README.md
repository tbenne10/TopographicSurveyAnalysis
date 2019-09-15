# TopographicSurveyAnalysis
<br>
CNN classification for scoring topographic images to measure spatial awareness
<br>
<p align="center">
<img src="https://github.com/tbenne10/TopographicSurveyAnalysis/blob/master/Assessment_page/q1_1.png" width="256" height="256">
<img src="https://github.com/tbenne10/TopographicSurveyAnalysis/blob/master/Assessment_page/q1_2.png" width="256" height="256">
</p>
<br>
This repo is designed to provide an example of how to implement a custom topographic survey setup that scores submitted responses using a convolutional neural network. The assessment page in this sample was combined into a single page to give a visual of the entire survey. 

*Please refer to the paper for notes regarding PHP configuration on web hosting providers as some hosts do not allow large transfer sizes by default*. 

These scripts will not work by default without configuration, as directories and models need to be specified for custom datasets. Only run_participants.py contains a GUI to allow directory selection.  Additionally, run_participants.py will require headers on required responses for custom data. 

Assessment_page includes the survey itself and scripting to transfer drawn responses. to a mysql database. 
Retrieve_page includes a page to retrieve individual student responses for review or hand scoring. 

create_models.py - Create a model for each question given a set of images and save files to .h5 and .json. 

run_participants.py - Load a CSV file containing string URIs of responses and score each individual response, produce an output file of results. Contains a GUI to specify the input/output directories and the prefix of responses to read (I.E, if only responses begining with "1" are needed to score, then score only these entries.)

validate_models.py - K-fold and LOOC scoring of models on the same image sets used for create_models.py. Used in evalutating model performance. 
