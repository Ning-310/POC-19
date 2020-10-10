# Prioritization of Optimal biomarker Combinations for COVID-19 (POC-19)

**Note: Based on the plasma proteomic data of Cohort 1, we developed a new computational pipeline named Prioritization of Optimal biomarker Combinations for COVID-19 (POC-19) for identifying potential biomarker combinations to classify COVID-19 cases. POC-19 contains three steps, including differential protein reservation (DPR) to select 112 highly ranked DEPs, candidate biomarker selection (CBS) to generate 1000 groups of initial biomarker combinations, and final biomarker determination (FBD) to get the protein combination with the highest area under the curve (AUC) value from the 5-fold cross-validation. In the step of FBD, a widely used machine learning algorithm, penalized logistic regression (PLR), was used for model training and parameter optimization.


## Requirements

The main requirements are listed below:

* Python 3.7
* Numpy
* Scikit-Learn
* Joblib
* Keras
* Pandas


## The description of POC-19 source codes

* DPR.py

    The code is used to select 112 highly ranked DEPs.

* FBD_Classification.py

    The code is used to generate 1000 groups of initial biomarker combinations, and get the final biomarker combination for classifying COVID-19 patients.
* FBD_Severe to fatal.py

    The code is used to generate 1000 groups of initial biomarker combinations, and get the final biomarker combination for predicting severe to fatal outcome.
* FBD_Mild to severe.py

    The code is used to generate 1000 groups of initial biomarker combinations, and get the final biomarker combination for predicting mild to severe outcome.
* FBD_Cured.py

    The code is used to generate 1000 groups of initial biomarker combinations, and get the final biomarker combination for predicting COVID-19 patients curable from the disease.

* Train_Classification.py

    The code is used to train the model for classifying COVID-19 patients.
* Train_Severe to fatal.py

    The code is used to train the model for predicting severe to fatal outcome.
* Train_Mild to severe.py

    The code is used to train the model for predicting mild to severe outcome.
* Train_Cured.py

    The code is used to train the model for predicting COVID-19 patients curable from the disease.
* Test_Classification.py

    The code is used to test the model for classifying COVID-19 patients.
* Test_Severe to fatal.py

    The code is used to test the model for predicting severe to fatal outcome.
* Test_Mild to severe.py

    The code is used to test the model for predicting mild to severe outcome.
* Test_Cured.py

    The code is used to test the model for predicting COVID-19 patients curable from the disease.
	
* ROC.py

    The code is used to illustrate the receiver operating characteristic (ROC) curve based on sensitivity and 1-specificity scores, and compute the AUC value.


## The models in POC-19

* Classification.model 

    The model is used for the classification of COVID-19 patient.
* Severe to fatal.model 

    The model is used for the prediction of severe to fatal outcome.
* Mild to severe.model 

    The model is used for the prediction of mild to severe outcome.
* Cured.model 

    The model is used for the prediction of COVID-19 patients curable from the disease.
