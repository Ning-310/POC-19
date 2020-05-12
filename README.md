# POC-19

**Note: To identify optimal biomarker combinations (OBCs) for accurately classifying different types of samples, we developed a machine learning-based algorithm named prioritization of OBCs for COVID-19 (POC-19). Based on the plasma proteomic data, we developed a new algorithm named POC-19 for identifying potential OBCs to classify COVID-19 patients, to predict severe to fatal outcome, to predict mild to severe outcome, and to predict COVID-19 patients curable from the disease, respectively. POC-19 contains three steps, including differential protein reservation (DPR) to select 112 highly ranked differentially expressed proteins (DEPs), candidate OBC selection (COS) to generate 1000 groups of initial OBCs, and final OBC determination (FOD) to get the protein combination with the highest area under the curve (AUC) value from the 5-fold cross-validation. In the step of FOD, a widely used machine learning algorithm, penalized logistic regression (PLR), was used for model training and parameter optimization.


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

* FOD_Classification.py

    The code is used to generate 1000 groups of initial OBCs, and get the final OBC for classifying COVID-19 patients.
* FOD_Severe to fatal.py

    The code is used to generate 1000 groups of initial OBCs, and get the final OBC for predicting severe to fatal outcome.
* FOD_Mild to severe.py

    The code is used to generate 1000 groups of initial OBCs, and get the final OBC for predicting mild to severe outcome.
* FOD_Cured.py

    The code is used to generate 1000 groups of initial OBCs, and get the final OBC for predicting COVID-19 patients curable from the disease.

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
