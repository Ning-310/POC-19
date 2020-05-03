# POC-19

**Note: To identify optimal biomarker combinations (OBCs) for accurately classifying different types of samples, we developed a machine learning-based algorithm named prioritization of OBCs for COVID-19 (POC-19). Based on the plasma proteomic data, we developed a new algorithm named POC-19 for identifying potential OBCs to classify COVID-19 patients, to predict severe to fatal outcome, to predict mild to severe outcome, and to predict COVID-19 patients curable from the disease, respectively. POC-19 contains three steps, including differential protein reservation (DPR) to select 112 highly ranked DEPs, candidate OBC selection (COS) to generate 1000 groups of initial OBCs, and final OBC determination (FOD) to get the protein combination with the highest area under the curve (AUC) value from the 5-fold cross-validation. In the step of FOD, a widely used machine learning algorithm, penalized logistic regression (PLR), was used for model training and parameter optimization.


## The raw mass spectrometric data

The raw mass spectrometric data of plasma proteomes was submitted into the integrated proteome resources (iProX, http://www.iprox.org/) with the dataset identifier XXX.


## Requirements

The main requirements are listed below:

* Python 3.7
* Numpy
* Scikit-Learn
* Joblib
* Keras
* Pandas


## The description of POC-19 source code

* Train_Classification.py

    The code is used to train model for the classification of COVID-19 patient classification.
*  Train_Severe to fatal.py

    The code is used to train model for the prediction of severe to fatal outcome.
* Train_Mild to severe.py

    The code is used to train model for the prediction of mild to severe outcome.
* Train_Cured.py

    The code is used to train model for the prediction of COVID-19 patients curable from the disease.
* Test_Classification.py

    The code is used to test model for the classification of COVID-19 patient classification.
* Test_Severe to fatal.py

    The code is used to test model for the prediction of severe to fatal outcome.
* Test_Mild to severe.py

    The code is used to test model for the prediction of mild to severe outcome.
* Test_Cured.py

The code is used to test model for the prediction of COVID-19 patients curable from the disease.


## The models in POC-19

* Classification.model 

    The model is used for the classification of COVID-19 patient classification.
* Severe to fatal.model 

    The model is used for the prediction of severe to fatal outcome.
* Mild to severe.model 

    The model is used for the prediction of mild to severe outcome.
* Cured.model 

    The model is used for the prediction of COVID-19 patients curable from the disease.
