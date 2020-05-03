import os
import numpy as np
import pandas as pd
import joblib

model=joblib.load("Model/Classification.model")
df = pd.read_excel('Data/OBCs.xlsx','Classification')
ind1 = ["H1","H2","H3","H4","H5","H6","H7","H8"]
ind2 = ["M1T1", "M2T1", "M3T1", "M4T1", "M5T1", "M6T1", "M7T1", "M8T1", "M9T1", "M10T1"]
ind3 = ["S1T1", "S2T1", "S3T1", "S4T1", "S5T1", "S6T1", "S7T1" ]
ind4 = ["F1T1", "F2T1", "F3T1", "F4T1", "F5T1", "F1T2", "F2T2", "F3T2", "F4T2", "F5T2" , "F1T3", "F2T3", "F3T3", "F4T3", "F5T3", "F1T4", "F2T4", "F3T4", "F4T4", "F5T4"]
ind = ind1 + ind2+ ind3+ ind4
lis=[0,1,2,3]
X_test = df.loc[lis, ind].T.values
y_test_scores = model.predict_proba(X_test)
print(y_test_scores)





















