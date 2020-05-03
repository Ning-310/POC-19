import os
import numpy as np
import pandas as pd
import joblib

model=joblib.load("Model/Cured.model")
df = pd.read_excel('Data/OBCs.xlsx','Cured')
indn=["F1T1","F2T1","F3T1","F4T1","F5T1",
     "S1T1","S2T1","S3T1","S4T1","S5T1","S6T1","S7T1",
     "M1T1","M2T1","M3T1","M4T1","M5T1","M6T1","M7T1","M8T1","M9T1","M10T1"]
indp=["S1T2","S2T2","S3T2","S4T2","S5T2","S6T2","S7T2",
     "M1T2","M2T2","M3T2","M4T2","M5T2","M6T2","M7T2","M8T2","M9T2","M10T2"]
ind = indp + indn
lis=[0,1,2]
X_test = df.loc[lis, ind].T.values
y_test_scores = model.predict_proba(X_test)
print(y_test_scores)
