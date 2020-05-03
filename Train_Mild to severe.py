import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

df = pd.read_excel('Data/OBCs.xlsx','Mild to severe')
indp = ["F1T1", "F2T1", "F3T1", "F4T1", "F5T1",
        "S1T1", "S2T1", "S3T1", "S4T1", "S5T1", "S6T1", "S7T1"]
indn = ["M1T1", "M2T1", "M3T1", "M4T1", "M5T1", "M6T1", "M7T1", "M8T1", "M9T1", "M10T1"]
ind = indp + indn
lis=[0,1,2]
X = df.loc[lis, ind].T.values
y = np.array([1 if x in indp else 0 for x in ind])

def Train_fold(i,r, X_train, y_train, X_test, y_test):
    Clist = [1]
    clf = LogisticRegressionCV(Cs=Clist, penalty='l2', fit_intercept=False, cv=5, solver='lbfgs', n_jobs=4,
                               refit=True, class_weight='balanced', multi_class='ovr')
    clf.fit(X_train, y_train)
    result = []
    coef = clf.coef_.ravel()
    result.append(clf.intercept_.tolist() + coef.tolist())
    y_test_scores = clf.predict_proba(X_test)[:, 1]
    AUC = roc_auc_score(y_test, y_test_scores)
    joblib.dump(clf, filename=str(n) + f'/Mild to severe.model')
    return y_test, y_test_scores, result, AUC

tests = np.array([])
y_tests = np.array([])
y_test_scores = np.array([[]])
n='Model'
for r in range(20):
    Skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
    for i, (train, test) in enumerate(Skf.split(X, y)):
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        y_test, y_test_score, result, AUC = Train_fold(i,r, X_train, y_train, X_test, y_test)
        print(AUC)

















