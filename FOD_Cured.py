import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

df = pd.read_excel('Data/DEPs.xlsx','DEPs')
indn=["F1T1","F2T1","F3T1","F4T1","F5T1",
     "S1T1","S2T1","S3T1","S4T1","S5T1","S6T1","S7T1",
     "M1T1","M2T1","M3T1","M4T1","M5T1","M6T1","M7T1","M8T1","M9T1","M10T1"]
indp=["S1T2","S2T2","S3T2","S4T2","S5T2","S6T2","S7T2",
     "M1T2","M2T2","M3T2","M4T2","M5T2","M6T2","M7T2","M8T2","M9T2","M10T2"]
ind = indp + indn

#COS
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

rd=[]
while rd.__len__()<1000:
    rd0=random_int_list(0, 112, 5)
    if rd0 not in rd:
        rd.append(rd0)

#FOD
for lis in rd:
    X = df.loc[lis, ind].T.values
    y = np.array([1 if x in indp else 0 for x in ind])

    def Train_fold(X_train, y_train, X_test, y_test, l):
        Clist = [1]
        clf = LogisticRegressionCV(Cs=Clist, penalty=l, fit_intercept=False, cv=5, solver='lbfgs', n_jobs=4,
                                   refit=True, class_weight='balanced', multi_class='ovr')
        clf.fit(X_train, y_train)
        result = []
        coef = clf.coef_.ravel()
        result.append(clf.intercept_.tolist() + coef.tolist())
        y_test_scores = clf.predict_proba(X_test)[:, 1]
        AUC = roc_auc_score(y_test, y_test_scores)
        joblib.dump(clf, filename=str(n) + f'/Cured.model')
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
            y_test, y_test_score, result, AUC = Train_fold(X_train, y_train, X_test, y_test,'l1')
        lis1=[]
        for w in result:
            for wi, w0 in enumerate(w):
                if w0 != 0:
                    lis1.append(lis[wi])
            X1 = df.loc[lis1, ind].T.values
            Skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
            for i, (train, test) in enumerate(Skf.split(X1, y)):
                X_train = X1[train]
                y_train = y[train]
                X_test = X1[test]
                y_test = y[test]
                y_test, y_test_score, result, AUC = Train_fold(X_train, y_train, X_test, y_test,'l2')
                print(lis1, AUC)















