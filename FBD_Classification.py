import os
import numpy as np
import pandas as pd
import joblib
from keras.utils.np_utils import to_categorical
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random

df = pd.read_excel('Data/DEPs.xlsx','DEPs')
ind1 = ["H1","H2","H3","H4","H5","H6","H7","H8"]
ind2 = ["M1T1", "M2T1", "M3T1", "M4T1", "M5T1", "M6T1", "M7T1", "M8T1", "M9T1", "M10T1"]
ind3 = ["S1T1", "S2T1", "S3T1", "S4T1", "S5T1", "S6T1", "S7T1" ]
ind4 = ["F1T1", "F2T1", "F3T1", "F4T1", "F5T1", "F1T2", "F2T2", "F3T2", "F4T2", "F5T2" , "F1T3", "F2T3", "F3T3", "F4T3", "F5T3", "F1T4", "F2T4", "F3T4", "F4T4", "F5T4"]
ind = ind1 + ind2+ ind3+ ind4

#CBS
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

#FBD
for lis in rd:
    X = df.loc[lis, ind].T.values
    y=[]
    for x in ind:
        if  x in ind1:
            y.append(0)
        if  x in ind2:
            y.append(1)
        if  x in ind3:
            y.append(2)
        if  x in ind4:
            y.append(3)
    y = np.array(y)

    def Train_fold(X_train, y_train, X_test, y_test,l):
        Clist = [25]
        clf = LogisticRegressionCV(Cs=Clist, penalty=l, fit_intercept=False, cv=5, solver='lbfgs', n_jobs=4,
                                   refit=True, class_weight='balanced', multi_class='ovr')
        clf.fit(X_train, y_train)
        result = []
        coef = clf.coef_.ravel()
        result.append(coef.tolist())
        AUC=[]
        y_test_scores = clf.predict_proba(X_test)
        y_test_onehot = to_categorical(y_test)
        for j in range(4):
            AUC.append(roc_auc_score(y_test_onehot[:,j],y_test_scores[:,j]))
        joblib.dump(clf, filename=str(n) + f'/Classification.model')
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
