import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from keras.utils.np_utils import to_categorical
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

df = pd.read_excel('MS.xlsx','Sheet1')
X_df = df.iloc[:, 3:]
ind1 = ["H1","H2","H3","H4","H5","H6","H7","H8"]
ind2 = ["M1T1", "M2T1", "M3T1", "M4T1", "M5T1", "M6T1", "M7T1", "M8T1", "M9T1", "M10T1"]
ind3 = ["S1T1", "S2T1", "S3T1", "S4T1", "S5T1", "S6T1", "S7T1" ]
ind4 = ["F1T1", "F2T1", "F3T1", "F4T1", "F5T1", "F1T2", "F2T2", "F3T2", "F4T2", "F5T2" , "F1T3", "F2T3", "F3T3", "F4T3", "F5T3", "F1T4", "F2T4", "F3T4", "F4T4", "F5T4"]
ind = ind1 + ind2+ ind3+ ind4
lis=[195,399,790,353]
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
p1 = df[['Description']].T.values
p1 = list(p1.flatten())
p2 = df[['Gene']].T.values
p2 = list(p2.flatten())


##################以上部分可以公用
def Train_fold(i, X_train, y_train, X_test, y_test):
    Clist = [25]
    clf = LogisticRegressionCV(Cs=Clist, penalty='l2', fit_intercept=False, cv=5, solver='lbfgs', n_jobs=4,
                               refit=True, class_weight='balanced', multi_class='ovr')
    clf.fit(X_train, y_train)
    result = []
    coef = clf.coef_.ravel()
    result.append(clf.intercept_.tolist() + coef.tolist())
    AUC=[]
    y_test_scores = clf.predict_proba(X_test)
    y_test_onehot = to_categorical(y_test)
    for j in range(4):
        AUC.append(roc_auc_score(y_test_onehot[:,j],y_test_scores[:,j]))
        print(f'Fold {i} AUC Type {j}',AUC)
    joblib.dump(clf, filename=str(n) + f'/PLR_3c_final_fold_{i}.model')
    print(f'PLR re_calc AUC in console')

    return y_test, y_test_scores, result, AUC

n = 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)  # random_state=17

Skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=6)  # random_state=7
tests = np.array([])
y_tests = np.array([])
y_test_scores = np.array([[]])  # .reshape()
n = 'result'
gene = []
gene1 = {}
for i, (train, test) in enumerate(Skf.split(X, y)):
    tests = np.append(tests, test)
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    y_test, y_test_score, result, AUC = Train_fold(i, X_train, y_train, X_test, y_test)
    y_tests = np.append(y_tests, y_test, axis=0)

    if i==0:y_test_scores=y_test_score
    else:y_test_scores = np.append(y_test_scores, y_test_score, axis=0)
    w = open(str(n) + '/' + str(i) + '-'.join(ind3) + '.txt', 'w')

    for ri, r in enumerate(result[0]):
        if r > 0 or r < 0:
            if  str(ri) + '\t' + str(p2[ri - 1]) not in gene:
                gene.append(str(ri) + '\t' + str(p2[ri - 1]))
                gene1[str(ri)] = 1
            else:
                gene1[str(ri)] = gene1[str(ri)] + 1
            w.write(p1[ri - 1] + '\t' + str(r) + '\t' + str(ri) + '\t' + str(AUC) + '\n')
    for g in gene:
        w.write(g + '\t' + str(gene1[g.split('\t')[0]]) + '\n')


    for ii,xi in enumerate(X_train):
        w.write(str(X_train[ii]) + '\t' + str(y_train[ii]) + '\n')
    for ii, xi in enumerate(X_test):
        w.write(str(X_test[ii]) + '\t' + str(y_test[ii])+ '\t' + str(y_test_score[ii]) +  '\n')
    w.write( str(result) + '\n')

font = {'family': 'arial',
        'weight': 'bold',
        'size': 22,
        }
import matplotlib.pylab as pylab

params = {
    'axes.labelsize': '22',
    'xtick.labelsize': '22',
    'ytick.labelsize': '22',
    'lines.linewidth': '4',

}
pylab.rcParams.update(params)
pylab.rcParams['font.family'] = 'sans-serif'
pylab.rcParams['font.sans-serif'] = ['Arial']
pylab.rcParams['font.weight'] = 'bold'
plt.figure(figsize=(6, 6), dpi=300)

y_test_onehots = to_categorical(y_tests)
AUC=[]
fpr1=[]
tpr1=[]
thresholds1=[]
for j in range(4):
    AUC.append(roc_auc_score(y_test_onehots[:,j],y_test_scores[:,j]))

    fpr1.append(roc_curve(y_test_onehots[:,j],y_test_scores[:,j])[0])
    tpr1.append(roc_curve(y_test_onehots[:,j],y_test_scores[:,j])[1])
    thresholds1.append(roc_curve(y_test_onehots[:,j],y_test_scores[:,j])[2])

plt.plot(fpr1[3], tpr1[3], linewidth='3', color='tomato', label='Fatal (AUC = {:.3f})'.format(AUC[3]))
plt.plot(fpr1[2], tpr1[2], linewidth='3', color='goldenrod', label='Severe (AUC = {:.3f})'.format(AUC[2]))
plt.plot(fpr1[1], tpr1[1], linewidth='3', color='steelblue', label='Mild (AUC = {:.3f})'.format(AUC[1]))
plt.plot(fpr1[0], tpr1[0], linewidth='3', color='seagreen', label='Healthy (AUC = {:.3f})'.format(AUC[0]))

plt.plot([0, 1], [0, 1], linewidth='1', color='grey', linestyle="--")
plt.yticks(np.linspace(0, 1, 6))
plt.xticks(np.linspace(0, 1, 6))
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.axis([-0.03, 1.03, -0.03, 1.03])
plt.legend(prop={'size': 20}, loc=4, frameon=False)
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
plt.xlabel('1–Specificity', font)
plt.ylabel('Sensitivity', font)

plt.savefig(str(n) + '/' + 'roc'+str(lis)+'.pdf')
plt.savefig(str(n) + '/' + 'roc'+str(lis)+'.jpg')
plt.show()





















