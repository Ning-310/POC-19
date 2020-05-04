import numpy as np
import pandas as pd
import scipy.stats as st
import math

df = pd.read_excel('Data/DEPs.xlsx','530')
dict={"FT1":["F1T1", "F2T1", "F3T1", "F4T1", "F5T1"],
      "FT2":["F1T2", "F2T2", "F3T2", "F4T2", "F5T2"],
      "FT3":["F1T3", "F2T3", "F3T3", "F4T3", "F5T3"],
      "FT4":["F1T4", "F2T4", "F3T4", "F4T4", "F5T4"],
      "ST1":["S1T1","S2T1","S3T1","S4T1","S5T1","S6T1","S7T1"],
      "ST2":["S1T2","S2T2","S3T2","S4T2","S5T2","S6T2","S7T2"],
      "MT1":["M1T1","M2T1","M3T1","M4T1","M5T1","M6T1","M7T1","M8T1","M9T1","M10T1"],
      "MT2":["M1T2","M2T2","M3T2","M4T2","M5T2","M6T2","M7T2","M8T2","M9T2","M10T2"],
      "H":["H1","H2","H3","H4","H5","H6","H7","H8"],
      'FT1,FT2,FT3,FT4':["F1T1", "F2T1", "F3T1", "F4T1", "F5T1","F1T2", "F2T2", "F3T2", "F4T2", "F5T2", "F1T3", "F2T3", "F3T3", "F4T3", "F5T3","F1T4", "F2T4", "F3T4", "F4T4", "F5T4"]}
tit=['FT1/H','FT2/H','FT3/H','FT4/H','FT1,FT2,FT3,FT4/H']
tim=["FT1","FT2","FT3","FT4","H",'FT1,FT2,FT3,FT4']
w=open('T-test.txt','w')
for xi in range(df[["H1"]].T.values.shape[1]):
    X = '\t'.join(np.array(df.loc[xi, :].T.values,dtype=str).tolist())
    indm = []
    for ttsp1 in tim:
        Xm = df[dict[ttsp1]].T.values
        indm.append(str(np.mean(Xm[:, xi])))
    rat=[]
    lg=[]
    pv = []
    for tt in tit:
        ttsp = tt.split('/')
        indp = dict[ttsp[0]]
        indn = dict[ttsp[1]]
        X1 = df[indp].T.values
        X2 = df[indn].T.values
        rat.append(str(np.mean(X1[:,xi])/np.mean(X2[:,xi])))
        lg.append(str(math.log(np.mean(X1[:, xi]) / np.mean(X2[:, xi]),2)))
        p = st.ttest_ind(X1[:, xi], X2[:, xi], equal_var=False)[1]
        pv.append(str(p))
    tst = X + '\t' + '\t' + '\t'.join(indm) + '\t' + '\t'+ '\t'.join(rat)+ '\t' + '\t'+ '\t'.join(lg)+ '\t' + '\t'+ '\t'.join(pv)
    w.write(tst+'\n')

