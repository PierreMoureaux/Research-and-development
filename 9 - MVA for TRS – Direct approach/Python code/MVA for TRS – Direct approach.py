# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 11:52:30 2023

@author: moureaux pierre
"""

import numpy as np

"TRS datas"
T = 1
sTRS = 0.1

"Model datas"
Nouter = 100
Ninner = 100
lambB = 0.1
lambC = 0.1
R = 0.4
sIM = 0.02
r = 0.1
sigma = 0.1
deltaT = 0.05
MPOR = 10
S0 = 100
K = 100
lT = int(T/deltaT)

def MVATRS():
    timeSum = 0
    #Loop for TRS life cycle
    for k in range(lT):
        dfMPOR = np.exp(-(lT-k-MPOR)*r)
        dfk = np.exp(-(lT-k)*r)
        quant = np.zeros(Nouter)
        #Outer Monte-Carlo loop
        for j in range(Nouter):
            innerSum = np.zeros(Ninner)
            #Inner Monte-Carlo loop
            for i in range(Ninner):
                phiS=np.random.normal(0, 1, lT)
                S = np.zeros(lT)
                S[0] = 100
                #Specific loop for risk factor generation(here only equity)
                for z in range(lT-1):
                    S[z+1] = S[z]*(r*deltaT+sigma*np.sqrt(deltaT)*phiS[z]+1)
                if ((k+MPOR)>lT):
                    innerSum[i] = dfMPOR*(S[lT-1]-K - sTRS*T)-dfk*(S[k]-K - sTRS*T)
                else:
                    innerSum[i] = dfMPOR*(S[k+MPOR-1]-K - sTRS*T)-dfk*(S[k]-K - sTRS*T)
            quant[j] = np.quantile(innerSum,0.99)
        outerSum = np.mean(quant)
        timeSum += np.exp(-k*(lambB + lambC+r))*outerSum
    return ((1-R)*lambB - sIM)*timeSum

print("Margin valuation adjustment for bullet fixed rate TRS is",MVATRS())