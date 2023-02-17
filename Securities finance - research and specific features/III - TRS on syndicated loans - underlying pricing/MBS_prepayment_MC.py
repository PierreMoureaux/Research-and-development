# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:39:10 2022

@author: pmoureaux
"""

import pandas as pd
import statistics as stat
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from numpy import linalg as LA
from sklearn.decomposition import PCA
import math
from scipy import stats
from scipy.stats import norm
from math import floor

Nominal = 1000
l = 10
spotRate = 0.05
volRate = 0.1
Schedule = np.arange(l)
fwdRates = np.zeros(l)
lamb = np.zeros(l)
EL = np.zeros(l)
ELoP = np.zeros(l)
fwdRates[0] = spotRate
amoPercent = 0.1
deltaT = 1
rho = 0.035
kappa = 0.27
theta = 0.5
lambda0 = 0.1
lamb[0] = lambda0
mu = 0.1
r_star = 0.2

for k in range(1,l):
    fwdRates[k] = fwdRates[k-1] + volRate*fwdRates[k-1]*np.sqrt(deltaT)*np.random.normal(0,1)
    lamb[k] = lamb[k-1] + kappa*(theta-lamb[k-1])*deltaT + mu*np.sqrt(lamb[k-1])*np.sqrt(deltaT)*np.random.normal(0,1)

def P(t1,t2):
    f = 0
    for i in range(t1,t2):
        f += fwdRates[i]
    return np.exp(-f)

def PP(t1,t2,r, r_star):
    if r > r_star:
        return 1 - np.exp(-rho*(t2-t1))
    else:
        e = 0
        for i in range(t1,t2):
            e += lamb[i]
        return 1 - np.exp(-rho*(t2-t1))*np.mean(np.exp(-e))

def impliedCoupon(N,p,t):
    num, den = 0,0
    c = N*p
    N_ = N
    for j in range(l):
        num += c*P(t,j)
        den += N_*P(t,j)
        N_ = N_*(1-p)
    return (N - num)/den

def ExpectedLoss(N,p):
    c = impliedCoupon(N,p,0)
    N_ = N*(1-amoPercent)
    for k in range(1,l):
        sum = 0
        for j in range(1,l):
            c_star = impliedCoupon(N*k*(1-amoPercent),p,k)
            sum += P(k,j)*N_*(c - c_star)
            N_ = N_*(1-amoPercent)
        EL[k] = np.maximum(sum,0)
    return EL

def ExpectedLossPre(N,p):
    for k in range(1,l):
        ELoP[k] = ExpectedLoss(N,p)[k]*PP(k-1,k,fwdRates[k], r_star)
    return ELoP

plt.plot(ExpectedLoss(Nominal,amoPercent))
plt.plot(ExpectedLossPre(Nominal,amoPercent))
