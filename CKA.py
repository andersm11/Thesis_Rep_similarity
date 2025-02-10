#!/usr/bin/env python

### Mot used as a python file at the moment
import numpy as np
import HSIC
import math
def CKA(time, space):
    weights_time_2d = np.squeeze(time)
    V = weights_time_2d[:,0].reshape(-1,1)
    subset_space = space[:,0,:,:]
    space_weights_2d = np.squeeze(subset_space)
    W = space_weights_2d[:,0].reshape(-1,1)
    K = np.matmul(V,V.T)
    L = np.matmul(W,W.T) 
    HSIC_K_K = HSIC.hsic_gam(K,K)
    HSIC_K_L = HSIC.hsic_gam(K,L)
    HSIC_L_L = HSIC.hsic_gam(L,L)

    CKA = HSIC_K_L[0]/(math.sqrt(HSIC_K_K[0]*HSIC_L_L[0])) ### Tuple, took [0] since (test-statistic, test-threshold for alpha)
    return CKA


