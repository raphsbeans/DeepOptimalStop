# -*- coding: utf-8 -*-
"""
This script will create simulations of our variables
"""

import numpy as np
import matplotlib.pyplot as plt

def multiBrownian(n, dim, T):
    '''
    A multidimensional independent Brownian motion 
    '''
    dt = T/n
    Z = np.sqrt(dt)*np.random.randn(dim, n)
    return np.cumsum(Z, axis = 1)
    
def reshape(array, dim):
    if type(array) == float or type(array) == int:
        array = array*np.ones(dim).reshape((dim,1))
    else:
        array = np.array(array).reshape((dim,1))
        
    return array
    
def geometricBM (n, dim, T, S0, r, mi, sigma, pho):
    '''
    This function will simulate a geometric BM
    '''
    mi = reshape(mi, dim)
    sigma = reshape(sigma, dim)
    S0 = reshape(S0, dim)
        
    Tp = np.arange(0, T, T/n)*np.ones((dim,n))
    
    mt = r - mi - sigma**2/2
    
    ms = sigma*sigma.transpose()
    Pho = pho*np.ones((dim, dim)) - (pho-1)*np.diag(np.ones(dim))
    ms = ms*Pho
    
    ms = np.linalg.cholesky(ms)
    
    Z = multiBrownian(n, dim, T)
    
    return S0 * np.exp(mt * Tp + np.matmul(ms, Z))
    
g = geometricBM (1024, 32, 1, 5, 0.4, 1, 0.1, 3/5)

