# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:11:49 2020

@author: Diego Ferruzzo
"""
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import funcoes as myfun

# função de teste 1
def f(x, y):
    return x * sqrt(y)

# função de teste 2
def lorenz(t,X):
    """
    Parameters
    ----------
    t : float
        this function is autonomous, so do not depend on t.
    X : matrix (3,1)
        the state.

    Returns
    -------
    TYPE
        the Lorenz funcion f(x).

    """
    sigma = 10
    beta = 8/3
    rho = 28
    x = X.A1[0]
    y = X.A1[1]
    z = X.A1[2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.matrix([[dx],[dy],[dz]])
#
ti = 0.1
tf = 30
dt = 0.01
x0 = np.matrix([[1],[2],[3]])
t, X = myfun.rk4(lorenz, ti, tf, dt, x0)    
#
x = X[0,:].A1
y = X[1,:].A1
z = X[2,:].A1
plt.plot(x,y)