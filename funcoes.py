# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:12:39 2020

Coleção de funções que utilizo comumente para pesquisa.

@author: Diego Ferruzzo
"""
import numpy as np

def rk4(f, ti, tf, dt, x0):
    """
    Versão vetorial do algoritmo Runge-Kutta 4, pode ser utilizada para
    funçoes escalares desde que sejam utilizadas as estruturas numpy.matrix
    para definir os objetos.
    Parameters
    ----------
    f : f(t,x)
        função a integrar.
    ti : float
        tempo inicial.
    tf : float
        tempo final.
    dt : float
        passo de integração. |tf-ti| >= dt.
    x0 : np.matrix shape (n,1)
        condições iniciais.

    Returns
    -------
    t : array dim steps + 1
        vetor tempo [ti,...,tf]
    x : numpy.matrix(n, steps + 1)
        solução do problema dx/dt=f(t,x).

    """
    n = x0.size
    steps = int((tf - ti) / dt)
    t = [0] * (steps + 1)
    x = np.zeros((n,steps + 1))
    t[0] = vt = ti
    x[:,0] = x0.A1
    vx = x0
    for i in range(1, steps + 1):
        k1 = dt * f(vt, vx)
        k2 = dt * f(vt + 0.5 * dt, vx + 0.5 * k1)
        k3 = dt * f(vt + 0.5 * dt, vx + 0.5 * k2)
        k4 = dt * f(vt + dt, vx + k3)
        t[i] = vt = ti + i * dt
        vx = vx + (k1 + k2 + k2 + k3 + k3 + k4) / 6
        x[:,i] = vx.A1
    return t, np.asmatrix(x)