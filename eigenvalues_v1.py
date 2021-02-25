#%% -*- coding: utf-8 -*-
"""
Created on Fri Aug 25/08, 2020

@author: Diego Ferruzzo
-----------------------------------

Script for testing the equilibria for the model.
di/dt    = alpha*(1-theta)*(1-i-sick-r)*i - (beta1 + beta2)*i - mu*i
dsick/dt = beta2*i-(beta2 + mu)*sick
dr/dt    = beta1*i + beta3*sick - (gamma + mu)*r 
"""
import pandas as pd 
import sympy as sym
# vector state
s,i,sick,r,v = sym.symbols('s,i,sick,r,v', positive=True)
x = sym.Matrix([i,sick,r])
# parameters
mu, gamma, alpha, theta, omega, beta1, beta2, beta3, zeta=\
    sym.symbols('mu,gamma,alpha,theta,omega,beta1,beta2,beta3,zeta',\
                positive=True) 
# The right-hand side
f = sym.Matrix([[alpha*(1-theta)*(1-i-sick-r)*i - (beta1 + beta2)*i - mu*i],\
                [beta2*i-(beta2 + mu)*sick],\
                [beta1*i + beta3*sick - (gamma + mu)*r]])
print('')
print('The right-hand side')
print('-------------------')
print('f =',f) 
#%% R0
R0 = alpha*(1-theta)/(beta1+beta2+mu)
print('R_0 = \n',R0)
# %% Computing eigenvalues for the free-disease equilibrium
A = f.jacobian(x)
print('\nThe Jacobian')
print('---------------------------------------------')
print('A=',A)
print('A=',sym.latex(A))
# %%
print('\nThe endemic equilibrium')
print('-----------------------')
# computing sick = sick(i) from f[1]
sick_i = sym.solve(f[1],sick)[0]
# computing r = r(i) from f[2]
r_i = sym.simplify(sym.solve(f[2].subs(sick,sick_i),r)[0])
# substituting sick and r into f[0] and solving fort i
i_en = sym.simplify(sym.solve(f[0].subs(sick,sick_i).subs(r,r_i),i)[0])
# computing sick
sick_en = sym.simplify(sick_i.subs(i,i_en))
# computing r
r_en = sym.simplify(r_i.subs(i,i_en))
# print endemic equilibrium
print('i^* = \n',i_en)
print('sick^* = \n',sick_en)
print('r^* = \n',r_en)
#%% The lyapunov candidate function
L = 