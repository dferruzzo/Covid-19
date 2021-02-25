# -*- coding: utf-8 -*-
"""
Created on Sat Fev 09, 2021

@author: Diego Ferruzzo
"""
# %%
# Carregando librarias
# from sympy import transpose as tp
import numpy as np
import pandas as pd 
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from myfunctions import rk4 
#%% Parâmetros para fitting
# Global variables
# Nesse modelo estou utilizando theta como função de t,
mu = 2e-5 # daily birth and death rate
#
def rhs(t,x,p):
    # modelo 2 -  simplificado
    # ds/dt = mu + gamma - alpha(1-theta)*s*i - (mu + gamma)*s - gamma*i - gamma*sick
    # di/dt = alpha*(1-theta)*s*i - (beta1 + beta2 + mu)*i
    # dsick/dt = beta2*i - (beta3 + mu)* sick
    # dtheta/dt = a
    #
    # this is the right-hand side function, implemented for numerical integration
    s = x[0]
    i = x[1]
    sick = x[2]
    theta = x[3]
    #
    mu = p[0]
    gamma = p[1]
    alpha = p[2]
    beta1 = p[3]
    beta2 = p[4]
    beta3 = p[5]
    #
    a = -2.556e-04
    #

    return np.array([mu+gamma-alpha*(1-theta)*s*i-(mu+gamma)*s-gamma*i-gamma*sick,
                     alpha*(1-theta)*s*i-(beta1+beta2+mu)*i,
                     beta2*i-(beta3+mu)*sick,
                     a])
#
"""
# testando a função rhs
b = 4.913e-01
x0 = np.array([0.99, 0.01, 0, b])
t0 = 0
tf = 100
h = 1
gamma = 0
alpha = 0.3
beta1 = 1/45
beta2 = 1/21
beta3 = 1/45
p = np.array([mu, gamma, alpha, beta1, beta2, beta3])
t, sol = rk4(lambda t, x: rhs(t, x, p), x0, t0, tf, h)

plt.figure()
plt.plot(t, sol[:, 3])
plt.grid(axis='both')
plt.title('Índice de isolamento para São Paulo - Aproximação Linear')
plt.xlabel('days')
plt.show()
"""
#
def f(x,gamma,alpha,beta1,beta2,beta3,s0):
        # in this version I adjust the initial condition so s0 + i0 + r0 = 1
        i0 = 1-s0
        r0 = 0
        theta0 = 4.913e-01
        x0 = np.array([s0,i0,r0, theta0]) # initial condition
        t0 = 0 # simulação comeza no dia 0
        tf = N-1 # até N-1 dias
        h = 1 # o paso de integração é um dia
        p = np.zeros(7)
        p[0] = mu
        p[1] = gamma
        p[2] = alpha
        p[3] = beta1
        p[4] = beta2
        p[5] = beta3
        t, sol = rk4(lambda t, x: rhs(t, x, p), x0, t0, tf, h)
        return sol[:, 2]
#
# Optimization parameters for all cases
# standard deviation of error is the data
sigma = None
absolute_sigma = False
# check if there is any NaN or InF in data
check_finite = True
# bounds
gamma_min = 0
alpha_min = 0.2
beta1_min = 1/45
beta2_min = 1/21
beta3_min = 1/45
s0_min = 0
#i0_min = 0
params_min = [gamma_min, alpha_min, beta1_min, beta2_min, beta3_min, s0_min]
gamma_max = 1
alpha_max = 2
beta1_max = 1/7
beta2_max = 1/5
beta3_max = 1/15
s0_max = 0.9999
#i0_max = 1
params_max = [gamma_max, alpha_max, beta1_max, beta2_max, beta3_max, s0_max]
bounds = (params_min, params_max)
# method
# ‘dogbox’ : dogleg algorithm with rectangular trust regions,
# 'trf' : Trust Region Reflective algorithm
method = 'trf'
# Jacobian
jac = None
#%% São Paulo
print('\nLoading the data - São Paulo')
print('----------------------------')
SaoPaulo_data = pd.read_csv("data/SaoPaulo_dados_covid.csv")
SaoPaulo_data = SaoPaulo_data.to_numpy()
SaoPaulo_tot_pop = int(np.mean(SaoPaulo_data[:, 4]))
SaoPaulo_cases = SaoPaulo_data[:, 2]
SaoPaulo_days = pd.to_datetime(SaoPaulo_data[:, 1], yearfirst=True)
SaoPaulo_time = np.arange(0, SaoPaulo_cases.size, 1)
N = SaoPaulo_time.size
print('total population:', SaoPaulo_tot_pop)
#
# initial guesses for parameters for optimization
gamma_0 = 0.01
alpha_0 = 0.2
beta1_0 = 0.06 #0.0714
beta2_0 = 0.15 #1/7
beta3_0 = 0.06 #0.05
s0_0 = 0.9
#i0_0 = 1-s0_0#0.001
p0 = np.array([gamma_0, alpha_0, beta1_0, beta2_0, beta3_0, s0_0])
# The optimization itself
print('\nRunning the optimization ...')
popt, pvoc = optimization.curve_fit(f, SaoPaulo_time,
                                    SaoPaulo_cases/SaoPaulo_tot_pop, p0, sigma,
                                    absolute_sigma, check_finite, bounds,
                                    method, jac)
print('optimização finalizada.')
#
# Testing the optimization
perr = np.sqrt(np.diag(pvoc))
print('Standard deviation errors on the parameters = ', perr)
print('\nParameters:')
print('-----------')
print('mu =', mu)
print('Fitted parameters:')
gamma_opt = popt[0]
print('gamma =', gamma_opt)
alpha_opt = popt[1]
print('alpha =', alpha_opt)
beta1_opt = popt[2]
print('beta1 =', beta1_opt)
beta2_opt = popt[3]
print('beta2 =', beta2_opt)
beta3_opt = popt[4]
print('beta3 =', beta3_opt)
s0_opt = popt[5]
print('s0 =', s0_opt)
i0_opt = 1-s0_opt
print('i0 =', i0_opt)
# initial condition
theta0 = 4.913e-01
x0 = np.array([s0_opt, i0_opt, 0, theta0])
# parameters
t0 = 0
tf = N-1
h = 1
#    
p=np.zeros(7)
p[0] = mu
p[1] = gamma_opt
p[2] = alpha_opt
p[3] = beta1_opt
p[4] = beta2_opt
p[5] = beta3_opt
#
t, sol =  rk4(lambda t, x: rhs(t, x, p), x0, t0, tf, h)
#
"""
# Saving parameters data to file
data = {'Parameter': ['mu', 'gamma', 'alpha', 'beta1', 'beta2', 'beta3', 's0', 'i0', 'sick0'],
        'Value': np.concatenate((p, x0))}
#print('parameters =',data)        
df_saopaulo = pd.DataFrame(data) 
print('Saving parameters in data/saopaulo_params.csv...')
df_saopaulo.to_csv("data/saopaulo_params.csv")
print('done.')
print('São Paulo parameters - Latex')
print(df_saopaulo.to_latex(index=False))
"""
# Plotting figure
print('Printing figure to figures/sick_saopaulo.png')
plt.figure()
plt.scatter(SaoPaulo_days, SaoPaulo_cases, alpha=0.5)
plt.plot(SaoPaulo_days, sol[:, 2]*SaoPaulo_tot_pop, c='r')
plt.grid(axis='both')
plt.legend(['Fitted curve', 'Real data'])
plt.title('São Paulo - Sick population')
plt.xlabel('days')
#plt.savefig('figures/sick_saopaulo.png')
print('done.')
plt.show()
#plt.close()
