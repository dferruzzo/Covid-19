# -*- coding: utf-8 -*-
"""
Created on Fri Ago 05, 2020

@author: Diego Ferruzzo
"""
#%%
#from sympy import transpose as tp
import numpy as np
import pandas as pd 
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from myfunctions import rk4 
#%% São Paulo
print('\nLoading the data - São Paulo')
print('----------------------------')
SaoPaulo_data = pd.read_csv("data/SaoPaulo_dados_covid.csv")
SaoPaulo_data = SaoPaulo_data.to_numpy()
SaoPaulo_tot_pop = int(np.mean(SaoPaulo_data[:,4]))
SaoPaulo_cases = SaoPaulo_data[:,2]
SaoPaulo_days = pd.to_datetime(SaoPaulo_data[:,1],yearfirst=True)
SaoPaulo_time = np.arange(0, SaoPaulo_cases.size, 1)
N = SaoPaulo_time.size
print('total population:',SaoPaulo_tot_pop)
#%%
#Creating f for parameters fitting
# Global variables
theta = 0.48 # updated 05/08/2020 social distancing mean for Sao Paulo county
mu = 2e-5 # daily birth and death rate
#
def rhs(t,x,p):
        # this is the right-hand side funtion, implemented for numerticalk integration
    s = x[0]
    i = x[1]
    sick = x[2]
    mu = p[0]
    gamma = p[1]
    alpha = p[2]
    theta = p[3]
    beta1 = p[4]
    beta2 = p[5]
    beta3 = p[6]
    return np.array([mu+gamma-alpha*(1-theta)*s*i\
                     -(mu+gamma)*s-gamma*i-gamma*sick,\
                     alpha*(1-theta)*s*i-(beta1+beta2+mu)*i,\
                     beta2*i-(beta3+mu)*sick])      
#
"""
def f(x,gamma,alpha,beta1,beta2,beta3,s0,i0):
    x0 = np.array([s0,i0,0]) # initial condition
    t0 = 0
    tf = N-1
    h = 1
    p=np.zeros(7)
    p[0] = mu
    p[1] = gamma
    p[2] = alpha
    p[3] = theta
    p[4] = beta1
    p[5] = beta2
    p[6] = beta3
    t,sol = rk4(lambda t,x: rhs(t,x,p),x0,t0,tf,h)
    return sol[:,2]
"""
#   
def f(x,gamma,alpha,beta1,beta2,beta3,s0):
        # in this version I adjust the initial condition so s0 + i0 + r0 = 1
        i0 = 1-s0
        r0 = 0
        x0 = np.array([s0,i0,r0]) # initial condition
        t0 = 0
        tf = N-1
        h = 1
        p=np.zeros(7)
        p[0] = mu
        p[1] = gamma
        p[2] = alpha
        p[3] = theta
        p[4] = beta1
        p[5] = beta2
        p[6] = beta3
        t,sol = rk4(lambda t,x: rhs(t,x,p),x0,t0,tf,h)
        return sol[:,2]
#%%
# initial guesses for parameters for optimization
gamma_0 = 0.01
alpha_0 = 0.6
beta1_0 = 1/14
beta2_0 = 1/7
beta3_0 = 1/20
s0_0 = 0.9
#i0_0 = 1-s0_0#0.001
p0 = np.array([gamma_0,alpha_0,beta1_0,beta2_0,beta3_0,s0_0])
# standard deviation of error is the data
sigma = None
absolute_sigma = False
# check if there is any NaN or InF in data
check_finite = True
# bounds
gamma_min = 0
alpha_min = 0.3
beta1_min = 1/45
beta2_min = 1/21
beta3_min = 1/45
s0_min = 0
#i0_min = 0
params_min = [gamma_min,alpha_min,beta1_min,beta2_min,beta3_min,s0_min]
gamma_max = 1
alpha_max = 2
beta1_max = 1/7
beta2_max = 1/5
beta3_max = 1/15
s0_max = 0.9999
#i0_max = 1
params_max = [gamma_max,alpha_max,beta1_max,beta2_max,beta3_max,s0_max]
bounds = (params_min,params_max)
# method
# ‘dogbox’ : dogleg algorithm with rectangular trust regions,
# 'trf' : Trust Region Reflective algorithm
method = 'trf'
# Jacobian
jac = None
# The optimization itself
print('\nRunning the optimization ...')
popt, pvoc = optimization.curve_fit(f,SaoPaulo_time,\
                                    SaoPaulo_cases/SaoPaulo_tot_pop,p0,sigma,\
                                    absolute_sigma,check_finite,bounds,\
                                    method,jac)
print('optimização finalizada.')
#%%
# Testing the optimization
perr = np.sqrt(np.diag(pvoc))
print('Standard deviation errors on the parameters = ',perr)
print('\nParameters:')
print('-----------')
print('mu =',mu)
print('theta =',theta)
print('Fitted parameters:')
gamma_opt = popt[0]
print('gamma =',gamma_opt)
alpha_opt = popt[1]
print('alpha =',alpha_opt)
beta1_opt = popt[2]
print('beta1 =',beta1_opt)
beta2_opt = popt[3]
print('beta2 =',beta2_opt)
beta3_opt = popt[4]
print('beta3 =',beta3_opt)
s0_opt = popt[5]
print('s0 =',s0_opt)
i0_opt = 1-s0_opt
print('i0 =',i0_opt)
# initial condition
x0 = np.array([s0_opt,i0_opt,0]) 
# parameters
t0 = 0
tf = N-1
h = 1
#    
p=np.zeros(7)
p[0] = mu
p[1] = gamma_opt
p[2] = alpha_opt
p[3] = theta
p[4] = beta1_opt
p[5] = beta2_opt
p[6] = beta3_opt
#
t,sol =  rk4(lambda t,x: rhs(t,x,p),x0,t0,tf,h)
#%%
# Saving parameters data to file
data = {'Parameter':['mu','gamma','alpha','theta','beta1','beta2','beta3','s0','i0','sick0'],\
        'Value':np.concatenate((p,x0))} 
#print('parameters =',data)        
df_saopaulo = pd.DataFrame(data) 
print('Saving parameters in data/saopaulo_params.csv...')
df_saopaulo.to_csv("data/saopaulo_params.csv")
print('done.')
print('São Paulo parameters - Latex')
print(df_saopaulo.to_latex(index=False))
# Plotting figure
print('Printing figure to figures/sick_saopaulo.png')
plt.figure()
plt.scatter(SaoPaulo_days,SaoPaulo_cases,alpha=0.5)
plt.plot(SaoPaulo_days,sol[:,2]*SaoPaulo_tot_pop,c='r')
plt.grid(axis='both')
plt.legend(['Fitted curve','Real data'])
plt.title('São Paulo - Sick population')
plt.xlabel('days')
plt.savefig('figures/sick_saopaulo.png')
print('done.')
#plt.show()
#plt.close()
#%% Santos
print('\nLoading the data - Santos')
print('----------------------------')
Santos_data = pd.read_csv("data/Santos_dados_covid.csv")
Santos_data = Santos_data.to_numpy()
Santos_tot_pop = int(np.mean(Santos_data[:,4]))
Santos_cases = Santos_data[:,2]
Santos_days = pd.to_datetime(Santos_data[:,1],yearfirst=True)
Santos_time = np.arange(0, Santos_cases.size, 1)
N = Santos_time.size
print('total population:',Santos_tot_pop)
# Global variables
theta = 0.47 # social distancing mean for Sao Paulo county
mu = 2e-5 # daily birth and death rate
# initial guesses for parameters for optimization
gamma_0 = 0.01
alpha_0 = 0.4
beta1_0 = 1/14
beta2_0 = 1/5
beta3_0 = 1/14
s0_0 = 0.999
i0_0 = 0.001
p0 = np.array([gamma_0,alpha_0,beta1_0,beta2_0,beta3_0,s0_0,i0_0])
# standard deviation of error is the data
sigma = None
absolute_sigma = False
# check if there is any NaN or InF in data
check_finite = True
# bounds
gamma_min = 0
alpha_min = 0.3
beta1_min = 1/45
beta2_min = 1/21
beta3_min = 1/45
s0_min = 0
i0_min = 0
params_min = [gamma_min,alpha_min,beta1_min,beta2_min,beta3_min,s0_min,i0_min]
gamma_max = 1
alpha_max = 1
beta1_max = 1
beta2_max = 1
beta3_max = 1
s0_max = 0.9999
i0_max = 1
params_max = [gamma_max,alpha_max,beta1_max,beta2_max,beta3_max,s0_max,i0_max]
bounds = (params_min,params_max)
# method
# ‘dogbox’ : dogleg algorithm with rectangular trust regions,
# 'trf' : Trust Region Reflective algorithm
method = 'trf'
# Jacobian
jac = None
# The optimization itself
print('\nRunning the optimization ...')
popt, pvoc = optimization.curve_fit(f,Santos_time,\
                                    Santos_cases/Santos_tot_pop,p0,sigma,\
                                    absolute_sigma,check_finite,bounds,\
                                    method,jac)
# Testing the optimization
perr = np.sqrt(np.diag(pvoc))
print('Standard deviation errors on the parameters = ',perr)
print('\nParameters:')
print('-----------')
print('mu =',mu)
print('theta =',theta)
print('Fitted parameters:')
gamma_opt = popt[0]
print('gamma =',gamma_opt)
alpha_opt = popt[1]
print('alpha =',alpha_opt)
beta1_opt = popt[2]
print('beta1 =',beta1_opt)
beta2_opt = popt[3]
print('beta2 =',beta2_opt)
beta3_opt = popt[4]
print('beta3 =',beta3_opt)
s0_opt = popt[5]
print('s0 =',s0_opt)
i0_opt = popt[6]
print('i0 =',i0_opt)
# initial condition
x0 = np.array([s0_opt,i0_opt,0]) 
# parameters
t0 = 0
tf = N-1
h = 1
#    
#mu = 2e-5 # daily birth and death rate
#theta = 0.4951 # social distancing mean for Sao Paulo county
p=np.zeros(7)
p[0] = mu
p[1] = gamma_opt
p[2] = alpha_opt
p[3] = theta
p[4] = beta1_opt
p[5] = beta2_opt
p[6] = beta3_opt
#
t,sol =  rk4(lambda t,x: rhs(t,x,p),x0,t0,tf,h)
# Saving parameters data to file
data = {'Parameter':['mu','gamma','alpha','theta','beta1','beta2','beta3','s0','i0','sick0'],\
        'Value':np.concatenate((p,x0))} 
#print('parameters =',data)        
df_santos = pd.DataFrame(data) 
print('Saving parameters in data/santos_params.csv...')
df_santos.to_csv("data/santos_params.csv")
print('done.')
print('Santos parameters - Latex')
print(df_santos.to_latex(index=False))
# Plotting figure
print('Printing figure to figures/sick_santos.png')
plt.figure()
plt.scatter(Santos_days,Santos_cases,alpha=0.5)
plt.plot(Santos_days,sol[:,2]*Santos_tot_pop,c='r')
plt.grid(axis='both')
plt.legend(['Fitted curve','Real data'])
plt.title('Santos - Sick population')
plt.xlabel('days')
plt.savefig('figures/sick_santos.png')
print('done.')
#plt.show()
#plt.close()
# %% Campinas
print('\nLoading the data - Campinas')
print('----------------------------')
Campinas_data = pd.read_csv("data/Campinas_dados_covid.csv")
Campinas_data = Campinas_data.to_numpy()
Campinas_tot_pop = int(np.mean(Campinas_data[:,4]))
Campinas_cases = Campinas_data[:,2]
Campinas_days = pd.to_datetime(Campinas_data[:,1],yearfirst=True)
Campinas_time = np.arange(0, Campinas_cases.size, 1)
N = Campinas_time.size
print('total population:',Campinas_tot_pop)
# Global variables
theta = 0.4786 # social distancing mean for Sao Paulo county
mu = 2e-5 # daily birth and death rate
# initial guesses for parameters for optimization
gamma_0 = 0.01
alpha_0 = 0.4
beta1_0 = 1/14
beta2_0 = 1/5
beta3_0 = 1/14
s0_0 = 0.999
i0_0 = 0.001
p0 = np.array([gamma_0,alpha_0,beta1_0,beta2_0,beta3_0,s0_0,i0_0])
# standard deviation of error is the data
sigma = None
absolute_sigma = False
# check if there is any NaN or InF in data
check_finite = True
# bounds
gamma_min = 0
alpha_min = 0.3
beta1_min = 1/45
beta2_min = 1/21
beta3_min = 1/45
s0_min = 0
i0_min = 0
params_min = [gamma_min,alpha_min,beta1_min,beta2_min,beta3_min,s0_min,i0_min]
gamma_max = 1
alpha_max = 1
beta1_max = 1
beta2_max = 1
beta3_max = 1
s0_max = 0.9999
i0_max = 1
params_max = [gamma_max,alpha_max,beta1_max,beta2_max,beta3_max,s0_max,i0_max]
bounds = (params_min,params_max)
# method
# ‘dogbox’ : dogleg algorithm with rectangular trust regions,
# 'trf' : Trust Region Reflective algorithm
method = 'trf'
# Jacobian
jac = None
# The optimization itself
print('\nRunning the optimization ...')
popt, pvoc = optimization.curve_fit(f,Campinas_time,\
                                    Campinas_cases/Campinas_tot_pop,p0,sigma,\
                                    absolute_sigma,check_finite,bounds,\
                                    method,jac)
# Testing the optimization
perr = np.sqrt(np.diag(pvoc))
print('Standard deviation errors on the parameters = ',perr)
print('\nParameters:')
print('-----------')
print('mu =',mu)
print('theta =',theta)
print('Fitted parameters:')
gamma_opt = popt[0]
print('gamma =',gamma_opt)
alpha_opt = popt[1]
print('alpha =',alpha_opt)
beta1_opt = popt[2]
print('beta1 =',beta1_opt)
beta2_opt = popt[3]
print('beta2 =',beta2_opt)
beta3_opt = popt[4]
print('beta3 =',beta3_opt)
s0_opt = popt[5]
print('s0 =',s0_opt)
i0_opt = popt[6]
print('i0 =',i0_opt)
# initial condition
x0 = np.array([s0_opt,i0_opt,0]) 
# parameters
t0 = 0
tf = N-1
h = 1
#    
#mu = 2e-5 # daily birth and death rate
#theta = 0.4951 # social distancing mean for Sao Paulo county
p=np.zeros(7)
p[0] = mu
p[1] = gamma_opt
p[2] = alpha_opt
p[3] = theta
p[4] = beta1_opt
p[5] = beta2_opt
p[6] = beta3_opt
#
t,sol =  rk4(lambda t,x: rhs(t,x,p),x0,t0,tf,h)
# Saving parameters data to file
data = {'Parameter':['mu','gamma','alpha','theta','beta1','beta2','beta3','s0','i0','sick0'],\
        'Value':np.concatenate((p,x0))} 
#print('parameters =',data)        
df_campinas = pd.DataFrame(data) 
print('Saving parameters in data/campinas_params.csv...')
df_campinas.to_csv("data/campinas_params.csv")
print('done.')
print('Campinas parameters - Latex')
print(df_campinas.to_latex(index=False))
# Plotting figure
print('Printing figure to figures/sick_campinas.png')
plt.figure()
plt.scatter(Campinas_days,Campinas_cases,alpha=0.5)
plt.plot(Campinas_days,sol[:,2]*Campinas_tot_pop,c='r')
plt.grid(axis='both')
plt.legend(['Fitted curve','Real data'])
plt.title('Campinas - Sick population')
plt.xlabel('days')
plt.savefig('figures/sick_campinas.png')
print('done.')
#plt.show()
#plt.close()
# 


# %%
