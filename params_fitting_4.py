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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from myfunctions import rk4
from scipy.integrate import odeint


"""
PAREI AQUI 10/02/2021
O AJUSTE POLINOMIAL DO ISOLAMENTO PARECE FUNCIONA EM R MAS NÃO CONSIGO REPRODUZIR AQUI EM PYTHON
PRECISO FAZER O AJUSTE DOS DADOS DE ISOLAMENTO AQUI EM PYTHON.

PASSOS
------
1. COLETAR OS DADOS DO ÍNDICE DE ISOLAMENTO (OKAY)
2. COMPUTAR A MÉDIA MÓVEL PARA 7, 14 3 21 DIAS (OKAY)
3. AJUSTAR A CURVA DA MÉDIA MÓVEL COM UM POLINÔMIO DE ORDEM 5, 7, 9
4. TESTAR OS POLINÔMIOS.
"""


"""
Nesse modelo estou utilizando theta como função de polinomial t, na forma
theta(t) = a9*t^9 + a8*t^8 + a7*t^7 + a6*t^6 + a5^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0

Dados obtidos de R
Coefficients:
                                    Estimate Std. Error t value Pr(>|t|)    
(Intercept)                        2.209e-01  7.384e-03  29.923  < 2e-16 ***
poly(tempo_media, 9, raw = TRUE)1  3.093e-02  1.256e-03  24.619  < 2e-16 ***
poly(tempo_media, 9, raw = TRUE)2 -1.173e-03  6.984e-05 -16.793  < 2e-16 ***
poly(tempo_media, 9, raw = TRUE)3  2.257e-05  1.795e-06  12.573  < 2e-16 ***
poly(tempo_media, 9, raw = TRUE)4 -2.519e-07  2.507e-08 -10.046  < 2e-16 ***
poly(tempo_media, 9, raw = TRUE)5  1.713e-09  2.056e-10   8.330 2.42e-15 ***
poly(tempo_media, 9, raw = TRUE)6 -7.178e-12  1.017e-12  -7.055 1.08e-11 ***
poly(tempo_media, 9, raw = TRUE)7  1.805e-14  2.985e-15   6.047 4.13e-09 ***
poly(tempo_media, 9, raw = TRUE)8 -2.491e-17  4.777e-18  -5.215 3.32e-07 ***
poly(tempo_media, 9, raw = TRUE)9  1.448e-20  3.212e-21   4.507 9.23e-06 ***

onde:
a0 = 2.209e-01
a1 = 3.093e-02
a2 = -1.173e-03 
a3 = 2.257e-05
a4 = -2.519e-07
a5 = 1.713e-09
a6 = -7.178e-12
a7 = 1.805e-14 
a8 = -2.491e-17
a9 = 1.448e-20        
"""

#-----------------------------------------------------------------------------------------------------------------------

def rhs(x, t, mu, gamma, alpha, beta1, beta2, beta3):
    # modelo 2 -  simplificado
    # di/dt = alpha*(1-(theta+a0))*s*i - (beta1 + beta2 + mu)*i
    # dsick/dt = beta2*i - (beta3 + mu)* sick
    # dtheta/dt = (a9*9)*t^8 + (a8*8)*t^7 + (a7*7)*t^6 + (a6*6)*t^5 +
    #             (a5*5)*t^4 + (a4*4)*t^3 + (a3*3)*t^2 + (a2*2)*t + a1
    #
    # this is the right-hand side function, implemented for numerical integration
    s = x[0]
    i = x[1]
    sick = x[2]
    theta = x[3]
    #
    # mu = p[0]
    # gamma = p[1]
    # alpha = p[2]
    # beta1 = p[3]
    # beta2 = p[4]
    # beta3 = p[5]
    #
    # Coeficientes para theta(t)
    # a0 = 2.209e-01
    a1 = 3.093e-02
    a2 = -1.173e-03
    a3 = 2.257e-05
    a4 = -2.519e-07
    a5 = 1.713e-09
    a6 = -7.178e-12
    a7 = 1.805e-14
    a8 = -2.491e-17
    a9 = 1.448e-20

    #
    return np.array([mu + gamma - alpha * (1 - theta) * s * i - (mu + gamma) * s - gamma * i - gamma * sick,
                     alpha * (1 - theta) * s * i - (beta1 + beta2 + mu) * i,
                     beta2 * i - (beta3 + mu) * sick,
                     (a9 * 9) * t ** 8 + (a8 * 8) * t ** 7 + (a7 * 7) * t ** 6 + (a6 * 6) * t ** 5 + (
                                 a5 * 5) * t ** 4 + (a4 * 4) * t ** 3 + (a3 * 3) * t ** 2 + (a2 * 2) * t + a1])

def theta_fun(t):
    # Função para testar os coeficientes.
    # Coeficientes para theta(t)
    a0 = 2.209e-01
    a1 = 3.093e-02
    a2 = -1.173e-03
    a3 = 2.257e-05
    a4 = -2.519e-07
    a5 = 1.713e-09
    a6 = -7.178e-12
    a7 = 1.805e-14
    a8 = -2.491e-17
    a9 = 1.448e-20
    return(a9 * t ** 9 + a8 * t ** 8 + a7 * t ** 7 + a6 * t ** 6 + a5 * t ** 5
           + a4 * t ** 4 + a3 * t ** 3 + a2 * t ** 2 + a1 * t + a0)

def rhs1(x, t, mu, gamma, alpha, beta1, beta2, beta3):
    # modelo -  simplificado
    #
    # theta(t) = a9*t^9 + a8*t^8 + a7*t^7 + a6*t^6 + a5^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
    #
    # di/dt = alpha*(1-(theta+a0))*s*i - (beta1 + beta2 + mu)*i
    # dsick/dt = beta2*i - (beta3 + mu)* sick
    #
    # this is the right-hand side function, implemented for numerical integration
    s = x[0]
    i = x[1]
    sick = x[2]
    #
    """
    mu = p[0]
    gamma = p[1]
    alpha = p[2]
    beta1 = p[3]
    beta2 = p[4]
    beta3 = p[5]
    """
    #
    # Coeficientes para theta(t)
    a0 = 2.209e-01
    a1 = 3.093e-02
    a2 = -1.173e-03
    a3 = 2.257e-05
    a4 = -2.519e-07
    a5 = 1.713e-09
    a6 = -7.178e-12
    a7 = 1.805e-14
    a8 = -2.491e-17
    a9 = 1.448e-20

    theta = a9 * t ** 9 + a8 * t ** 8 + a7 * t ** 7 + a6 * t ** 6 + a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3 + a2 * t ** 2 + a1 * t + a0
    #
    return np.array([mu + gamma - alpha * (1 - theta) * s * i - (mu + gamma) * s - gamma * i - gamma * sick,
                     alpha * (1 - theta) * s * i - (beta1 + beta2 + mu) * i,
                     beta2 * i - (beta3 + mu) * sick])

#-----------------------------------------------------------------------------------------------------------------------

# carregando data do índice de isolamento


saopaulo_isol_data = pd.read_csv("data/SaoPaulo_isolamento.csv")
#saopaulo_isol_data = saopaulo_isol_data.to_numpy()

#saopaulo_isol = saopaulo_isol_data[:, 2]
#saopaulo_isol_days = pd.to_datetime(saopaulo_isol_data[:, 1], yearfirst=True)
#saopaulo_isol_time = np.arange(0, saopaulo_isol.size, 1)
#N = saopaulo_isol_time.size

saopaulo_isol_df = pd.DataFrame(saopaulo_isol_data)
saopaulo_isol_df.drop(labels='Unnamed: 0', axis=1, inplace=True)
saopaulo_isol_df = saopaulo_isol_df.rename(columns={'Isol':'Dados'})
print(saopaulo_isol_df.head())
saopaulo_isol_df['SMA 7'] = saopaulo_isol_df.iloc[:, 1].rolling(window=7).mean()
saopaulo_isol_df['SMA 14'] = saopaulo_isol_df.iloc[:, 1].rolling(window=14).mean()
saopaulo_isol_df['SMA 21'] = saopaulo_isol_df.iloc[:, 1].rolling(window=21).mean()

ax1 = saopaulo_isol_df.plot(x='Data', y='SMA 7', kind='line', linewidth=2)
ax2 = saopaulo_isol_df.plot(x='Data', y='SMA 14', kind='line', linewidth=2, ax=ax1)
ax3 = saopaulo_isol_df.plot(x='Data', y='SMA 21', kind='line', linewidth=2, ax=ax2)
saopaulo_isol_df.plot(x='Data', y='Dados', kind='scatter', color='gray', ax=ax3)
plt.legend(['Média Móvel 7d', 'Média Móvel 14d', 'Média Móvel 21d', 'Dados'])
plt.xlabel('Datas')
plt.ylabel('Índice de isolamento')
plt.grid()
plt.show()

# Ajustando a curva de isolamento
#X = np.arange(0, , 1)

"""
# testando a função rhs
# a0 = 2.209e-01
# x0 = np.array([0.99, 0.01, 0, a0])
x0 = np.array([0.99, 0.01, 0])
t0 = 0
tf = 240
h = 1
tempo = np.linspace(t0, tf, 1 * (tf - t0), endpoint=True)
gamma = 0
alpha = 0.3
beta1 = 1 / 45
beta2 = 1 / 21
beta3 = 1 / 45
p = np.array([mu, gamma, alpha, beta1, beta2, beta3])
# t, sol = rk4(lambda t, x: rhs(t, x, p), x0, t0, tf, h)
sol = odeint(rhs1, x0, tempo, args=(mu, gamma, alpha, beta1, beta2, beta3))
print(sol)
#
plt.figure()
plt.plot(tempo, theta_fun(tempo))
plt.grid(axis='both')
#plt.title('Índice de isolamento para São Paulo - Aproximação Linear')
#plt.xlabel('days')
plt.show()

plt.figure()
plt.plot(t, sol[:, 3])
plt.grid(axis='both')
plt.title('Índice de isolamento para São Paulo - Aproximação Linear')
plt.xlabel('days')
plt.show()
#


# %% Parâmetros para fitting
# Global variables

mu = 2e-5  # daily birth and death rate

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
"""
