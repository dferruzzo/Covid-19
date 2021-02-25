# -*- coding: utf-8 -*-
"""
Created on Sat Fev 06, 2021

@author: Diego Ferruzzo
"""
#%%
# Carregando librarias
# from sympy import transpose as tp
import numpy as np
import pandas as pd 
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from myfunctions import rk4 
import datetime
#%% São Paulo
print('\nCarregando Índice de Isolamento São Paulo')
print('----------------------------')
SaoPaulo_isol_data = pd.read_csv("data/SaoPaulo_isolamento.csv")
SaoPaulo_isol_data = SaoPaulo_isol_data.to_numpy()
#x_values = [datetime.datetime.strptime(d,"%Y-%m-%d").date() for d in SaoPaulo_isol_data[:,1]]
#%% Calculando a média movil a 7 dias


#%%
# Plotando
#print('Printing figure to figures/sick_saopaulo.png')
plt.figure()
#plt.scatter(SaoPaulo_days,SaoPaulo_cases,alpha=0.5)
plt.plot(SaoPaulo_isol_data[:,1],SaoPaulo_isol_data[:,2])
#plt.grid(axis='both')
#plt.legend(['Fitted curve','Real data'])
#plt.title('São Paulo - Sick population')
#plt.xlabel('days')
#plt.savefig('figures/sick_saopaulo.png')
#print('done.')
plt.show()
#plt.close()
# %%
