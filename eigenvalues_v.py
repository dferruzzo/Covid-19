#%% -*- coding: utf-8 -*-
"""
Created on Fri Aug 08, 2020

@author: Diego Ferruzzo
-----------------------------------

Script for testing the free-disease equilibrium and for computing the
endemic equilibrium for the model.
"""
import pandas as pd 
import sympy as sym
# vector state
s,i,sick,r,v = sym.symbols('s,i,sick,r,v', positive=True)
x = sym.Matrix([s,i,sick,v])
# parameters
mu, gamma, alpha, theta, omega, beta1, beta2, beta3, zeta=\
    sym.symbols('mu,gamma,alpha,theta,omega,beta1,beta2,beta3,zeta',\
                positive=True) 
# The right-hand side
f = sym.Matrix([[mu + gamma - alpha*(1-theta)*s*i\
                 -(mu + gamma + omega)*s - gamma*i - gamma*sick\
                     +(omega*(1-zeta)-gamma)*v],\
                [alpha*(1-theta)*s*i-(beta1+beta2+mu)*i],\
                [beta2*i-(beta3 +mu)*sick],\
                [omega*s-(omega*(1-zeta)+mu)*v]])
print('')
print('The right-hand side')
print('-------------------')
print('f =',f) 
#%% Computing the free-disease equilibrium (s_fd,i_fd,sick_fd,v_fd)
# computing s = s(v)
print('\nThe free-disease equilibrium')
print('----------------------------')
s_v_fd = sym.solve(f[0].subs(i,0).subs(sick,0),s)[0]
i_fd = 0
sick_fd = 0
# substituting s = s(v) into f[3]
v_fd = sym.solve(f[3].subs(s,s_v_fd),v)[0]
# computing s
s_fd = sym.simplify(s_v_fd.subs(v,v_fd))
#
# testing the free-disease equilibrium
print('\nTesting the free-disease equilibrium')
print('------------------------------------')
print('s^* =',s_fd)
print('i^* =',i_fd)
print('sick^* =',sick_fd)
print('v^* =',v_fd)
print(sym.simplify(f.subs(s, s_fd).subs(i, 0).subs(sick, 0).subs(v,v_fd)))
print('s^* =',sym.latex(s_fd))
print('v^* =',sym.latex(v_fd))
# %% Computing eigenvalues for the free-disease equilibrium
A = f.jacobian(x)
print('\nThe Jacobian for the free-disease equilibrium')
print('---------------------------------------------')
print('A=',A)
print('A=',sym.latex(A))
# %% Computing eigenvalues for the free-disease equilibrium
Eig_fd = list(A.subs(i,i_fd).eigenvals())
print('\nEigenvalues for the free-disease equilibrium')
print('--------------------------------------------')
print('Lambda 1 =',sym.latex(Eig_fd[0]))
print('Lambda 2 =',sym.latex(Eig_fd[1]))
print('Lambda 3 =',sym.latex(Eig_fd[2]))
print('Lambda 4 =',sym.latex(Eig_fd[3]))
# %% Asymptotic stability condition as a function of w
w_con = sym.solve(Eig_fd[3].subs(s,s_fd),omega)
print('\nThe asymptotic stability condition for the free-disease equilibrium')
print('-------------------------------------------------------------------')
print('w_cond =',w_con)
print('w_con =',sym.latex(w_con))
# %%
print('\nThe endemic equilibrium')
print('-----------------------')
# computing s = s(i) from f[1]
s_en = sym.solve(f[1],s)[0]
print('s^* =',sym.latex(s_en))
# computing v from f[3]
v_en = sym.simplify(sym.solve(f[3].subs(s,s_en),v)[0])
print('v^* =',sym.latex(v_en))
# computing sick = sick(i) from f[2]
sick_i = sym.solve(f[2],sick)[0]
print('sick_i =',sym.latex(sick_i))
# substituting s, v and sick_i into f[0] ans solve for i
#i_en = sym.simplify(sym.solve(f[0].subs(s,s_en).subs(v,v_en).subs(sick,sick_i),i)[0])
i_en = sym.simplify(sym.solve(f[0].subs(s,s_en).subs(sick,sick_i),i)[0])
print('i^* =',sym.latex(sym.factor(i_en)))
# finally, compute sick_en, it's going to be a long expression
sick_en = sym.simplify(sick_i.subs(i,i_en))
print('sick^* =',sym.latex(sym.factor(sick_en)))
#
print('\nTesting the endemic equilibrium')
print('f(s^*,i^*,sick^*,v^*) =',sym.simplify(f.subs(s,s_en).subs(i,i_en).subs(sick,sick_en).subs(v,v_en)))
# %%
#sym.collect(sym.collect(sym.collect(sym.collect(sym.collect(sym.collect(sym.collect(sym.factor(i_en),mu),alpha),theta),omega),beta1),beta3),beta2)
print('\ni^* as a function of v^*')
print('------------------------')
print('i^*=',sym.latex(sym.collect(sym.collect(sym.collect(sym.collect(sym.collect(sym.collect(sym.factor(i_en),v),gamma),alpha),omega),mu),theta)))
#%% substituting v^* into i^*
i_en_f = sym.simplify(i_en.subs(v,v_en))
num_i, den_i = fraction(i_en_f)
a = simplify(num_i.leadterm(omega)[0])
b_temp = num_i-a*omega**2
b = b_temp.leadterm(omega)[0]
c = sym.simplify(num_i - a*omega**2 - b*omega)
#%% 
""" 
The existence condition for the endemic equilibrium depend on the 
positiveness the numetaror of i^*, in this case num_i, so we solve num_i for 
omega
"""
omega_cond = sym.solve(num_i,omega,dict=True,set=True,simplify=True)
