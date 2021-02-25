#%% -*- coding: utf-8 -*-
"""
Created on Aug 09 2020

@author: Diego Ferruzzo
-----------------------------------

Script for testing the free-disease equilibrium and for computing the
endemic equilibrium for the model:

 ds/dt = mu + gamma - alpha*(1-theta)*s*i -(mu+gamma+omega)*s - gamma*i
         -gamma*sick;
 di/dt = alpha*(1-theta)*s*i - (beta1+beta2+mu)*i
 dsick/dt = beta2*i -(beta3 +mu)*sick
 
"""
import pandas as pd 
import sympy as sym
# vector state
s,i,sick = sym.symbols('s,i,sick', positive=True)
x = sym.Matrix([s,i,sick])
# pamareters
mu, gamma, alpha, theta, omega, beta1, beta2, beta3=\
    sym.symbols('mu,gamma,alpha,theta,omega,beta1,beta2,beta3', positive=True) 
varphi = sym.symbols('varphi', real=True)    
#%% Model without vaccination
f1 = sym.Matrix([[mu + gamma - alpha*(1-theta)*s*i\
                 -(mu + gamma )*s - gamma*i - gamma*sick],\
                [alpha*(1-theta)*s*i-(beta1+beta2+mu)*i],\
                [beta2*i-(beta3 +mu)*sick]])
print('')
print('The right-hand side of the model without vacciantion')
print('---------------------------------------------------')
print('f1 =',f1) 
# The jacobian
A1 = f1.jacobian([s,i,sick])
print('')
print('The Jacobian for the model without vacination')
print('---------------------------------------------')
print('A1=',A1)
# The free-disease equilibrium
s1,i1,sick1 = 1,0,0
# Eigenvalues at the free-disease equilibrium
Eig_fd_m1 = A1.subs(s,1).subs(i,0).subs(sick,0).eigenvals()
print('')
print('Eigenvalues at the free-disease equilibrium for the model without vaccination')
print('-----------------------------------------------------------------------------')
print('lambda 1 = ',list(Eig_fd_m1)[0])
print('lambda 2 = ',list(Eig_fd_m1)[1])
print('lambda 3 = ',list(Eig_fd_m1)[2])
#
#R0 = alpha*(1-theta)/(beta1+beta2+mu)

# The endemic equilibrium
s_en_m1 = sym.solve(f1[1],s)[0]
sick_i_m1 = sym.solve(f1[2],sick)[0]
i_en_m1 = sym.simplify(sym.solve(f1[0].subs(s,s_en_m1).subs(sick,sick_i_m1),i)[0]).factor()
sick_en_m1 = sym.simplify(sym.solve(f1[2].subs(i,i_en_m1),sick)[0]).factor()
#
i_en_m1_a = (beta3+mu)*varphi
sick_en_m1_a = beta2*varphi
#
# computing eigenvalues at the endemic equilibrium
charpol_en_m1 = A1.subs(s,s_en_m1).subs(i,i_en_m1_a).subs(sick,sick_en_m1_a).charpoly()
#
a1_m1 = charpol_en_m1.coeffs()[1]
a2_m1 = charpol_en_m1.coeffs()[2]
a3_m1 = charpol_en_m1.coeffs()[3]
#
b1_m1 = sym.simplify(a2_m1 - (a3_m1/a1_m1))





#%% Model with vaccination
# The right-hand side
f = sym.Matrix([[mu + gamma - alpha*(1-theta)*s*i\
                 -(mu + gamma + omega)*s - gamma*i - gamma*sick],\
                [alpha*(1-theta)*s*i-(beta1+beta2+mu)*i],\
                [beta2*i-(beta3 +mu)*sick]])
print('')
print('The right-hand side')
print('-------------------')
print('f =',f) 
#%% Computing the free-disease equilibium (s_fd,i_fd,sick_fd)
s_fd = sym.solve(f[0].subs(i,0).subs(sick,0),s)[0]
i_fd = 0
sick_fd = 0
# testing the free-disease equilibrium
print('Testing the free-disease equilibrium')
print('------------------------------------')
print('s^* =',s_fd)
print('i^* =',i_fd)
print('sick^* =',sick_fd)
print(sym.simplify(f.subs(s, s_fd).subs(i, 0).subs(sick, 0)))
# Computing the jacobian
A = f.jacobian([s,i,sick])
print('')
print('The Jacobian')
print('------------')
print('A =',A,  )
# substituting the free-disease equilibrium in the jacobian
A_fd = A.subs(s, s_fd).subs(i, i_fd)
# computing eigenvalues
Eig_fd = A_fd.eigenvals()
print('')
print('Free-disease equilibrium - Eigenvalues')
print('--------------------------------------')
Eig1_fd = list(Eig_fd)[0]
print('Lambda 1 =',Eig1_fd)
Eig2_fd = list(Eig_fd)[1]
print('Lambda 2 =',Eig2_fd)
Eig3_fd = list(Eig_fd)[2]
print('Lambda 3 =',Eig3_fd)
#%% Computing the endemic equilibrium (s_en,i_en, sick_en)
# from f[1] computing s_en
s_en = sym.solve(f[1],s)[0]
# computing from f[2], computing sick = sick(i)
sick_i = sym.solve(f[2],sick)[0]
# substituting s_en and sick_i in f[0]
i_en = sym.solve(f[0].subs(s, s_en).subs(sick, sick_i),i)[0]
# computing sick_en
sick_en = sick_i.subs(i, i_en)
# testing the endemic equilibrium
print('')
print('Testing the Endemic equilibrum')
print('------------------------------')
print('s^* =',s_en)
print('i^* =',i_en)
print('sick^* =',sick_en)
print('')
print('f(s^*,i^*,sick^*) =')
print(sym.simplify(f.subs(s, s_en).subs(i, i_en).subs(sick, sick_en)))
#%% defining an auxiliary variable
varphi = sym.symbols('varphi',real=True)
i_en_1 = (beta3+mu)*varphi
sick_en_1 = beta2*varphi
#%% computing the endemic-equilibrium existence condition
num_cond_en,den_cond_en = fraction(sick_en/beta2)
exist_cond_en = sym.solve(num_cond_en,omega)[0]
#%%
# substituting the endemic equilibrium into the Jacobian
A_en = A.subs(s, s_en).subs(i, i_en).subs(sick, sick_en)
A_en_1 = A.subs(s, s_en).subs(i, i_en_1).subs(sick, sick_en_1)
# computing eigenvalues
print('')
print('Eigenvlaues at endemic equilibrium')
print('---------------------------------')
Eig_en = A_en.eigenvals()
Eig1_en = list(Eig_en)[0]
print('Lambda 1 =',Eig1_en)
Eig2_en = list(Eig_en)[1]
print('Lambda 2 =',Eig2_en)
Eig3_en = list(Eig_en)[2]
print('Lambda 3 =',Eig3_en)
#%%
Eig_en_1 = A_en_1.eigenvals()
Eig1_en_1 = list(Eig_en_1)[0]
print('Lambda 1 =',Eig1_en_1)
Eig2_en_1 = list(Eig_en_1)[1]
print('Lambda 2 =',Eig2_en_1)
Eig3_en_1 = list(Eig_en_1)[2]
print('Lambda 3 =',Eig3_en_1)
#%% characteristic polynomial
pol_A_en_1= A_en_1.charpoly()
# extracting the coefficients
polcoeff = pol_A_en_1.coeffs()
a0 = polcoeff[0]
a1 = polcoeff[1]
a2 = polcoeff[2]
a3 = polcoeff[3]
#%% computing the Ruth-Hurtwitz third coefficient
b1 = sym.simplify((a1*a2-a3)/a1)
num,den = fraction(b1)
#%% computing the Ruth-Hurtwitz third coefficient for the coefficients of A_en
pol_A_en= A_en.charpoly()
# extracting the coefficients
polcoeff_c = pol_A_en.coeffs()
c0 = polcoeff_c[0]
c1 = polcoeff_c[1]
c2 = polcoeff_c[2]
c3 = polcoeff_c[3]
# computing the Ruth-Hurtwitz third coefficient
d1 = sym.simplify((c1*c2-c3)/c1)
num_1,den_1 = fraction(d1)
#%%
h1 = sym.simplify(sym.solve(num_1,omega)[0])
h2 = sym.simplify(sym.solve(den_1,omega)[0])

#%% Computing numerically eigenvalues. 
# getting data from csv file
print('')
print('Loading parameters')
print('------------------')
saopaulo_params = pd.read_csv("data/saopaulo_params.csv").to_numpy()
print(saopaulo_params)
#%% 
mu_opt = saopaulo_params[0,2]
gamma_opt = saopaulo_params[1,2]
alpha_opt = saopaulo_params[2,2]
theta_opt = saopaulo_params[3,2]
beta1_opt = saopaulo_params[4,2]
beta2_opt = saopaulo_params[5,2]
beta3_opt = saopaulo_params[6,2]
#%% computing stability / existence condition for the free-disease equilibrium
def sta_cond(alpha,theta,beta1,beta2,mu,gamma):
    return ((alpha*(1-theta)/(beta1+beta2+mu))-1)*(mu+gamma)
#
w_cond = sta_cond(alpha_opt,theta_opt,beta1_opt,beta2_opt,mu_opt,gamma_opt)
print('Omega condition =',w_cond)
#%% computing eigenvalues for the free-disease equilibrium
print('')
print('Computing eigenvalues for free-disease equilibrium')
print('--------------------------------------------------')
Eig1n_fd = Eig1_fd.subs(mu, mu_opt).subs(gamma, gamma_opt).\
    subs(alpha, alpha_opt).subs(theta, theta_opt).subs(beta1, beta1_opt).\
        subs(beta2, beta2_opt).subs(beta3, beta3_opt)
print('Lambda 1 =', Eig1n_fd)
#
Eig2n_fd = Eig2_fd.subs(mu, mu_opt).subs(gamma, gamma_opt).\
    subs(alpha, alpha_opt).subs(theta, theta_opt).subs(beta1, beta1_opt).\
        subs(beta2, beta2_opt).subs(beta3, beta3_opt)
print('Lambda 2 =', Eig2n_fd)
#
Eig3n_fd = Eig3_fd.subs(mu, mu_opt).subs(gamma, gamma_opt).\
    subs(alpha, alpha_opt).subs(theta, theta_opt).subs(beta1, beta1_opt).\
        subs(beta2, beta2_opt).subs(beta3, beta3_opt)
print('Lambda 3 =', Eig3n_fd)
# %% computing eigenvalues for the endemic equilibrium
print('')
print('Computing eigenvalues for endemic equilibrium')
print('--------------------------------------------------')
Eig1n_en = Eig1_en.subs(mu, mu_opt).subs(gamma, gamma_opt).\
    subs(alpha, alpha_opt).subs(theta, theta_opt).subs(beta1, beta1_opt).\
        subs(beta2, beta2_opt).subs(beta3, beta3_opt).evalf()
print('Lambda 1 =', Eig1n_en)
#
Eig2n_en = Eig2_en.subs(mu, mu_opt).subs(gamma, gamma_opt).\
    subs(alpha, alpha_opt).subs(theta, theta_opt).subs(beta1, beta1_opt).\
        subs(beta2, beta2_opt).subs(beta3, beta3_opt).evalf()
print('Lambda 2 =', Eig2n_en)
#
Eig3n_en = Eig3_en.subs(mu, mu_opt).subs(gamma, gamma_opt).\
    subs(alpha, alpha_opt).subs(theta, theta_opt).subs(beta1, beta1_opt).\
        subs(beta2, beta2_opt).subs(beta3, beta3_opt).evalf()
print('Lambda 3 =', Eig3n_en)
#%% 
h1_num =  h1.subs(alpha,alpha_opt).subs(beta1,beta1_opt).subs(beta2,beta2_opt).subs(beta3,beta3_opt).subs(gamma,gamma_opt).subs(mu,mu_opt).subs(theta,theta_opt).evalf()
h2_num =  h2.subs(alpha,alpha_opt).subs(beta1,beta1_opt).subs(beta2,beta2_opt).subs(beta3,beta3_opt).subs(gamma,gamma_opt).subs(mu,mu_opt).subs(theta,theta_opt).evalf()
exist_cond_en_n = exist_cond_en.subs(alpha,alpha_opt).subs(beta1,beta1_opt).subs(beta2,beta2_opt).subs(beta3,beta3_opt).subs(gamma,gamma_opt).subs(mu,mu_opt).subs(theta,theta_opt).evalf()