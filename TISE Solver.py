# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:26:13 2017
File Name: Solving Schrodinger's equation
@author: Daniel Martin
Last edited 07/03/2018

This python script calculates the solution to the Schrodinger equation for an 
inifinite square well, a parabolic well and a linear well. The results are then
plotted and the graphs are displayed. The script uses the numpy, matplotlib and
scipy libraries. Version details are below:
Python version: 3.6.4
sciPy version: 1.0.0
numpy version: 1.14.0
matplotlib version: 2.1.2

This script was written in the Spyder IDE, version 3.2.6
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy import constants
#import timeit
plt.close("all")
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

hbar = constants.hbar
m = constants.m_e
eV = constants.elementary_charge
k = 1.0
omega = np.sqrt(k / m)
grad_lin = 0.25e-11

#Define number of points for matrix
N = 1000 

#Define step size dx
Lmax = 10.0e-9 #Max x of well
L = np.linspace(-Lmax, Lmax, N)
dx = L[1] - L[0]

#Define 1D array for V
V_square = np.zeros(N)
V_parabola = 0.5 * k * (L)**2
V_linear = grad_lin * (L + Lmax)

#Define beta  
beta_square = 2.0 + (2*(dx**2)*m / hbar**2)*V_square
beta_parabola = 2.0 + (2*(dx**2)*m / hbar**2)*V_parabola
beta_linear = 2.0 + (2*(dx**2)*m / hbar**2)*V_linear

#Define matrix M
M_square = np.diagflat(beta_square) - np.eye(N, k=1) - np.eye(N, k=-1)
M_square = M_square * hbar**2 / (2.0 * m * dx**2)
M_parabola = np.diagflat(beta_parabola) - np.eye(N, k=1) - np.eye(N, k=-1)
M_parabola = M_parabola * hbar**2 / (2.0 * m * dx**2)
M_linear = np.diagflat(beta_linear) - np.eye(N, k=1) - np.eye(N, k=-1)
M_linear = M_linear * hbar**2 / (2.0 * m * dx**2)

#Solve for eigenvalues/eigenvectors
eigvals_square, eigvect_square = linalg.eigh(M_square, UPLO='L')
eigvals_parabola, eigvect_parabola = linalg.eigh(M_parabola, UPLO='L')
eigvals_linear, eigvect_linear = linalg.eigh(M_linear, UPLO='L')


"""
#Calculate time taken to calculate eigenvals for linalg.eig, linalg.eigh(UPLO='L'),
#and linalg.eigh(UPLO='U') and average them
start_time = np.zeros(10)
elapsed = np.zeros(10)
#Solve for eigenvalues/eigenvectors
for i in range(10):
    start_time[i] = timeit.default_timer()
    eigvals_square, eigvect_square = linalg.eig(M_square)
    eigvals_parabola, eigvect_parabola = linalg.eig(M_parabola)
    elapsed[i] = timeit.default_timer() - start_time[i]
print("linalg.eig takes %.3fs" %np.mean(elapsed))

for i in range(10):
    start_time[i] = timeit.default_timer()
    eigvals_square, eigvect_square = linalg.eigh(M_square, UPLO='L')
    eigvals_parabola, eigvect_parabola = linalg.eigh(M_parabola, UPLO='L')
    elapsed[i] = timeit.default_timer() - start_time[i]
print("linalg.eigh takes %.3fs with lower part" %np.mean(elapsed))

for i in range(10):
    start_time[i] = timeit.default_timer()
    eigvals_square, eigvect_square = linalg.eigh(M_square, UPLO='U')
    eigvals_parabola, eigvect_parabola = linalg.eigh(M_parabola, UPLO='U')
    elapsed[i] = timeit.default_timer() - start_time[i]
print("linalg.eigh takes %.3fs with upper part" %np.mean(elapsed))
"""


# Calculate En values for square and parabolic wells
En_square = np.zeros(3)
print("For Square potential:")
for i in range(np.size(En_square)):
    En_square[i] = (hbar**2 * np.pi**2 * (i+1)**2 / (2.0 * m * (2*Lmax)**2)) / eV
    print('Expected E_%d = %.3feV from formula' %(i+1, En_square[i]))
    print('Calculated E_%d= %.3feV from linalg' %(i+1, eigvals_square[i] / eV))
    print('error is %.3eeV\n' %abs(En_square[i] - (eigvals_square[i] / eV)))


En_parabola = np.zeros(3)
print("\n\nFor Parabolic potential:")
for i in range(np.size(En_parabola)):
    En_parabola[i] = (i + 0.5) * hbar * np.sqrt(k / m) / eV
    print('Expected E_%d = %.3feV from formula' %(i, En_parabola[i]))
    print('Calculated E_%d= %.3feV from linalg' %(i, eigvals_parabola[i] / eV))
    print('error is %.3eeV\n'%abs(En_parabola[i] - (eigvals_parabola[i] / eV)))


En_linear = np.zeros(3)
print("\n\nFor Linear potential:")
for i in range(np.size(En_linear)):
    En_linear[i] = np.power(hbar**2 / (2.0*m), 1.0/3.0) * \
        np.power(1.5*np.pi * grad_lin * (i+1 - 0.25), 2.0/3.0) / eV
    print('Expected E_%d = %.3feV from formula' %(i+1, En_linear[i]))
    print('Calculated E_%d= %.3feV from linalg' %(i+1, eigvals_linear[i] / eV))
    print('error is %.3eeV\n'%abs(En_linear[i] - (eigvals_linear[i] / eV)))


#PLOTS


#Plot square well
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for i in range(3):
    ax1.plot(L*1e9, (i+1)**2 + (-1*eigvect_square[:, i])*25, label=r"$\psi_%d$"%(i+1))
    ax2.plot(np.array([-10e-9, 10e-9])*1e9, np.ones(2)*eigvals_square[i]*1e3/eV, '--',\
        alpha=0.5, label=r"$E_%d$"%(i+1))


ax1.plot(np.array([-Lmax, -Lmax, Lmax, Lmax])*1e9, np.array([11, 0, 0, 11]), 'k')

ax1.set_xlabel("L (nm)")
ax1.set_ylabel(r'$\psi_n$')
ax2.set_ylabel(r'$E_n$ (meV)')
ax1.set_ylim(-0.1, 11)
ax2.set_ylim(-0.0001*1e3, 0.0105*1e3)
plt.xlim(-1.6e-8*1e9, 1.6e-8*1e9)
ax1.legend(loc="lower left")
ax2.legend(loc="lower right")
#plt.title("Infinite Square Well")

#plt.savefig("TISE Square well.png", dpi = 500)



#Plot parabolic well
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for i in range(3):
    ax1.plot(L*1e9, (i + 0.5) + eigvect_parabola[:, i]*2.5, label=r"$\psi_%d$"%i)
    ax2.plot(np.array([-0.15e-8, 0.15e-8])*1e9, eigvals_parabola[i]*np.ones(2)/eV,\
        '--', alpha=0.5, label=r"$E_%d$"%i)

ax1.plot(L*1e9, V_parabola/eV, 'k')

ax1.set_xlabel("L (nm)")
ax1.set_ylabel(r'$\psi_n$')
ax2.set_ylabel(r'$E_n$ (eV)')
ax1.set_ylim(0, 3)
ax2.set_ylim(0, 3*np.average((eigvals_parabola[0:3]/eV)/np.array([0.5, 1.5, 2.5])))
plt.xlim(-0.15e-8*1e9, 0.15e-8*1e9)
ax1.legend(loc="lower left") 
ax2.legend(loc="lower right")
#plt.title("Parabolic Well")

#plt.savefig("TISE Parabolic well.png", dpi = 500)



# Plot linear well
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for i in range(3):
    ax1.plot((L + Lmax)*1e9, (i + 1) - eigvect_linear[:, i]*5, label=r"$\psi_%d$"%(i+1))
    ax2.plot(np.array([0, Lmax+0.05e-8])*1e9, eigvals_linear[i]*np.ones(2)*1e3/eV,\
        '--', alpha=0.5, label=r"$E_%d$"%(i+1))

ax1.plot(np.array([0, 0, max(L)+1e-8])*1e9, np.array([4, 0, max(V_linear)/(0.042*eV)]), 'k')

ax1.set_xlabel("L (nm)")
ax1.set_ylabel(r'$\psi_n$')
ax2.set_ylabel(r'$E_n$ (meV)')
ax1.set_ylim(-0.2, 3.7)
ax2.set_ylim(-0.008*1e3, 0.15*1e3)
plt.xlim(-0.05e-8*1e9, 1.05e-8*1e9)
ax1.legend(loc="lower center")
ax2.legend(loc="lower right")
#plt.title("Linear Well")

#plt.savefig("TISE Linear well.png", dpi = 500)

plt.show()
