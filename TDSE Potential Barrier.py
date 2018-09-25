# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:13:50 2018
File Name: TDSE Potential Barrier
@author: C1518116
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import constants
plt.close("all")


hbar = constants.hbar
m = constants.m_e
eV = constants.elementary_charge
k0 = 1e10

#Define number of points for matrix
N = 1000

#Define step size dx
Lmax = 10.0e-9 #Max x
L = np.linspace(-Lmax, Lmax, N)
dx = L[1] - L[0]

#Define 1D array for V
V_box = np.zeros(N)
V_box[700:750] = 6e-19

#Define beta  
beta_box = 2.0 + (2*(dx**2)*m / hbar**2)*V_box

#Define matrix M
M_box = np.diagflat(beta_box) - np.eye(N, k=1) - np.eye(N, k=-1)
M_box = M_box * hbar**2 / (2.0 * m * dx**2)

#Solve for eigenvalues/eigenvectors
eigvals_box, eigvect_box = linalg.eigh(M_box, UPLO='L')

#Define time domain to evaluate over
T_box = np.linspace(0, 1e-13, N)


"""
Positive V potential

|             |
|             |
|  ->   ___   |
|__/\___|  |__|

"""

#Define initial wave function
Phi_ini_box = np.roll((1e9/np.sqrt(2.0*np.pi))*np.exp(-(L**2) / (2*1e-9**2)) \
                      * np.exp(1j * -k0 * L) * 5.954e-5, -200)

#Calculate Cn for 100 eigenstates
n_box = 100
Cn_box = np.zeros(n_box) + 0j
for i in range(np.size(Cn_box)):
    Cn_box[i] = np.sum(eigvect_box[:,i] * Phi_ini_box)


#Calc Phi(t)
Phi_tb = np.zeros((N, N)) +0j
for j in range(N):
    for i in range(np.size(Cn_box)):
        Phi_tb[:, j] += Cn_box[i] * eigvect_box[:,i] * np.exp(1j * eigvals_box[i] * T_box[j] / hbar)


#Plot initial wave function alongside calculated initial wave function and 
#compare.
plt.figure("Box well")
plt.plot(L, np.abs(Phi_ini_box), label=r'$\Psi_i$')
plt.plot(L, np.abs(Phi_tb[:, 0]), '--', label=r'$\Psi$')
plt.xlabel(r'$x$')
plt.ylabel('$\Psi$')
plt.xlim(-1e-8, 1e-8)
plt.legend(loc="best")
plt.show()


"""ANIMATION"""
#Animate the probability density of the wave functions over time.

# First set up the figure, the axis, and the plot element we want to animate
figBox = plt.figure("Animation - Box")
axBox = plt.axes(xlim=(-12, 12), ylim=(-1e8, 10e8))
lineModSqBox, = axBox.plot([], [], label=r'$|\Psi$(t)$|^2$', lw=2)
time_text = axBox.text(0.1, 0.93, '', transform=axBox.transAxes)
lineInfWellBox, = axBox.plot([], [], 'k', lw=2)

# initialization function: plot the background of each frame
def initBox():
    lineModSqBox.set_data([], [])
    lineInfWellBox.set_data([], [])
    time_text.set_text('')
    return lineModSqBox, lineInfWellBox, time_text, 

# animation function.  This is called sequentially
def animateBox(i):
    global L, Phi_tb, V_box

    lineModSqBox.set_data(L*1e9, (Phi_tb[:,i] * Phi_tb[:,i].conjugate()).real) 
    lineInfWellBox.set_data(np.array([-1e-8, -1e-8, L[700], L[700], L[750], L[750], 1e-8, 1e-8])*1e9,\
                            [1e9, 0, 0, V_box[700]*1e8/eV, V_box[749]*1e8/eV, 0, 0, 1e9])
    time_text.set_text(r'$t$ = %#04.1f fs' %(T_box[i] / 1e-15))
    
    return lineModSqBox, lineInfWellBox, time_text,  

# call the animator.  blit=True means only re-draw the parts that have changed.
animSquareBox = animation.FuncAnimation(figBox, animateBox, init_func=initBox, 
                                        frames=500, interval=15, blit=True)

plt.xlabel('L (nm)')
plt.ylabel(r'$|\Psi$(t)$|^2$')
plt.legend(loc=[0.73, 0.9], frameon=False)

#Save animation - requires ffmpeg install!
#animSquareBox.save('TDSE Positive Barrier.mp4', fps=30, dpi=200)

plt.show()


#Plot snapshots of animation 
plt.figure()

ax1 = plt.subplot(311)
plt.plot(L/1e-9, (Phi_tb[:,0] * Phi_tb[:,0].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, L[700], L[700], L[750], L[750], 1e-8, 1e-8])/1e-9,\
         np.array([1e9, 0, 0, V_box[700]*1e8/eV, V_box[749]*1e8/eV, 0, 0, 1e9]), 'k-')
plt.text(-9.5, 8e8, 't = %#04.2ffs'%(T_box[0]*1e15))
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylim(-1e8, 10e8)

ax2 = plt.subplot(312, sharex=ax1)
plt.plot(L/1e-9, (Phi_tb[:,67] * Phi_tb[:,67].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, L[700], L[700], L[750], L[750], 1e-8, 1e-8])/1e-9,\
         np.array([1e9, 0, 0, V_box[700]*1e8/eV, V_box[749]*1e8/eV, 0, 0, 1e9]), 'k-')
plt.text(-9.5, 8e8, 't = %#04.2ffs'%(T_box[67]*1e15))
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel(r'$|\Psi$(t)$|^2$')
plt.ylim(-1e8, 10e8)

ax3 = plt.subplot(313, sharex=ax1)
plt.plot(L/1e-9, (Phi_tb[:,128] * Phi_tb[:,128].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, L[700], L[700], L[750], L[750], 1e-8, 1e-8])/1e-9,\
         np.array([1e9, 0, 0, V_box[700]*1e8/eV, V_box[749]*1e8/eV, 0, 0, 1e9]), 'k-')
plt.text(-9.5, 8e8, 't = %#04.2ffs'%(T_box[128]*1e15))
plt.ylim(-1e8, 10e8)
plt.xlabel('L (nm)')

plt.subplots_adjust(hspace=0.25)

#plt.savefig('TDSE Positive Barrier.png', dpi=200)
plt.show()


