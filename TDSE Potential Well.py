# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:20:11 2018
File Name: TDSE Finite Potential Well
@author: C1518116
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import constants
plt.close("all")

#Define constants
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
V_well = np.zeros(N)
V_well[700:750] = -6e-19

#Define beta  
beta_well = 2.0 + (2*(dx**2)*m / hbar**2)*V_well

#Define matrix M
M_well = np.diagflat(beta_well) - np.eye(N, k=1) - np.eye(N, k=-1)
M_well = M_well * hbar**2 / (2.0 * m * dx**2)

#Solve for eigenvalues/eigenvectors
eigvals_well, eigvect_well = linalg.eigh(M_well, UPLO='L')

#Calc time needed
T_well = np.linspace(0, 1e-13, N)

"""
Negative V Potential

|             |
|             |
|             |
|             |
|   ->        |
|___/\___  ___|
         |_|
"""

#Define initial wave function
Phi_ini_well = np.roll((1e9/np.sqrt(2.0*np.pi))*np.exp(-(L**2) / (2*1e-9**2))\
                       * np.exp(1j * -k0 * L) * 5.954e-5, -200)

#Calculate Cn for 100 eigenstates
n_well = 100
Cn_well = np.zeros(n_well) + 0j
for i in range(np.size(Cn_well)):
    Cn_well[i] = np.sum(eigvect_well[:,i] * Phi_ini_well)

#Calc Phi(t)
Phi_tw = np.zeros((N, N)) + 0j
for j in range(N):
    for i in range(np.size(Cn_well)):
        Phi_tw[:, j] += Cn_well[i] * eigvect_well[:,i] * np.exp(1j \
              * eigvals_well[i] * T_well[j] / hbar)


#Plot initial wave function with calculated initial wave function and compare
plt.figure("Negative well")
plt.plot(L*1e9, np.abs(Phi_ini_well), label=r'$\Psi_i$')
plt.plot(L*1e9, np.abs(Phi_tw[:, 0]), '--', label=r'$\Psi$')
plt.xlabel(r'$x$')
plt.ylabel('$\Psi$')
plt.xlim(-11, 11)
plt.legend(loc="best")
plt.show()


"""ANIMATION"""
#Animate the probability density of the wave functions over time.

# First set up the figure, the axis, and the plot element we want to animate
figWell = plt.figure("Animation - Well")
axWell = plt.axes(xlim=(-12, 12), ylim=(-0.4e9, 0.7e9))
lineModSqWell, = axWell.plot([], [], label=r'$|\Psi$(t)$|^2$', lw=2)
time_text = axWell.text(0.1, 0.93, '', transform=axWell.transAxes)
lineInfWellWell, = axWell.plot([], [], 'k', lw=2)

# initialization function: plot the background of each frame
def initWell():
    lineModSqWell.set_data([], [])
    lineInfWellWell.set_data([], [])
    time_text.set_text('')
    
    return  lineModSqWell, lineInfWellWell, time_text,

# animation function.  This is called sequentially
def animateWell(i):
    global L, Phi_tw, V_well, T_well
    
    lineModSqWell.set_data(L*1e9, (Phi_tw[:,i] * Phi_tw[:,i].conjugate()).real)
    lineInfWellWell.set_data(np.array([-1e-8, -1e-8, L[700], L[700], L[750], L[750], 1e-8, 1e-8])*1e9, \
                             [1e9, 0, 0, V_well[700]*1e8/eV, V_well[749]*1e8/eV, 0, 0, 1e9])
    time_text.set_text(r'$t$ = %#04.1f fs' %(T_well[i] / 1e-15))
    
    return lineModSqWell, lineInfWellWell, time_text,

# call the animator.  blit=True means only re-draw the parts that have changed.
animSquareWell = animation.FuncAnimation(figWell, animateWell, init_func=initWell, 
                                        frames=500, interval=10, blit=True)

plt.xlabel('L (nm)')
plt.ylabel(r'$|\Psi$(t)$|^2$')
plt.legend(loc=[0.73, 0.9], frameon=False)

#Save animation - requires ffmpeg install!
#animSquareWell.save('TDSE Negative Barrier.mp4', fps=30, dpi=200)

plt.show()

#Plot snapshots of animation
plt.figure()
ax1 = plt.subplot(311)
plt.plot(L/1e-9, (Phi_tw[:,0] * Phi_tw[:,0].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, L[700], L[700], L[750], L[750], 1e-8, 1e-8])*1e9,\
         [1e9, 0, 0, V_well[700]*1e8/eV, V_well[749]*1e8/eV, 0, 0, 1e9], 'k')
plt.text(-9.5, 5e8, 't = %#04.2ffs'%(T_well[0]*1e15))
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylim(-4.1e8, 7e8)

ax2 = plt.subplot(312, sharex=ax1)
plt.plot(L/1e-9, (Phi_tw[:,70] * Phi_tw[:,70].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, L[700], L[700], L[750], L[750], 1e-8, 1e-8])*1e9,\
         [1e9, 0, 0, V_well[700]*1e8/eV, V_well[749]*1e8/eV, 0, 0, 1e9], 'k')
plt.text(-9.5, 5e8, 't = %#04.2ffs'%(T_well[70]*1e15))
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel(r'$|\Psi$(t)$|^2$')
plt.ylim(-4.1e8, 7e8)

ax3 = plt.subplot(313, sharex=ax1)
plt.plot(L/1e-9, (Phi_tw[:,110] * Phi_tw[:,110].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, L[700], L[700], L[750], L[750], 1e-8, 1e-8])*1e9,\
         [1e9, 0, 0, V_well[700]*1e8/eV, V_well[749]*1e8/eV, 0, 0, 1e9], 'k')
plt.text(-9.5, 5e8, 't = %#04.2ffs'%(T_well[110]*1e15))
plt.ylim(-4.1e8, 7e8)
plt.xlabel('L (nm)')

plt.subplots_adjust(hspace=0.25)

#plt.savefig('TDSE Negative Barrier.png', dpi=200)
plt.show()
