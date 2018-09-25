# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:01:30 2018
File Name:TDSE Infinite Square Well - Offset
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
V_square = np.zeros(N)

#Define beta  
beta_square = 2.0 + (2*(dx**2)*m / hbar**2)*V_square

#Define matrix M
M_square = np.diagflat(beta_square) - np.eye(N, k=1) - np.eye(N, k=-1)
M_square = M_square * hbar**2 / (2.0 * m * dx**2)

#Solve for eigenvalues/eigenvectors
eigvals_square, eigvect_square = linalg.eigh(M_square, UPLO='L')

#Calc time needed
Tmax_square = 1e-13
T_square = np.linspace(0, Tmax_square, N)


"""
OFFSET SQUARE WELL

|              |
|              |
|              |
|___/\_________|

"""



#Calc initial Phi(t=0)
Phi_ini_square = np.roll((1e9/np.sqrt(2.0*np.pi))*np.exp(-(L**2) / (2*1e-9**2))\
                         * 5.953913e-5, -200)

#Calculate Cn's for 25 eigenstates.
n_square = 25
Cn_square = np.zeros(n_square)
for i in range(np.size(Cn_square)):
    Cn_square[i] = np.sum(eigvect_square[:,i] * Phi_ini_square)
    
    

#Calc Phi(t)
Phi_ts = np.zeros((N, N)) + 0j
for j in range(N):
    for i in range(np.size(Cn_square)):
        Phi_ts[:, j] += Cn_square[i] * eigvect_square[:,i] * \
            np.exp(1j * eigvals_square[i] * T_square[j] / hbar)

#Plot comparison between the initial wave function and the initial wave function
#reconstructed from the eigenstates.
plt.figure("Square well offset")
plt.plot(L*1e9, Phi_ini_square, label=r'$\Psi_i$')
plt.plot(L*1e9, Phi_ts[:, 0].real, '--', label=r'$\Psi$')
plt.xlabel(r'$x$')
plt.ylabel('$\Psi$')
plt.xlim(-11, 11)
plt.legend(loc="best")
plt.show()


"""ANIMATION"""
#Animate the probability density of the wave functions over time.

# First set up the figure, the axis, and the plot element we want to animate
figSquare = plt.figure("Animation - Re Im")
ax = plt.axes(xlim=(-12, 12), ylim=(-12e7, 6e8))
lineModSq, = ax.plot([], [], label=r'$|\Psi$(t)$|^2$', lw=2)
lineRe, = ax.plot([], [], label=r'$10^4\times\Re$($\Psi$(t))', lw=2)
lineIm, = ax.plot([], [], label=r'$10^4\times\Im$($\Psi$(t))', lw=2)
lineInfWell, = ax.plot([], [], 'k', lw=2)
time_text = ax.text(0.1, 0.93, '', transform=ax.transAxes)

# initialization function: plot the background of each frame
def initSquare():
    lineRe.set_data([], [])
    lineIm.set_data([], [])
    lineModSq.set_data([], [])
    lineInfWell.set_data([], [])
    time_text.set_text('')
    
    return lineRe, lineIm, lineModSq, lineInfWell, time_text, 

# animation function.  This is called sequentially
def animateSquare(i):
    global L, Phi_ts
    
    lineRe.set_data(L*1e9, Phi_ts[:,i].real*1e4)
    lineIm.set_data(L*1e9, Phi_ts[:,i].imag*1e4)
    lineModSq.set_data(L*1e9, (Phi_ts[:,i] * Phi_ts[:,i].conjugate()).real) # * 1e17 to put on same scale
    lineInfWell.set_data([-10, -10, 10, 10], [6e8, 0, 0, 6e8])
    time_text.set_text(r'$t$ = %#04.1f fs' %(T_square[i] / 1e-15))
    
    return lineRe, lineIm, lineModSq, lineInfWell, time_text, 

# call the animator.  blit=True means only re-draw the parts that have changed.
animSquare = animation.FuncAnimation(figSquare, animateSquare, init_func=initSquare,
                               frames=950, interval=10, blit=True)

plt.xlabel('L (nm)')
plt.ylabel(r'$\Psi$(t)')
plt.legend(loc=[0.63, 0.76], frameon=False)


#Save animation - requires ffmpeg install!
#animSquare.save('TDSE Square Well - Offset.mp4', fps=60, dpi=200)

plt.show()


#Plot animation snapshots
plt.figure()

ax1 = plt.subplot(311)
plt.plot(L/1e-9, (Phi_ts[:,0] * Phi_ts[:,0].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([6e8, 0, 0, 6e8]), 'k-')
plt.text(-9.5, 4.75e8, 't = %#05.2ffs'%(T_square[0]*1e15))
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylim(-5e7, 6e8)

ax2 = plt.subplot(312, sharex=ax1)
plt.plot(L/1e-9, (Phi_ts[:,300] * Phi_ts[:,300].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([6e8, 0, 0, 6e8]), 'k-')
plt.text(-9.5, 4.75e8, 't = %#05.2ffs'%(T_square[300]*1e15))
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel(r'$|\Psi$(t)$|^2$')
plt.ylim(-5e7, 6e8)

ax3 = plt.subplot(313, sharex=ax1)
plt.plot(L/1e-9, (Phi_ts[:,600] * Phi_ts[:,600].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([6e8, 0, 0, 6e8]), 'k-')
plt.text(-9.5, 4.75e8, 't = %#05.2ffs'%(T_square[600]*1e15))
plt.ylim(-5e7, 6e8)
plt.xlabel('L (nm)')

plt.subplots_adjust(hspace=0.25)

#plt.savefig('TDSE Inf Well.png', dpi=200)
plt.show()
