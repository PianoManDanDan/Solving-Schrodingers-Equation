# -*- coding: utf-8 -*-
"""
Created on Fri May 11 04:50:58 2018
File Name: TDSE Infinite Square Well - Stationary
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

#Define number of points for matrix
N = 1000

#Define step size dx
Lmax = 10.0e-9 #Max x
L = np.linspace(-Lmax, Lmax, N)
dx = L[1] - L[0]

#Define time step dt
Tmax = 1e-13 #seconds
T = np.linspace(0, Tmax, N)

#Define 1D array for V
V_square = np.zeros(N)

#Define beta  
beta_square = 2.0 + (2*(dx**2)*m / hbar**2)*V_square

#Define matrix M
M_square = np.diagflat(beta_square) - np.eye(N, k=1) - np.eye(N, k=-1)
M_square = M_square * hbar**2 / (2.0 * m * dx**2)

#Solve for eigenvalues/eigenvectors
eigvals_square, eigvect_square = linalg.eigh(M_square, UPLO='L')


""" calculate TDSE as sum of eigenstates for square well """

#Calc initial Phi(t=0)
Phi_ini = (1e9/np.sqrt(2.0*np.pi))*np.exp(-(L**2) / (2*1e-9**2)) * 5.953913e-5


#Calc Cn's for first 20 eigenstates
Cn_square = np.zeros(20)
for i in range(np.size(Cn_square)):
    Cn_square[i] = np.sum(eigvect_square[:,i] * Phi_ini)
    
    
#Calc Phi(t)
Phi_t = np.zeros((N, N)) + 0j
for j in range(N):
    for i in range(np.size(Cn_square)):
        Phi_t[:, j] += Cn_square[i] * eigvect_square[:,i] * \
            np.exp(1j * eigvals_square[i] * T[j] / hbar)


#Plot the initial wave function calculated from the eigenstates and compare with
#the true initial wave function.
plt.figure("Phi_i and Phi_t")
plt.plot(L*1e9, Phi_ini, label=r'True $\Psi_i$')
plt.plot(L*1e9, Phi_t[:,0].real, '--', label=r'Calculated $\Psi$(t=0)')
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])*1e9, [25e3, 0, 0, 25e3], 'k')
plt.xlabel('L (nm)')
plt.ylabel('$\Psi$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlim(-11, 11)
plt.ylim(-1e3, 25e3)
plt.legend(loc="best", framealpha=1)
plt.show()

#plt.savefig('Square Well Phi ini comparison.png', dpi = 200)



plt.figure("Mod Square")
for i in range(0, 501, 125):
    plt.plot(L*1e9, (Phi_t[:, i] * Phi_t[:, i].conjugate()).real, \
             label=r'$|\Psi$ (t = %#05.2ffs)$|^2$' %(T[i]/1e-15))

plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])*1e9, [5.9e8, 0, 0, 5.9e8], 'k-')

plt.xlabel('L (nm)')
plt.ylabel(r'|$\Psi$(t)$|^2$')
plt.xlim(-11, 11)
plt.ylim(-0.1e8, 5.9e8)
plt.legend(loc=[0.6, 0.64], frameon=False) #This appears slightly high in the frame, but scales 
plt.show()                                 #to the correct position when saving the image.
#plt.savefig('TDSE Mod Square Well - Central.png', dpi = 200)



"""*****ANIMATION*****"""
#Animate the probability density of the wave functions over time.

# First set up the figure, the axis, and the plot element we want to animate
figSquare = plt.figure("Animation - Square Well")
ax = plt.axes(xlim=(-12, 12), ylim=(-1e8, 6e8))
lineModSq, = ax.plot([], [], label=r'$|\Psi$(t)$|^2$', lw=2)
lineRe, = ax.plot([], [], label=r'$10^4\times\Re$($\Psi$(t))', lw=2)
lineIm, = ax.plot([], [], label=r'$10^4\times\Im$($\Psi$(t))', lw=2)
lineInfWell, = ax.plot([], [], 'k', lw=2)
time_textSq = ax.text(0.1, 0.93, '', transform=ax.transAxes)


# initialization function: plot the background of each frame
def initSquare():
    lineRe.set_data([], [])
    lineIm.set_data([], [])
    lineModSq.set_data([], [])
    lineInfWell.set_data([], [])
    time_textSq.set_text('')
    
    return lineRe, lineIm, lineModSq, lineInfWell, time_textSq, 

# animation function.  This is called sequentially
def animateSquare(i):
    global L, Phi_t

    lineModSq.set_data(L*1e9, (Phi_t[:,i] * Phi_t[:,i].conjugate()).real)
    lineRe.set_data(L*1e9, Phi_t[:,i].real*1e4)
    lineIm.set_data(L*1e9, Phi_t[:,i].imag*1e4)
    lineInfWell.set_data(np.array([-1e-8, -1e-8, 1e-8, 1e-8])*1e9, [6e8, 0, 0, 6e8])
    time_textSq.set_text(r'$t$ = %#04.1f fs' %(T[i] / 1e-15))
    
    return lineRe, lineIm, lineModSq, lineInfWell, time_textSq,

# call the animator.  blit=True means only re-draw the parts that have changed.
animSquare = animation.FuncAnimation(figSquare, animateSquare, init_func=initSquare,
                               frames=750, interval=10, blit=True)

plt.xlabel('L(nm)')
plt.ylabel(r'$\Psi$(t)')
plt.legend(loc=[0.63, 0.76], frameon=False)

#Save animation - requires ffmpeg install!
#animSquare.save('TDSE Square Well - Central.mp4', fps=60, dpi=200)

plt.show()
