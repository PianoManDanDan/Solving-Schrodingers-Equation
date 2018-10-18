# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:06:19 2018
File Name:TDSE Infinite Square Well - Velocity
@author: Daniel Martin
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
OFFSET SQUARE WITH VELOCITY

|              |
|              |
|   ->         |
|___/\_________|

"""

#Define initial wave function
Phi_ini_vel = np.roll((1e9/np.sqrt(2.0*np.pi))*np.exp(-(L**2) / (2*1e-9**2)) \
                      * np.exp(1j * -k0 * L) * 5.954e-5, -200)

#Calculate Cn for 100 eigenstates
n_square_vel = 100 
Cn_square_vel = np.zeros(n_square_vel) + 0j
for i in range(np.size(Cn_square_vel)):
    Cn_square_vel[i] = np.sum(eigvect_square[:,i] * Phi_ini_vel)

#Calc Phi(t)
Phi_ts_vel = np.zeros((N, N)) + 0j
for j in range(N):
    for i in range(np.size(Cn_square_vel)):
        Phi_ts_vel[:, j] += Cn_square_vel[i] * eigvect_square[:,i] \
            * np.exp(1j * eigvals_square[i] * T_square[j] / hbar)


#Plot real and imaginary parts for the inital and calculated initial wave 
#functions and compare them to each other.
plt.figure("Square well velocity")
ax1 = plt.subplot(211)
plt.plot(L*1e9, (Phi_ini_vel.real), label=r'$\Re(\Psi_i)$')
plt.plot(L*1e9, (Phi_ts_vel[:, 0].real), '--', label=r'$\Re(\Psi(t=0))$')
plt.plot([-10, -10, 10, 10], [25e3, 0, 0, 25e3], 'k')
plt.ylabel('$\Re(\Psi)$')
plt.xlim(-11, 11)
plt.ylim(-25e3, 25e3)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend(loc="lower right")

ax2 = plt.subplot(212, sharex = ax1)
plt.plot(L*1e9, (Phi_ini_vel.real), label=r'$\Im(\Psi_i)$')
plt.plot(L*1e9, (Phi_ts_vel[:, 0].real), '--', label=r'$\Im(\Psi(t=0))$')
plt.plot([-10, -10, 10, 10], [25e3, 0, 0, 25e3], 'k')
plt.xlabel('L (nm)')
plt.ylabel('$\Im(\Psi)$')
plt.xlim(-11, 11)
plt.ylim(-25e3, 25e3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend(loc="lower right")

#plt.savefig('TDSE Square Well - Velocity initial wave.png', dpi=200)
plt.show()



"""ANIMATION""" 
#Animate the probability density of the wave functions over time.

# First set up the figure, the axis, and the plot element we want to animate
figSquareVel = plt.figure("Animation - Velocity")
axVel = plt.axes(xlim=(-12, 12), ylim=(-0.5e8, 12e8))
lineModSqVel, = axVel.plot([], [], label=r'$|\Psi$(t)$|^2$', lw=2)
lineInfWellVel, = axVel.plot([], [], 'k', lw=2)
time_text = axVel.text(0.1, 0.93, '', transform=axVel.transAxes)

# initialization function: plot the background of each frame
def initSquareVel():
    lineModSqVel.set_data([], [])
    lineInfWellVel.set_data([], [])
    time_text.set_text('')
    return lineModSqVel, lineInfWellVel,time_text,

# animation function.  This is called sequentially
def animateSquareVel(i):
    global L, Phi_ts_vel
    
    lineModSqVel.set_data(L*1e9, (Phi_ts_vel[:,i] * Phi_ts_vel[:,i].conjugate()).real)
    lineInfWellVel.set_data([-10, -10, 10, 10], [12e8, 0, 0, 12e8])
    time_text.set_text(r'$t$ = %#04.1f fs' %(T_square[i] / 1e-15))
    
    return lineModSqVel, lineInfWellVel,time_text,

# call the animator.  blit=True means only re-draw the parts that have changed.
animSquareVel = animation.FuncAnimation(figSquareVel, animateSquareVel, 
                init_func=initSquareVel, frames=750, interval=10, blit=True)


plt.xlabel('L (nm)')
plt.ylabel(r'$|\Psi$(t)$|^2$')
plt.legend(loc=[0.73, 0.9], frameon=False)

#Save animation - requires ffmpeg install!
#animSquareVel.save('TDSE Infinite Well Well - Velocity.mp4', fps=30, dpi=200)

plt.show()


#Plot animation snapshots
fig = plt.figure(figsize=(6, 8))

ax1 = plt.subplot(611)
plt.plot(L/1e-9, (Phi_ts_vel[:,0] * Phi_ts_vel[:,0].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([1.3e9, 0, 0, 1.3e9]), 'k-')
plt.text(-9.5, 0.9e9, 't = %#05.2ffs'%(T_square[0]*1e15))
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylim(-0.1e9, 1.3e9)

ax2 = plt.subplot(612, sharex=ax1)
plt.plot(L/1e-9, (Phi_ts_vel[:,67] * Phi_ts_vel[:,67].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([1.3e9, 0, 0, 1.3e9]), 'k-')
plt.text(-9.5, 0.9e9, 't = %#05.2ffs'%(T_square[67]*1e15))
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylim(-0.1e9, 1.3e9)

ax3 = plt.subplot(613, sharex=ax1)
plt.plot(L/1e-9, (Phi_ts_vel[:,128] * Phi_ts_vel[:,128].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([1.3e9, 0, 0, 1.3e9]), 'k-')
plt.text(-9.5, 0.9e9, 't = %#05.2ffs'%(T_square[128]*1e15))
plt.setp(ax3.get_xticklabels(), visible=False)
plt.ylabel(r'$|\Psi$(t)$|^2$', position=(-15, -0.11))
plt.ylim(-0.1e9, 1.3e9)

ax4 = plt.subplot(614, sharey=ax1)
plt.plot(L/1e-9, (Phi_ts_vel[:,180] * Phi_ts_vel[:,180].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([1.3e9, 0, 0, 1.3e9]), 'k-')
plt.text(-9.5, 0.9e9, 't = %#05.2ffs'%(T_square[180]*1e15))
plt.setp(ax4.get_xticklabels(), visible=False)
plt.ylim(-0.1e9, 1.3e9)

ax5 = plt.subplot(615, sharex=ax1)
plt.plot(L/1e-9, (Phi_ts_vel[:,240] * Phi_ts_vel[:,240].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([1.3e9, 0, 0, 1.3e9]), 'k-')
plt.text(-9.5, 0.9e9, 't = %#05.2ffs'%(T_square[240]*1e15))
plt.setp(ax5.get_xticklabels(), visible=False)
plt.ylim(-0.1e9, 1.3e9)

ax6 = plt.subplot(616, sharex=ax1)
plt.plot(L/1e-9, (Phi_ts_vel[:,300] * Phi_ts_vel[:,300].conjugate()).real)
plt.plot(np.array([-1e-8, -1e-8, 1e-8, 1e-8])/1e-9, np.array([1.3e9, 0, 0, 1.3e9]), 'k-')
plt.text(-9.5, 0.9e9, 't = %#05.2ffs'%(T_square[300]*1e15))
plt.ylim(-0.1e9, 1.3e9)
plt.xlabel('L (nm)')

plt.subplots_adjust(hspace=0.35)

#plt.savefig('TDSE Infinite Square Well - Velocity.png', dpi=200)
plt.show()