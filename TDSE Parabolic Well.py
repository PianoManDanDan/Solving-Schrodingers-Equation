# -*- coding: utf-8 -*-
"""
Created on Fri May 11 05:01:14 2018
File Name: TDSE Parabolic Well
@author: C1518116
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import constants
#import timeit
plt.close("all")


hbar = constants.hbar
m = constants.m_e
eV = constants.elementary_charge
k = 1e-3
omega = np.sqrt(k / m)

#Define number of points for matrix
N = 1000

#Define step size dx
Lmax = 10.0e-9 #Max x
L = np.linspace(-Lmax, Lmax, N)
dx = L[1] - L[0]

#Define 1D array for V
V_parabola = 0.5 * k * (L)**2

#Define beta  
beta_parabola = 2.0 + (2*(dx**2)*m / hbar**2)*V_parabola

#Define matrix M
M_parabola = np.diagflat(beta_parabola) - np.eye(N, k=1) - np.eye(N, k=-1)
M_parabola = M_parabola * hbar**2 / (2.0 * m * dx**2)

#Solve for eigenvalues/eigenvectors
eigvals_parabola, eigvect_parabola = linalg.eigh(M_parabola, UPLO='L')

#Define time step dt
Tmax_parabola = hbar * 2 * np.pi / eigvals_parabola[0] #seconds
T_parabola = np.linspace(0, Tmax_parabola, N)


"""PARABOLA"""

Phi_ini_parabola = np.roll(eigvect_parabola[:, 0], -200) * 2.235e5

#Calc Cn's for first 150 eigenstates
n_parabola = 20
Cn_parabola = np.zeros(n_parabola)
for i in range(np.size(Cn_parabola)):
    Cn_parabola[i] = np.sum(eigvect_parabola[:,i] * Phi_ini_parabola)
    
    
#Calc Phi(t)
Phi_tp = np.zeros((N, N)) + 0j
for j in range(N):
    for i in range(np.size(Cn_parabola)):
        Phi_tp[:, j] += Cn_parabola[i] * eigvect_parabola[:,i] * \
            np.exp(1j * eigvals_parabola[i] * T_parabola[j] / hbar)


#Plot initial wave function calculated using eigenvalues and compare with true
# inital wave function
            
plt.figure("Phi_i and Phi_t - Parabola")
plt.plot(L, -Phi_ini_parabola, label=r'$\Psi_i$')
plt.plot(L, -Phi_tp[:,0].real, '--', label=r'$\Psi$ (t=0)')
plt.title(r'$\Psi_i$ and $\Psi$(t=0)')
plt.xlabel(r'$x$')
plt.ylabel('$\Psi$')
plt.xlim(-1e-8, 1e-8)
plt.legend(loc="best")
plt.show()



"""ANIMATION"""
#Animate the probability density of the wave functions over time.

# First set up the figure, the axis, and the plot element we want to animate
figParab = plt.figure("Animation - Parabola")
ax = plt.axes(xlim=(-11, 11), ylim=(-0.1e8, 3.38e8))
line, = ax.plot([], [], label=r'$|\Psi$(t)$|^2$', lw=2)
lineParabWell, = ax.plot([], [], 'k', lw=2)
time_text = ax.text(0.018, 0.93, '', transform=ax.transAxes)

# initialization function: plot the background of each frame
def initParab():
    line.set_data([], [])
    lineParabWell.set_data([], [])
    time_text.set_text('')
    
    return line, lineParabWell, time_text,

# animation function.  This is called sequentially
def animateParab(i):
    global L, Phi_tp, eV, T_parabola
    
    line.set_data(L*1e9, (Phi_tp[:,i] * np.conjugate(Phi_tp[:,i])).real)
    lineParabWell.set_data(L*1e9, 4e9*V_parabola/eV)
    time_text.set_text(r'$t$ = %#05.1f fs' %(T_parabola[i] / 1e-15))
                       
    return line, lineParabWell, time_text,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim2 = animation.FuncAnimation(figParab, animateParab, init_func=initParab,
                               frames=500, interval=10, blit=True)

plt.xlabel('L(m)')
plt.ylabel(r'$|\Psi$(t)$|^2$')
plt.legend(loc=[0.8, 0.9], frameon=False)

#Save animation - requires ffmpeg install!
#anim2.save('TDSE Parabolic Well.mp4', fps=60, dpi=200)

plt.show()


#Plot snapshots of animation

plt.figure()
ax1 = plt.subplot(411)
plt.plot(L*1e9, (Phi_tp[:,0] * np.conjugate(Phi_tp[:,0])).real)
plt.plot(L*1e9, 4e9*V_parabola / eV, 'k-')
plt.text(-9.7, 3e8, 't = %#05.2ffs'%(T_parabola[0]*1e15))
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylim(-0.2e8, 4e8)
plt.xlim(-10, 10)

ax2 = plt.subplot(412, sharex=ax1)
plt.plot(L*1e9, (Phi_tp[:,120] * np.conjugate(Phi_tp[:,120])).real)
plt.plot(L*1e9, 4e9*V_parabola / eV, 'k-')
plt.text(-9.7, 3e8, 't = %#05.2ffs'%(T_parabola[120]*1e15))
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel(r'$|\Psi$(t)$|^2$', position=(-10, -0.05))
plt.ylim(-0.2e8, 4e8)
plt.xlim(-10, 10)

ax3 = plt.subplot(413, sharex=ax1)
plt.plot(L*1e9, (Phi_tp[:,240] * np.conjugate(Phi_tp[:,240])).real)
plt.plot(L*1e9, 4e9*V_parabola / eV, 'k-')
plt.text(-9.7, 3e8, 't = %#05.2ffs'%(T_parabola[240]*1e15))
plt.setp(ax3.get_xticklabels(), visible=False)
plt.xlim(-10, 10)
plt.ylim(-0.2e8, 4e8)

ax4 = plt.subplot(414, sharex=ax1)
plt.plot(L*1e9, (Phi_tp[:,360] * np.conjugate(Phi_tp[:,360])).real)
plt.plot(L*1e9, 4e9*V_parabola / eV, 'k-')
plt.text(-9.7, 3e8, 't = %#05.2ffs'%(T_parabola[360]*1e15))
plt.setp(ax3.get_xticklabels(), visible=False)
plt.ylim(-0.2e8, 4e8)
plt.xlim(-10, 10)
plt.xlabel('L (nm)')

plt.subplots_adjust(hspace=0.35)

#plt.savefig('TDSE Parabolic Well.png', dpi = 200)
