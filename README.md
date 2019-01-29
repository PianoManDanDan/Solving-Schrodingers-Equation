# Solving Schrodinger's equation using python routines
## 3rd Year Physics Research project by Daniel Martin

These python scripts calculate the time-dependent and time-independent solutions to Schrodinger's
equation for various potentials. 

All of the scripts use the numpy, matplotlib and scipy libraries. Version details are below:
- Python version: 3.6.4 (available [here](https://www.python.org/ "Python home page"))
- sciPy version: 1.0.0 (available [here](https://www.scipy.org/ "SciPy home page"))
- numpy version: 1.14.0 (available [here](http://www.numpy.org/ "Numpy home page"))
- matplotlib version: 2.1.2 (available [here](https://matplotlib.org/ "Matplotlib home page"))

The time-dependent scripts also require a video encoder such as ```ffmpeg``` to be installed in order
to save animations. The ```ffmpeg``` encoder can be found [here](https://www.ffmpeg.org/ "ffmpeg homepage").

This script was written in the Spyder IDE, version 3.2.6 (available [here](https://anaconda.org/anaconda/spyder) 
as part of the anaconda package).

A brief outline of what each script does is outlined below:

#### ```TISE Solver.py```

This script solves the time-independent energy eigenvalues and corresponding eigenfunctions for an electron
in an infinite square well, a parabolic well and a linear well.

#### ```TDSE Infinite Square Well - Stationary.py```

This script solves the time-dependent wave function for a stationary particle in a 1D infinite square well
centered about the origin.

#### ```TDSE Infinite Square Well - Offset.py```

This script solves the time-dependent wave function for a stationary particle in a 1D infinite square well
centered about -4nm.

#### ```TDSE Parabolic Well.py```

This script solves the time-dependent wave function for a stationary particle in a 1D parabolic well
centered about -4nm.

#### ```TDSE Infinite Square Well - Velocity.py```

This script solves the time-dependent wave function for a moving particle in a 1D infinite square well.

#### ```TDSE Potential Barrier.py```

This script solves the time-dependent wave function for a particle moving towards a finite, positive potential barrier.

#### ```TDSE Potential Well.py```

This script solves the time-dependent wave function for a particle moving towards a finite, negative potential well.
