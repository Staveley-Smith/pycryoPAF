#!/usr/bin/env python

# python2 print compatibility
from __future__ import print_function

import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt
from scipy import integrate
import sys

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2

files = sys.argv[1:]
n = len(files)
if n < 1:
    exit("ERROR: need > 1 or more spectra")
    
# Initialise
lines = [""]* n
fval = [[]]*n
flux = [[]]*n
ichan = [0]*n

# Read spectra
for i in range(n):
    with open(files[i], 'r') as f:
        lines[i] = f.readlines()
    f.close()

# Push data into arrays 
for i in range(n):
    ich = 0
    for line in lines[i]:
        if line[0] != '#':
            col = line.split()
            if ich == 0:
                fval[i] = [float(col[0])]
                flux[i] = [float(col[1])]
            else:
                fval[i].append(float(col[0]))
                flux[i].append(float(col[1]))
            ich += 1
    ichan[i] = ich
    print("{}: {} channels read".format(files[i], ichan[i]))

# Plot
plt.rc('font',family='serif',size=18)
plt.rcParams['axes.linewidth'] = 2
plt.figure('spectrum')
plt.rc('font',family='serif')
for i in range(n):
    qfval = np.array(fval[i])
    qflux = np.array(flux[i])
    # ---- NGC6744 only
    qfval = 1420.4058/(1+qfval/299792.46)
    if i==1:
        qflux *= 0.000962 # (18.2 arcmin beam; 0.01 deg pixel)
        int1 = np.sum(qflux[(qfval>1415.5) & (qfval<1417.5)])*np.abs(qfval[0]-qfval[-1])/ichan[i]
        print(int1)
    else:
        # the HIPASS data is from webplot, so irregularly spaced!
        int0 = np.abs(integrate.trapezoid(qflux[(qfval>1415.5) & (qfval<1417.5)], qfval[(qfval>1415.5) & (qfval<1417.5)]))
        print(int0)
    plt.xlim([1415.1, 1417.9])
    # ----
    plt.plot(qfval, qflux, label=files[i])
plt.xlabel(r'Frequency (MHz)', size=30)
plt.ylabel(r'Flux density (Jy)', size=30)
plt.legend(loc=1, ncol=1, prop={'size': 18})
plt.tight_layout()
plt.show()
plt.close()

exit()

