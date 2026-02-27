#!/usr/bin/env python
#
# Lister Staveley-Smith

# Purpose:
# Plots Tsys from a CryoPAF json calibration table (different IFs and Stokes modes labelled as extra beams)

# Useage:
# python paf-tsys.py paf_250223_233303_ON_cal.json

# Libraries
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import re
import os

# Constants
boltzmann = 1.381e-23
area = np.pi*(64.0/2.0)**2

# json filenames
files = sys.argv[1:]
n = len(files)

if n == 0:
    exit("ERROR: no file specified")
if re.search(r"\*", files[0]):
    exit("ERROR: no json file found")

# Parset name
parset = "closepack72.parset"
#parset = "3x3.parset"

#shortparset = parset.split("/")[-1]
shortparset = parset

# Parset search paths
path = ["/Users/00060351/Drive/PROJECTS/CRYOPAF/COMMISSIONING/", "/Users/lss/Drive/PROJECTS/CRYOPAF/COMMISSIONING/"]

# Parset defaults
offsets = -9                # Az, El offset vectors
pitch = -9                  # Beam spacing 

# Find, open and read parset
success = False
for i in range(len(path)):
    if os.path.isfile(path[i]+parset):
        ps = open(path[i]+parset)
        print("parset found: {}".format(path[i]+parset))
        for line in ps:
            if (line[:7] == "offsets") or ("pitch" in line):
                exec(line)
        success = True
        break

if not success:
    exit("ERROR: parset file not found or not able to read - "+parset)

# Check parset parameters present
if (offsets == -9) or (pitch == -9):
    exit("ERROR: parset parameters missing")

# Apply pitch
xoff = pitch*np.array(offsets)

# Apply direction error
xoff = -xoff

fsys = []
rad = []
bm = []
fx = []
fq = []

for ifile in range(n):
    fcal = open(files[ifile], 'r')
    calparams = json.load(fcal)

    BEAMS = list(calparams)

    if len(BEAMS) !=  xoff.shape[0]:
        exit("ERROR: different numbers of beams in parset and json files")

    for BEAM in BEAMS:
        ibeam = int(BEAM[-2:])
        NPOLS = list(calparams[BEAM])
        for NPOL in NPOLS:                              # Use dual polarisation only
            if NPOL == "npol_2":
                SBS = list(calparams[BEAM][NPOL])
                for SB in SBS:
                    if SB == "band_SB1":                # Use band_SB1 only
                        bm.append(ibeam)
                        fsys.append(calparams[BEAM][NPOL][SB]['tsys'])
                        rad.append(np.sqrt((xoff[ibeam,0]**2 + xoff[ibeam,1]**2)))
                        fq.append(calparams[BEAM][NPOL][SB]['freq'])
                        if 'fluxunit' in calparams[BEAM][NPOL][SB]:
                            fx.append(calparams[BEAM][NPOL][SB]['fluxunit'])
                        else:
                            fx.append("Jy")

BM = np.array(bm)
FSYS = np.array(fsys)
TSYS = FSYS*1e-26*area/(2.0*boltzmann)
RAD = np.array(rad)
FX = np.array(fx)

if "Jy" in FX:
    tsys_av = np.average(TSYS[(RAD<0.3) & (FX=="Jy")])
    print("Average Tsys/dish efficiency (<0.3 deg) = {:.2f}".format(tsys_av))
if "K" in FX:
    tsys_av = np.average(FSYS[(RAD<0.3) & (FX=="K")])
    print("Average Tsys/beam efficiency (<0.3 deg) = {:.2f}".format(tsys_av))

# Make a plot
plt.rc('font',family='serif',size=18)
plt.rcParams['axes.linewidth'] = 2
plt.figure('Tsys measurement number') 
plt.xlabel("Measurement number")
plt.ylabel(r"$T_{\rm sys}$")
plt.plot(np.arange(FSYS.shape[0]), FSYS[:,0], marker="+",color="red",label="X")
plt.plot(np.arange(FSYS.shape[0]), FSYS[:,1], marker="+",color="blue",label="Y")
plt.ylim(15,50)
#plt.title('CryoPAF system temperatures - {}'.format(shortparset))
plt.legend()
plt.tight_layout()
plt.show()

title = {"Jy": "Continuum source measurements", "K": "Galactic HI measurements"}

for unit in ["Jy", "K"]:
    if unit in FX:
        if unit == "Jy":
            ylab = r"$T_{\rm sys}$"
        else:
            ylab = r"$T_{\rm sys}/\eta_{\rm mb}$"
            
        plt.figure('Tsys_v_beam') 
        plt.xlabel("Beam")
        plt.ylabel(ylab+" ({})".format(unit))
        plt.scatter(BM[FX==unit], FSYS[:,0][FX==unit], marker="+",color="red",label="X")
        plt.scatter(BM[FX==unit], FSYS[:,1][FX==unit], marker="+",color="blue",label="Y")
        plt.ylim(15,50)
#        plt.title('{} - {}'.format(title[unit], shortparset))
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure('Tsys_v_radius') 
        plt.xlabel("Distance from optical axis (deg)")
        plt.ylabel(ylab+" ({})".format(unit))
        plt.scatter(RAD[FX==unit], FSYS[:,0][FX==unit], marker="+",color="red",label="X")
        plt.scatter(RAD[FX==unit], FSYS[:,1][FX==unit], marker="+",color="blue",label="Y")
        plt.ylim(15,50)
#        plt.title('{} - {}'.format(title[unit], shortparset))
        plt.legend()
        plt.tight_layout()
        plt.show()

        if unit == "Jy":
            plt.figure('Tsysovereta_v_radius') 
            plt.xlabel("Distance from optical axis (deg)")
            plt.ylabel(r"$T_{\rm sys}/\eta_{\rm d}$"+" (K)")
            plt.scatter(RAD[FX==unit], TSYS[:,0][FX==unit], marker="+",color="red",label="X")
            plt.scatter(RAD[FX==unit], TSYS[:,1][FX==unit], marker="+",color="blue",label="Y")
            plt.ylim(15,50)
#            plt.title('{} - {}'.format(title[unit], shortparset))
            plt.legend()
            plt.tight_layout()
            plt.show()

if ("Jy" in FX) and ("K" in FX):
    BAD = np.sqrt((xoff[:,0]**2 + xoff[:,1]**2))
    KJY = np.zeros((len(BEAMS),2))
    KPK = np.zeros((len(BEAMS),2))
    for i in BM:
        for j in range(2):
            KJY[i,j] = np.average(FSYS[:,j][np.logical_and(BM==i, FX=="K")])/np.average(FSYS[:,j][np.logical_and(BM==i, FX=="Jy")])
            KPK[i,j] = np.average(FSYS[:,j][np.logical_and(BM==i, FX=="K")])/np.average(TSYS[:,j][np.logical_and(BM==i, FX=="Jy")])
            
    plt.figure('KelvinperJy_v_radius') 
    plt.xlabel("Distance from optical axis (deg)")
    plt.ylabel(r"$T_{\rm B}/S$ (K Jy$^{-1}$)")
    plt.scatter(BAD, KJY[:,0], marker="+",color="red",label="X")
    plt.scatter(BAD, KJY[:,1], marker="+",color="blue",label="Y")
#    plt.title('Kelvin per Jy'.format(shortparset))
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure('KperK_v_radius') 
    plt.xlabel("Distance from optical axis (deg)")
    plt.ylabel(r"Efficiency ratio, $\eta_{\rm d}/\eta_{\rm mb}$")
    plt.scatter(BAD, KPK[:,0], marker="+",color="red",label="X")
    plt.scatter(BAD, KPK[:,1], marker="+",color="blue",label="Y")
#    plt.title('Dish over main beam efficiency'.format(shortparset))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
exit()

