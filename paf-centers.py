#!/usr/bin/env python
#
# Lister Staveley-Smith

# Libraries

import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.wcs import WCS
from astropy.wcs import utils
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import os

# Feed rotation relative to sky (feed angle - parallactic angle)
fang = -28 # deg


# Pointing parameters 
# - spacing is the spacing of the cryopaf pointings RA and Dec (deg)
# - delta is deprecated
# - delta_x is the extra spacing for the first, second etc rows (deg)
# - delta_y is the extra spacing for the first, second etc columns (deg)
# - roll is the rolling spacing in x & y
fieldname = 'LMC_feed_angle_-28deg'
coord = [80.0, -69.0]  # deg, deg
#spacing = [1.0, 1.0]    # deg, deg - best for closepack72 uniform sensitivity when not parallactifying
#delta = [spacing[0]/2.0, 0.07] # deg, deg
#spacing = [1.1, 1.0]    # deg, deg - best for closepack72 uniform sensitivity when parallactifying
#delta = [spacing[0]/2.0, 0.07] # deg, deg
spacing = [1.3, 1.25]    # deg, deg - best for hextile72 uniform sensitivity when parallactifying
#spacing = [1.1, 1.05]    # deg, deg - best for hextile72 uniform sensitivity any feed angle
#delta = [0.85, 0.288675] # deg, deg
#delta = [-0.3, 0.0] # deg, deg - best for hextile72 and fang=-28
delta_x = [0.0, -0.3, 0.3] # deg, deg - best for hextile72 and fang=-28
delta_y = [0.0, 0.0, 0.0] # deg, deg - best for hextile72 and fang=-28
roll = [0.0, 0.0]
fieldsize = [8.0,8.0]   # deg, deg
beam = 0.23             # deg
#fieldname = 'SMCz'
#coord = [15.0, -72.8]  # deg, deg
#spacing = [1.0, 1.0]    # deg, deg
#delta = [spacing[0]/2.0, 0.07] # deg, deg
#fieldsize = [5.0,4.0]       # deg, deg
#beam = 0.23             # deg
#fieldname = 'LMC'
#coord = [80.0, -69.0]  # deg, deg
#spacing = [1.2, 1.2]    # deg, deg
#fieldsize = [8.0,8.0]   # deg, deg
#beam = 0.23             # deg
#fieldname = 'SMC'
#coord = [15.0, -72.8]  # deg, deg
#spacing = [1.2, 1.2]    # deg, deg
#fieldsize = [5.0,4.0]       # deg, deg
#beam = 0.23             # deg

# Parset search paths
path = ["./", "/Users/00060351/Drive/PROJECTS/CRYOPAF/COMMISSIONING/", "/Users/lss/Drive/PROJECTS/CRYOPAF/COMMISSIONING/"]

# Parset name (for plots)
#parset = "/Users/lss/Google Drive/PROJECTS/CRYOPAF/COMMISSIONING/3x3.parset"
#parset = "closepack72.parset"
parset = "hextile72.parset"
#parset = "/Users/lss/Google Drive/PROJECTS/CRYOPAF/COMMISSIONING/grid72.parset"

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
            if (line[:7] == "offsets") or ("pitch" in line) or ("beamlabel" in line) or ("overwriterefbeam" in line):
                exec(line)
        success = True
        break

if not success:
    exit("ERROR: parset file not found or not able to read - "+parset)
    
# Open and read parset
#try:
#    with open(parset) as ps:
#        for line in ps:
#            if (line[:7] == "offsets") or ("pitch" in line):
#                exec(line)
#except:
#    exit("ERROR: parset file not found or not able to read - "+parset)

# Check parset parameters present
if (offsets == -9) or (pitch == -9):
    exit("ERROR: parset parameters missing")

# Put beam offsets into pixel grid
OFF_X = np.zeros(len(offsets))
OFF_Y = np.zeros(len(offsets))
for i in range(len(offsets)):
    OFF_X[i] = offsets[i][0]*pitch/spacing[0]  # offsets in pixel space
    OFF_Y[i] = offsets[i][1]*pitch/spacing[1]  # offsets in pixel space

# Lay out a pointing grid in pixel space
dim = (np.array(fieldsize)/np.array(spacing)).astype(int)+1
if dim[0] % 2 == 0: 
    dim[0] += 1
if dim[1] % 2 == 0: 
    dim[1] += 1
print("Grid will be {}x{} pointings, centre {} deg, spacing {} deg".format(dim[0], dim[1], coord, spacing))
refpix = (np.array(dim)/2.0).astype(int)

# Define a FITS header 
hduc = fits.PrimaryHDU()
hduc.header['NAXIS'] = 2
for i in range(len(dim)):
    hduc.header['CRVAL'+str(i+1)] = coord[i]
    hduc.header['CDELT'+str(i+1)] = spacing[i]
    hduc.header['CRPIX'+str(i+1)] = refpix[i]+1
hduc.header['CTYPE1'] = "RA---SIN"
hduc.header['CTYPE2'] = "DEC--SIN"
hduc.header['EQUINOX'] = 2000.0

# Define a WCS
wcs = WCS(header=hduc.header)

# Create list of pixel coordinates for pointings
PX = np.zeros(dim[0]*dim[1])
PY = np.zeros(dim[0]*dim[1])
k = 0
for i in range(dim[0]):
    for j in range(dim[1]):
        # rolling RA, Dec offsets for different rows/columns        
        PY[k] = j + i*roll[1]/spacing[1]
        PX[k] = i + j*roll[0]/spacing[0]
        # alternating RA, Dec offsets for different rows/columns
        try:
            if k == 0:
                print("Grid will have alternating RA, Dec offsets: {}, {} deg".format(delta[0], delta[1]))
                print("Grid will have rolling     RA, Dec offsets: {}, {} deg".format(roll[0], roll[1]))
            PY[k] += delta_y[i%len(delta_y)]/spacing[1]
            PX[k] += delta_x[j%len(delta_x)]/spacing[0]
        except:
            m=0 # dummy
        k += 1

# Calculate coordinates for all pointings
ST = utils.pixel_to_skycoord(PX, PY, wcs=wcs, origin=0, mode='wcs')

# Create list of pixel coordinates for all beams
BX = np.zeros(dim[0]*dim[1]*len(offsets))
BY = np.zeros(dim[0]*dim[1]*len(offsets))
l = 0
#for i in range(dim[0]):
#    for j in range(dim[1]):
#        for k in range(len(offsets)):
#            BX[l] = i+OFF_X[k]
#            BY[l] = j+OFF_Y[k]
#            l += 1
for k in range(len(PX)):
    for i in range(len(offsets)):
        BX[l] = PX[k]+OFF_X[i]*np.cos(np.pi*fang/180.0)-OFF_Y[i]*np.sin(np.pi*fang/180.0)
        BY[l] = PY[k]+OFF_X[i]*np.sin(np.pi*fang/180.0)+OFF_Y[i]*np.cos(np.pi*fang/180.0)
        l += 1

# Calculate coordinates for all pointings
#ST = utils.pixel_to_skycoord(PX, PY, wcs=wcs, origin=0, mode='wcs')

# List coordinates
for i in range(len(PX)):
    print("{}_{:03d}: ra={:.4f} dec={:.4f} (J2000)".format(fieldname, i, ST[i].ra.deg, ST[i].dec.deg))

# Plot size
plot_x = dim[0]*spacing[0]
plot_y = dim[1]*spacing[1]
plot_m = max(plot_x,plot_y)/10.0
plot_x /= plot_m
plot_y /= plot_m

# Make a plot
fig, ax = plt.subplots(figsize=(plot_x,plot_y), subplot_kw=dict(projection=wcs),num=fieldname) 
#plt.imshow(np.zeros((dim[0], dim[1])), origin='lower')
plt.xlabel("RA (J2000)")
plt.ylabel("DEC (J2000)")
plt.grid(color='white', ls='solid', alpha=0.1)
plt.scatter(PX, PY, marker="+", color="grey")
#ax.set_autoscale_on(False)
patches = []
for i in range(len(BX)):
    patches.append(Circle((BX[i], BY[i]), radius=beam/(2.0*spacing[1]), color='grey', fc=None, fill=False, alpha=0.4))
coll = PatchCollection(patches, color='grey', fc=None, alpha=0.4)
ax.add_collection(coll)
# Extend view beyond image boundaries
ax.set_xlim(-1, dim[0])
ax.set_ylim(-1, dim[1])
plt.tight_layout()
plt.show()

exit()

