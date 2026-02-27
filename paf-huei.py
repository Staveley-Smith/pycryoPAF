#!/usr/bin/env python
#
# Lister Staveley-Smith
# 2025 Jul 31: paf-huei
# From paf-rgb: a moment 0 image coloured by velosity

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import os

# Input file
input = "LMC_S9_r=0.15_I_G_S=2.7.fits"

if input != "none":
    if not os.path.exists(input):
        exit('ERROR: no file found - {}'.format(input))

# Open FITS cube
hdu = fits.open(input, mode='denywrite')
hdu.info()
data = np.copy(hdu[0].data)
datamin = 0.1
data[np.isnan(data)] = datamin
data[data<datamin] = datamin
dim = data.shape

# Define a WCS
wcs = WCS(header=hdu[0].header)

if hdu[0].header['CTYPE3'] == "FREQ":
    vel =  True
    freq  = (np.linspace(1,dim[0]+1,dim[0]+1)-hdu[0].header['CRPIX3'])*hdu[0].header['CDELT3']+hdu[0].header['CRVAL3']
    velocity  = (1420.4058e6/freq-1.0)*299792.46
else:
    vel = False

#COLOR = "white"
COLOR = "black"
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.edgecolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

fig=plt.figure(num=input,figsize=[5,5],dpi=200)
# White background
fig.patch.set_facecolor('white')
ax = fig.add_subplot(projection=wcs, slices=('x', 'y', 0)) 
ax.set_xlabel("RA (J2000)")
ax.set_ylabel("DEC (J2000)")
ax.grid(ls='solid', color="white", alpha=0.2)
first = False

smax = np.max(data)

image_r = np.mean(data[0:135,:,:],axis=0)
image_g = np.mean(data[100:150,:],axis=0)
image_b = np.mean(data[130:200,:,:],axis=0)
smax = np.max([np.max(image_r),np.max(image_g),np.max(image_b)])

image = make_lupton_rgb(image_r/smax, image_g/smax, image_b/smax, stretch=0.15)
ax.imshow(image)
plt.tight_layout()
plt.savefig('rgb_huei.png')
plt.show()

hdu.close()
exit()