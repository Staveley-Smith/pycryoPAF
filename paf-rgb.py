#!/usr/bin/env python
#
# Lister Staveley-Smith
# 2024 Nov 29: paf-rgb
# then /Applications/FFMPEG/ffmpeg -framerate 20 -pattern_type glob -i 'rgb_*.png' -c:v libx264 -pix_fmt yuv420p LMC_S9new_rgb.mp4

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import os

# Input file
#input = "LMC_S9_r=0.15_I_G_S=2.7.fits"
input = "LMC_S9_wide.fits"

if input != "none":
    if not os.path.exists(input):
        exit('ERROR: no file found - {}'.format(input))

# Open FITS cube
hdu = fits.open(input, mode='denywrite')
hdu.info()
data = np.copy(hdu[0].data)
data[np.isnan(data)] = 0.0
dim = data.shape

# Define a WCS
wcs = WCS(header=hdu[0].header)

if hdu[0].header['CTYPE3'] == "FREQ":
    vel =  True
    freq  = (np.linspace(1,dim[0]+1,dim[0]+1)-hdu[0].header['CRPIX3'])*hdu[0].header['CDELT3']+hdu[0].header['CRVAL3']
    velocity  = (1420.4058e6/freq-1.0)*299792.46
else:
    vel = False

# Plot
# SMC
#iw = 3 # width of each rgb channel
#im = 3 # spacing between rgb channels
#ik = 1 # skip factor
# LMC
iw = 12 # width of each rgb channel
im = 12 # spacing between rgb channels
ik = 36 # skip factor

minnoise = 1.0 # lower limit 

#COLOR = "white"
COLOR = "black"
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.edgecolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

first = True
#for i in range(0,dim[0]-(iw+2*im),ik):
for i in range(0,dim[0],ik):
    if i+2*im+iw > dim[0]:
        break
    if i%10 == 0:
        print("Rendering channel {}".format(i))
    if first:
        fig=plt.figure(num=input,figsize=[5,5],dpi=200)
        # Black background
        #fig.patch.set_facecolor('black')
        # White background
        fig.patch.set_facecolor('white')
        # slices seems a bit pointless
        ax = fig.add_subplot(projection=wcs, slices=('x', 'y', i)) 
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("DEC (J2000)")
        ax.grid(ls='solid', color="white", alpha=0.2)
        first = False
    channel = i+int(1.5*im)
    smax = np.max(np.mean(data[i:i+2*im+iw,:,:],axis=0))#/2.0
    #print(i, smax)
    if smax < minnoise:
        smax = minnoise
    image_r = np.mean(data[i:i+iw,:,:],axis=0)/smax
    image_g = np.mean(data[i+im:i+im+iw,:,:],axis=0)/smax
#    image_b = np.mean(data[i+im+iw:i+im+2*iw,:,:],axis=0)/smax
    image_b = np.mean(data[i+2*im:i+2*im+iw,:,:],axis=0)/smax
    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
    ax.imshow(image)
    if vel:
        ax.set_title("({:.1f},{:.1f})".format(velocity[i+int(iw/2)], velocity[i+int(iw/2)+2*im])+" km s$^{-1}$")
        ax.set_title("({:.1f},{:.1f})".format(velocity[i+2*im+iw],velocity[i])+" km s$^{-1}$")
    else:
        ax.set_title("Channel {}".format(channel))
    #plt.pause(0.001)
    plt.tight_layout()
    plt.savefig('rgb_{:04d}.png'.format(i))
    #plt.show(block=False)

hdu.close()
exit()