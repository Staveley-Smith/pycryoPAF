#!/usr/bin/env python

# Lister Staveley-Smith
# 2024 June 11: python 3 and UWL/PAFcompatibility

import h5py
import sys
import re
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt

files = sys.argv[1:]
n = len(files)

if n == 0:
    exit("ERROR: no files specified")
if re.search(r"\*", files[0]):
    exit("ERROR: no files found")

for f in files:
    newfile = False
    try:
        hdu=h5py.File(f, 'r')
    except:
        exit("ERROR: unable to read {}".format(f))
    BEAMS = [x for x in list(hdu.keys()) if "beam" in x]
    #BEAM = BEAMS[0]
    SBS = [x for x in list(hdu[BEAMS[0]].keys()) if "band" in x]
    SB0 = SBS[0]
    #SBX = "band_SB5"
    try:
        SB = hdu[BEAMS[0]][SB0]
    except:
        exit("ERROR: file can't be read: "+f)
    rf = f.split("/")[-1]
    UWL = True
    newfile = False
    if n == 1:
      print("Only 1 file specified, so executing deep dive for", rf, SB0)
    try:
      #  UWL
      N_DUMPS = hdu[BEAMS[0]]['metadata']['band_params'][0]['N_DUMPS']
    except:
      # PAF
      N_DUMPS = hdu[BEAMS[0]]['metadata']['band_parameters'][0]['NUMBER_OF_INTEGRATIONS']
      UWL = False
    if 'metadata' in list(hdu.keys()):
        if UWL:
          SOURCE = hdu['metadata']['beam_params'][0]['SOURCE'].decode()
          CAL_MODE = hdu['metadata']['primary_header'][0]['CAL_MODE'].decode()
        else:
          SOURCE = hdu['metadata']['beam_parameters'][0]['SOURCE'].decode()
          CAL_MODE = hdu['metadata']['primary_header'][0]['CALIBRATION_MODE'].decode()
    else:
        SOURCE = 'N/A'
        CAL_MODE= 'N/A'
    try:
      # New files
      int_times = hdu[BEAMS[0]][SB0]['metadata']['integ_times'][:]
      avfilenames = hdu[BEAMS[0]][SB0]['metadata']['averaged_filenames'][:]
      newfile = True
    except:
      # Parkes files
      if UWL:
        int_times = np.array([hdu[BEAMS[0]]['metadata']['band_params'][0]['DUMP_TIME']]*N_DUMPS)
      else:
        int_times = np.array([hdu[BEAMS[0]]['metadata']['band_parameters'][0]['REQUESTED_INTEGRATION_TIME']]*N_DUMPS)
    ltimes = len(int_times)
    int_time = np.sum(int_times)
    if n != 1:
      nstr = ""
    if f == files[0]:
      if 'metadata' in list(hdu.keys()):
          if UWL:
            hdrv = hdu['metadata']['primary_header'][0]['HDR_DEFN_VERSION'].decode()
          else:
            hdrv = hdu['metadata']['primary_header'][0]['HEADER_DEFINITION_VERSION'].decode()
          if newfile:
              print("Processed data detected (first file, {})".format(SB0))
      else:
          hdrv = "N/A"
      print("HDR_DEFN_VERSION (first file):", hdrv)
      print('{:21.21} {:20.20} {:} {:} {:10.10} {:9.9} {:3.3} {:5.5}'.format('  HDF file', ' Source', 'B', 'I','  RA', '  Dec', 'cal', '  sec'))

    PR = []
    PD = []
    PB = []
    PT = []
    
    if (n == 1) and (len(BEAMS) > 1):
        endBEAM = len(BEAMS)
    
    else:
        endBEAM = 1
    for b in range(endBEAM):
        BEAM = BEAMS[b]
        if newfile:
          CAL_MODE = "N/A"
          SOURCE = hdu[BEAM][SB0]['metadata']['source_params'][:,0][0].decode()
          RA_T = hdu[BEAM][SB0]['metadata']['source_params'][:,1]
          RA_STR = [x.decode() for x in RA_T]
          DEC_T= hdu[BEAM][SB0]['metadata']['source_params'][:,2]
          DEC_STR = [x.decode() for x in DEC_T]
          DRIVE = [1]*ltimes
        else:
          # Parkes file
          if UWL:  
            RA_T = hdu[BEAM][SB0]['metadata']['obs_params']['RA_STR']
            RA_STR = [x.decode() for x in RA_T]
            DEC_T = hdu[BEAM][SB0]['metadata']['obs_params']['DEC_STR']
            DEC_STR = [x.decode() for x in DEC_T]
            UTC_T = hdu[BEAM][SB0]['metadata']['obs_params']['UTC']
            UTC = [x.decode() for x in UTC_T]
            UT_DATE = hdu[BEAM][SB0]['metadata']['obs_params']['UT_DATE']
            t1 = hdu[BEAM][SB0]['metadata']['obs_params']['MJD']
          else:
            RA_T = hdu[BEAM][SB0]['metadata']['observation_parameters']['RIGHT_ASCENSION']
            RA_STR = [x.decode() for x in RA_T]
            DEC_T = hdu[BEAM][SB0]['metadata']['observation_parameters']['DECLINATION']
            DEC_STR = [x.decode() for x in DEC_T]
            UTC_T = hdu[BEAM][SB0]['metadata']['observation_parameters']['UTC']
            UTC = [x.decode() for x in UTC_T]
            t1 = hdu[BEAM][SB0]['metadata']['observation_parameters']['MJD']
            DRIVE = hdu[BEAM][SB0]['metadata']['observation_parameters']['DRIVE_STATUS']
          t2 = Time(t1, format='mjd')
          UT_DATE = t2.to_value(format='iso', subfmt='date')

        for i in range(ltimes): #(9,ltimes) for LMC plot!
          if (DEC_STR[i][0] != "-") and (DEC_STR[i][0] != "+"):
              DEC_STR[i] = "+"+DEC_STR[i]
          if n == 1:
              it = int_times[i]
              nstr = "("+str(i)+")"
              if newfile:
                rf = avfilenames[i].decode()
              else:
                rf = UT_DATE[i]+"-"+UTC[i]
          else:
              it = int_time

          # Keep positions for plotting
          if DRIVE[i] != 0:
              PR.append(RA_STR[i])
              PD.append(DEC_STR[i])
              PB.append(b)
              PT.append(i)

          # Output summary
          if b == 0:
              print('{:21.21} {:20.20} {:} {:} {:10.10} {:9.9} {:3.3} {:.1f} {:}'.format(rf, SOURCE, len(BEAMS), len(SBS), RA_STR[i], DEC_STR[i], CAL_MODE, it, nstr))
          if n > 1:
              break
          
    # Make a plot if doing a deep dive
    if n==1:
      # System RA, Dec columns
      #RADEC = SkyCoord(RA_STR, DEC_STR, unit=(u.hourangle, u.deg), frame='icrs')
      RADEC = SkyCoord(PR, PD, unit=(u.hourangle, u.deg), frame='icrs')
      dcoord = [-0.5, 0.5]
      #xlo = np.min(RADEC[DRIVE!=0].ra.deg)
      #xhi = np.max(RADEC[DRIVE!=0].ra.deg)
      xlo = np.min(RADEC.ra.deg)
      xhi = np.max(RADEC.ra.deg)
      coord = [(xlo+xhi)/2.0]
      #ylo = np.min(RADEC[DRIVE!=0].dec.deg)
      #yhi = np.max(RADEC[DRIVE!=0].dec.deg)
      ylo = np.min(RADEC.dec.deg)
      yhi = np.max(RADEC.dec.deg)
      coord.append((ylo+yhi)/2.0)
      dim = [3+abs(int(np.cos(np.pi*coord[1]/180.0)*(xhi-xlo)/dcoord[0])), 3+abs(int((yhi-ylo)/dcoord[1]))]
      refpix = (1+np.array(dim)/2.0).astype(int)
      
      # Define a FITS header 
      hduc = fits.PrimaryHDU()
      hduc.header['NAXIS'] = 2
      for j in range(2):
          hduc.header['NAXIS'+str(j+1)] = dim[j]
          hduc.header['CRVAL'+str(j+1)] = coord[j]
          hduc.header['CDELT'+str(j+1)] = dcoord[j]
          hduc.header['CRPIX'+str(j+1)] = refpix[j]
      hduc.header['CTYPE1'] = "RA---SIN"
      hduc.header['CTYPE2'] = "DEC--SIN"
      hduc.header['EQUINOX'] = 2000.0

      # Define a WCS
      wcs = WCS(header=hduc.header)

      # Sky to pixel conversion
      PX, PY = utils.skycoord_to_pixel(RADEC, wcs=wcs, origin=0, mode='wcs')
      
      # Temp 
      MBPOS = SkyCoord('05h07m10s','-69d14m41s',frame='icrs')
      print(MBPOS)
      sep = RADEC.separation(MBPOS)
      dmin=np.min(sep.value)
      wmin=np.argwhere(sep.value==dmin)[0][0]
      print(wmin, PB[wmin], PT[wmin], dmin)
  
      # Plot parameters
      plt.rc('font',family='serif',size=18)
      plt.rcParams['axes.linewidth'] = 4
      
      # Plot
      ax = plt.subplot(projection=wcs) 
      plt.imshow(np.ones((dim[1], dim[0], 3)), origin='lower')
      plt.xlabel("RA (J2000)")
      plt.ylabel("DEC (J2000)")
      plt.grid(color='grey', ls='solid', alpha=0.3)
      ax.set_autoscale_on(False)
      plt.scatter(PX, PY, marker="+", color="black", lw=0.5, alpha=0.5)
      plt.scatter(PX[wmin], PY[wmin], marker="+", color="red", lw=2, s=100)
      plt.tight_layout()
      plt.show()
        
    hdu.close()
exit()