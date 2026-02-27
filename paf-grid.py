#!/usr/bin/env python
#
# Example useage: 
# python paf-grid.py -s test.hdf5 -o test.fits -c test.counts.fits -S
# sdhdf version 1.9 and  python 3.11

# Lister Staveley-Smith

# python2 print compatibility
from __future__ import print_function

# Libraries (more are imported on demand once options are selected)

import sys
import re
import numpy as np
import h5py
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.wcs import WCS
from astropy.wcs import utils
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy.interpolate import CloughTocher2DInterpolator
from scipy import signal
import getopt
import os
import glob
import pkg_resources

# NGC 6744 gridding parameters
#coord = [287.4, -63.9, 1416.25]  # deg, deg, MHz
#dim = [150, 150, 460]
#dim = [150, 150, 1] 
#dcoord = [-0.01, 0.01, 0.005]      # deg, deg, MHz

# SMC gridding parameters
#coord = [15.0, -72.8, 1419.7]  # deg, deg, MHz  for SMC
#dim = [120, 120, 100] # SMC
#dcoord = [-0.05, 0.05, 0.005]      # deg, deg, MHz

# LMC4 gridding parameters
#coord = [83.0, -66.45, 1419.0]  # deg, deg, MHz
#dim = [40, 40, 100]
#dcoord = [-0.05, 0.05, 0.005]      # deg, deg, MHz

# LMC all gridding parameters
coord = [80.0, -69.0, 1419.0]  # deg, deg, MHz 
dim = [160, 160, 400] # LMC
# dim = [160, 160, 40] # LMC
dcoord = [-0.05, 0.05, 0.005]      # deg, deg, MHz

# Other (more constant parameters)
fwhm = 3.0  # FWHM convolving kernel in pixels (smooth=True)
rad  = 0.15 # radius of gridding kernel (deg)


# Process command line parameters
sig = 'none'
stokes = True # Only read Stokes I
qsub = False
asub = 0
bsub = 0
aBeam = 0
bBeam = 0
qBeam = False
start_ch = 0
end_ch = 0
output = "none"
counts = "none"
interpolate = False
smooth = False
grid = "none"
kernel = "tophat"
kernellist = ["tophat", "gaussian"]
history = ""

optionlist = ['?','a=', 'b=', 'A=', 'B=', 'c=', 'Grid=', 'I', 'k=', 'o=', 'sig=', 'S', 'v=', 'w=']

optionhelp = ['help', 'start sub-band', 'end sub-band', 'start beam', 'end beam', 'counts FITS file', 'Gridding text file', 'interpolate', 'kernel', 'Output FITS file', 'Input data file', 'Smooth', 'start channel', 'end channel']

try:
    options, remainder = getopt.getopt(sys.argv[1:], '?r:a:b:A:B:c:G:Irk::o:s:Sr:v:w:', optionlist)
    history = " ".join(sys.argv)
except getopt.error as err:
    print(str(err))
    print('OPTION LIST:', optionlist)
    print("Use -? option for help")
    sys.exit(2)

for opt, arg in options:
    if opt in ('-?', '--?'):      # help
        print("OPTION   DESCRIPTION")
        for i in range(len(optionlist)):
            print("{:8s} {}".format(optionlist[i], optionhelp[i]))
        sys.exit(0)   
    elif opt in ('-a', '--a'):      # start sub-band
        qsub = True
        asub = int(arg)
    elif opt in ('-b', '--b'):      # end sub-band
        bsub = int(arg)
    elif opt in ('-A', '--A'):      # start beam
        qBeam = True
        aBeam = int(arg)
    elif opt in ('-B', '--B'):      # start beam
        qBeam = True
        bBeam = int(arg)
    elif opt in ('-o', '--out'):    # filename for output cube
        output = arg    
    elif opt in ('-I', '--I'):    # interpolate
        interpolate = True    
    elif opt in ('-k', '--k'):    # kernel
        kernel = arg.lower()   
    elif opt in ('-c', '--counts'):    # filename for counts image
        counts = arg    
    elif opt in ('-s', '--sig'):    # filename for input spectra
        sig = arg
    elif opt in ('-G', '--Grid'):    # filename for input spectra
        grid = arg
    elif opt in ('-S', '--S'):    # smooth/convolve option (post-grid/interpolate)
        smooth = True
    elif opt in ('-v', '--v'):      # start channel
        start_ch = int(arg)
    elif opt in ('-w', '--w'):      # end channel
        end_ch = int(arg)

if len(remainder) != 0:
    print('Unrecognised arguments:', remainder)

# Certain option combinations not allowed
if sig == 'none':
  exit('ERROR: need input spectra (-s <hdf filename>)')
if bsub == 0:
    bsub=asub+1        
if output == 'none' and counts == 'none':
  exit('ERROR: need output FITS filename (-o <filename> and/or -c <filename>)')

        
# Further option processing
BEAM = "beam_{:02d}".format(aBeam)
if stokes:
    npol=1
else:
    npol=2
poldict = 'npol_'+str(npol)

if kernel not in kernellist:
    exit("ERROR: kernel \"{}\" is not available - please use: {}".format(kernel, kernellist))

# Check that output files don't exist
if output != "none":
    if os.path.exists(output):
        exit('ERROR: output file exists - {}'.format(output))
if counts != "none":
    if os.path.exists(counts):
        exit('ERROR: output file exists - {}'.format(counts))
    if output == counts:
        exit('ERROR: use different filenames for output and counts')

# Check that gridding file exists
success = False
if grid != "none":
    if os.path.exists(grid):
        ps = open(grid)
        print("Gridding instructions found: {}".format(grid))
        for line in ps:
            if any(x in line for x in ["coord", "dim", "dcoord", "fwhm", "rad"]):
                exec(line)
                success = True
        ps.close()
else:
    print('ERROR: gridding file does not exist - {}'.format(grid))
    print('It should be a text file which looks like this:')
    print('  # LMC all gridding parameters')
    print('  coord = [80.0, -69.0, 1419.0]  # deg, deg, MHz')
    print('  dim = [160, 160, 400]          # LMC')
    print('  dcoord = [-0.05, 0.05, 0.005]  # deg, deg, MHz')
    print('  # Other (more constant parameters)')
    print('  fwhm = 3.0  # FWHM convolving kernel in pixels (smooth=True)')
    exit('  rad  = 0.15 # radius of gridding kernel (deg)')
if not success:
    exit("ERROR: grid file not able to read - "+grid)

# Reference pixel (0-based integers only)
frefpix = np.array(dim)/2.0
refpix = frefpix.astype(int)

# Pixel centres
xcVALS = np.arange(dim[0]+1)
ycVALS = np.arange(dim[1]+1)
ZCVALS = np.arange(dim[2]+1)-refpix[2]
# Pixel edges
xeVALS = xcVALS-0.5
yeVALS = ycVALS-0.5
ZEVALS = ZCVALS-0.5

# Kernel for uv convolution
def gaussian_kernel(n, std, normalised=True):
    '''
    Generates a n x n matrix with a centered gaussian 
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian1D = signal.windows.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

# Define FITS header (counts) [make reference pixel 1-based]
hduc = fits.PrimaryHDU()
for i in range(len(dim)-1):
    hduc.header['CRVAL'+str(i+1)] = coord[i]
    hduc.header['CDELT'+str(i+1)] = dcoord[i]
    hduc.header['CRPIX'+str(i+1)] = refpix[i]+1
hduc.header['CTYPE1'] = "RA---SIN"
hduc.header['CTYPE2'] = "DEC--SIN"
hduc.header['EQUINOX'] = 2000.0
hduc.header.set('HISTORY', history)

# Define a WCS
wcs = WCS(header=hduc.header)

# Define FITS header (cube) [make reference pixel 1-based]
hduo = fits.PrimaryHDU()
for i in range(len(dim)-1):
    hduo.header['CRVAL'+str(i+1)] = coord[i]
    hduo.header['CDELT'+str(i+1)] = dcoord[i]
    hduo.header['CRPIX'+str(i+1)] = refpix[i]+1
hduo.header['CRVAL3'] = coord[2]*1.0e6
hduo.header['CDELT3'] = dcoord[2]*1.0e6
hduo.header['CRPIX3'] = refpix[2]+1
hduo.header['CUNIT3'] = 'HZ'
hduo.header['BUNIT'] = 'JY'
hduo.header['CTYPE1'] = "RA---SIN"
hduo.header['CTYPE2'] = "DEC--SIN"
hduo.header['EQUINOX'] = 2000.0
hduo.header['CTYPE3'] = "FREQ"
hduo.header['RESTFREQ'] = 1420.4058e6
hduo.header.set('HISTORY', history)

# Initial number of spectra
nspec = 0
# Initial number of coordinates
ncoord = 0

# Initialise data storage arrays
PX = np.array(0)
PY = np.array(0)
STORE = np.zeros((nspec, dim[2]))
STORE_COUNTS = np.zeros(nspec)

# miriad-like wildcard and multiple filename processing (@ prefix directs to a list of files); entries with wildcards need to be quoted on command line    
def filenameextractor(infile):
    if infile[0]=="@":
        try:
            with open(infile[1:], 'r') as fat:
                files = [line.rstrip() for line in fat]
            fat.close()
        except:
            exit("ERROR: unable to open {}".format(infile[1:]))
    elif re.search(r"\*", infile):
        files=glob.glob(infile)
    else:
        files = [infile]
    return files
    
# Sig filename handling        
sigfiles = filenameextractor(sig)
lsigfiles = len(sigfiles)
print(lsigfiles)
print(sigfiles)
 
for ifile in range(lsigfiles):
    file = sigfiles[ifile]
    filecore = file.split('/')[-1]
    print(ifile)
    print(file,"$")
    print(filecore)
    # Check that file exists
    if not os.path.exists(file):
        exit('ERROR: missing input file')  
    # open hdf file (tested on version 1.9)
    try:
        hdu = h5py.File(file, 'r')
    except:
        exit('ERROR: cannot read input file: '+file)      
    try:
      version = hdu['metadata']['primary_header'][0]['HEADER_DEFINITION_VERSION']
      # python 3 compatibility
      if isinstance(version,bytes):
          version = version.decode()
      print("SDHDF version number: {}".format(version))
      if float(version[:3]) < 1.9:
        print("WARNING: script may not work for SDHDF versions earlier than 5.0")
    except:
        exit("ERROR: check SDHDF version is 1.9 or greater")

    # See if BEAM exists
    if qBeam:
        if not BEAM in hdu.keys():
            exit("ERROR: requested beam not available: "+BEAM)
        if bBeam <= aBeam:
            bBeam = aBeam + 1
        endBEAM = 'beam_'+'{:02d}'.format(bBeam-1)
        if endBEAM not in list(hdu.keys()):
            exit("ERROR: requested end beam not available: "+endBEAM)
        BEAMLIST = [BEAM]
        for qb in range(aBeam+1, bBeam):
            BEAMLIST.append('beam_'+'{:02d}'.format(qb))
    else:
        BEAMLIST = [m for m in list(hdu.keys()) if 'beam' in m]
        BEAM = BEAMLIST[0]
    print("Data to be included:", BEAMLIST)

    # See if subband exists
    asubband = "band_SB{:d}".format(asub)
    if qsub:
        if asubband in hdu[BEAM].keys():
            print("Subband available:", asubband)
        else:
            exit("ERROR: requested subband not available: "+asubband)
    else:
        asubband = list(hdu[BEAM].keys())[0]
        print("Subband not selected: using", asubband)
        asub = int(asubband[-1])
        # Update bsub/srange
        if bsub <= asub:
            bsub += 1        
        srange = [asub, bsub]
            
    # Figure out source parameters (raw data v processed data)
    processed = False
    try:
        source = hdu[BEAM][asubband]['metadata']['source_params'][0,0]
        if isinstance(source,bytes):
            source = source.decode()
        RA_STR = hdu[BEAM][asubband]['metadata']['source_params'][0,1]
        DEC_STR= hdu[BEAM][asubband]['metadata']['source_params'][0,2]
        processed = True
        print("NOTIFICATION: processed data detected")
    except:
        try:
            source = hdu['metadata']['beam_parameters'][0]['SOURCE']
            if isinstance(source,bytes):
                source = source.decode()
            RA_STR = hdu[BEAM][asubband]['metadata']['observation_parameters'][0]['RIGHT_ASCENSION']
            DEC_STR = hdu[BEAM][asubband]['metadata']['observation_parameters'][0]['DECLINATION']
        except:
            exit("ERROR: requested beam or sub-band not available")
    print("Input:      {} ({})".format(file, source))
    
    # Loop over beams
    for BEAM in BEAMLIST:
        # Loop over subbands
        for isub in range(asub, bsub):
            csubband = "band_SB{:d}".format(isub)
            srange = [isub, isub+1]
            print("Subband:     SB"+str(isub))
    

            # Contents

            #Use h1ls or hdu.keys() to view file contents
            try:
                data_dims = hdu[BEAM][csubband]['astronomy_data']['data'].shape
                ndims = len(data_dims)
            except:
                exit("ERROR: requested beam {} and subband {} data do not exist".format(BEAM,csubband))
            try:
                freq_dims = hdu[BEAM][csubband]['astronomy_data']['frequency'].shape
            except:
                exit("ERROR: requested subband {} frequency data does not exist".format(csubband))

            # Figure out integration time parameters (raw data v processed data)

            try:
                int_times = hdu[BEAM][csubband]['metadata']['integ_times'][:]
            except:
                DUMP_TIME = hdu[BEAM]['metadata']['band_parameters'][0]['REQUESTED_INTEGRATION_TIME']
                N_DUMPS = hdu[BEAM]['metadata']['band_parameters'][0]['NUMBER_OF_INTEGRATIONS']
                int_times = np.array([DUMP_TIME]*N_DUMPS)
            int_time = np.sum(int_times)
            if (len(int_times) > 1):
                tsint = np.sort(int_times)
                if (tsint[0]+tsint[-1]) != 0.0:
                    if abs(2.0*(tsint[0]-tsint[-1])/(tsint[0]+tsint[-1])) > 0.1:
                        print("WARNING: spectra found with integration times differing by > 10%")
        
            print('Dimensions of sub-band data:', data_dims)

            # Estimated data size (NB. second dimension was a dummy axis for SDHDF version numbers <~ 2.4
            xpix1 = hdu[BEAM]['band_SB'+str(srange[0])]['astronomy_data']['data'].shape[-2]
            xpix = xpix1*(srange[1]-srange[0])
            ypix = hdu[BEAM]['band_SB'+str(srange[0])]['astronomy_data']['data'].shape[0]
            ppix = hdu[BEAM]['band_SB'+str(srange[0])]['astronomy_data']['data'].shape[-3]
            print('Pixels available: ({},{},{})'.format(xpix, ypix, ppix))
            if ppix < npol:
                print("WARNING: only {:d} polarisation available".format(ppix))
    
            # Channel range
            if end_ch == 0:
                end_ch = data_dims[-2]
    
            # Read signal (and reference), looping over (or averaging) polarisations
            apol = min(npol,ppix)
            for pol in range(apol):
                # Prepare slice arrays
                ind1 = [0]*ndims
                ind1[0] = slice(None)
                ind1[-3] = pol
                ind1[-2] = slice(start_ch,end_ch)
                ind2 = list(ind1)
                ind2[-3] = pol+1
                # Data
                tdata =  np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['astronomy_data']['data'][tuple(ind1)] for subband in range(srange[0],srange[1])]), axis=1)
                # Single or dual polarisation?
                if (npol == 1) and (ppix > 1):
                    t2data = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['astronomy_data']['data'][tuple(ind2)] for subband in range(srange[0],srange[1])]), axis=1)
                    tdata = (tdata + t2data)/2.0

                # Frequencies (concatenate subbands - deprecated)
                fval = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['astronomy_data']['frequency'][0][start_ch:end_ch] for subband in range(srange[0],srange[1])]))
                #print(fval.shape)
                lfval = len(fval)
                if lfval == 0:
                    exit("ERROR: no data available - adjust start/end channels?")
                    
                # Coordinates [NEED TO UPGRADE TO REJECT DATA WITH DRIVE ERRORS]
                RA = hdu[BEAM][csubband]['metadata']['observation_parameters']['RIGHT_ASCENSION']
                DEC = hdu[BEAM][csubband]['metadata']['observation_parameters']['DECLINATION']
                DRIVE = hdu[BEAM][asubband]['metadata']['observation_parameters']['DRIVE_STATUS']
                if np.any(DRIVE==0):
                    exit("ERROR: drive error detected")
                # Convert to degrees
                RADEC = SkyCoord([x.decode() for x in RA], [x.decode() for x in DEC], unit=(u.hourangle, u.deg), frame='icrs')
                # Pixel coordinates
                TX, TY = utils.skycoord_to_pixel(RADEC, wcs=wcs, origin=0, mode='wcs')
                # In case there are more coordinates than data values (e.g. paf.py has leftover metadata), just use the first
                ncoord += tdata.shape[0]
                PX.resize(ncoord)
                PY.resize(ncoord)
                PX[-tdata.shape[0]:] = TX[:tdata.shape[0]]
                PY[-tdata.shape[0]:] = TY[:tdata.shape[0]]
                OFF_Z = (fval - coord[2])/dcoord[2]
                
                # Spectral histograms for each time stamp
                for i in range(tdata.shape[0]):
                    T1, BINZ = np.histogram(np.resize(OFF_Z,tdata.shape[0]*tdata.shape[1]), bins=ZEVALS)
                    T2, BINZ = np.histogram(np.resize(OFF_Z,tdata.shape[0]*tdata.shape[1]), bins=ZEVALS, weights=tdata.flatten())
                    nspec += 1
                    STORE.resize((nspec,dim[2]), refcheck=False)
                    STORE[nspec-1,:] = T2/T1
                    STORE_COUNTS.resize(nspec, refcheck=False)
                    STORE_COUNTS[nspec-1] = T1[int(refpix[2])]
        # End of suband loop
    # End of beam loop
    hdu.close()
# End of file loop
    
# Add pixel markers in SE one pixel west of corner and at reference pixel
postest = False
if postest:
    PX.resize(ncoord+2)
    PY.resize(ncoord+2)
    STORE.resize((nspec+2,dim[2]), refcheck=False)
    PX[-2] = 1
    PY[-2] = 0
    PX[-1] = refpix[0]
    PY[-1] = refpix[1]
    STORE[nspec,:] = np.zeros(dim[2])
    STORE[nspec,0] = 1000.0
    STORE[nspec+1,:] = np.zeros(dim[2])
    STORE[nspec+1,0] = 1000.0
    ST = utils.pixel_to_skycoord(PX[-2], PY[-2], wcs=wcs, origin=0, mode='wcs')
    print("Adding #1 test point at (RA, Dec) ({:.3f}, {:.3f}) deg".format(ST.ra.deg, ST.dec.deg))
    ST = utils.pixel_to_skycoord(PX[-1], PY[-1], wcs=wcs, origin=0, mode='wcs')
    SC = SkyCoord(ST.ra.deg, ST.dec.deg, unit=(u.deg, u.deg), frame='icrs')
    PT = utils.skycoord_to_pixel(SC, wcs=wcs, origin=0, mode='wcs')
    print("Adding #2 test point at (RA, Dec) ({:.3f}, {:.3f}) deg, zero-based (X,Y) = ({:.1f}, {:.1f})".format(ST.ra.deg, ST.dec.deg, PT[0], PT[1]))

# Pixel-data 2D distances
GX, GY = np.mgrid[0:dim[0],0:dim[1]]
GZ = np.transpose([GX.flatten(), GY.flatten()])
PZ = np.transpose([PX, PY])
dists = cdist(GZ, PZ)
near = []
weights = []
for i in range(dists.shape[0]):
    nearest = np.where(dists[i,:] < np.abs(rad/dcoord[1]))
    near.append(nearest[0])
    if kernel == 'tophat':
        weights.append(np.ones(len(near[i])))
    elif kernel == 'gaussian':
        weights.append(np.exp(-dists[i,nearest[0]]**2/(2.0*np.pi*(fwhm/2.355)**2)))

# Counts 
# Histogram
#COUNTS, EX, EY = np.histogram2d(PY, PX, bins=[yeVALS,xeVALS])
# Top-hat kernel
COUNTS = np.zeros((dim[1],dim[0]))
for i in range(dists.shape[0]):
    COUNTS[GZ[i,1],GZ[i,0]] = np.sum(weights[i])

# Write FITS data (counts)
if counts != "none":
    hduc.data = np.float32(COUNTS)
    hduc.writeto(counts)


# Spatial images
if output != "none":
    fitsdata = np.zeros((dim[2], dim[1], dim[0]), dtype=np.float32)
    for k in range(dim[2]):
        if k%10 == 0:
            print("Gridding channel {}".format(k))
        # Simple histogram
        #SLICE, EX, EY = np.histogram2d(PY, PX, bins=[yeVALS,xeVALS], weights=STORE[:,k])
        # Normalise if pixel is occupied by data
        #SLICE[COUNTS!=0.0] = SLICE[COUNTS != 0.0]/COUNTS[COUNTS != 0.0]
        # Blank pixels with missing data
        # Top hat kernel
        SLICE = np.zeros((dim[1],dim[0]))
        for i in range(dists.shape[0]):
            if len(near[i]) > 0:
                SLICE[GZ[i,1],GZ[i,0]] = np.average(STORE[near[i],k], weights=weights[i])
        # Blank missing data
        SLICE[COUNTS==0.0] = np.nan
        # Optional interpolation into non-isolated pixels
        if  interpolate:
            x = GZ[:,1][COUNTS.flatten()!=0.0]
            y = GZ[:,0][COUNTS.flatten()!=0.0]
            X = GZ[:,1][COUNTS.flatten()==0.0]
            Y = GZ[:,0][COUNTS.flatten()==0.0]
            z = SLICE[COUNTS!=0.0]
            interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
            Z = interp(X, Y)
            for i in range(len(X)):
                SLICE[Y[i],X[i]] = Z[i]
        # Optional post-grid/interpolation Gaussian smooth
        if smooth:
            SLICE2 = np.copy(SLICE)
            SLICE[np.isnan(SLICE)] = 0.0 # Can't use nan's with FFT convolution
            ngauss = 2*int(fwhm*2.0)+1 # Truncate Gaussian 
            nsl = int(ngauss/2.0) # purpose???
            # Convolve
            SLICE = signal.convolve(SLICE, gaussian_kernel(ngauss,fwhm/2.355), mode='same', method='auto')
            SLICE[np.isnan(SLICE2)] = np.nan
        fitsdata[k,:,:] = np.float32(SLICE)
    
# Write FITS data (cube)
if output != "none":
    hduo.data = fitsdata
    hduo.writeto(output)

# Check
if counts != "none":
    hduc = fits.open(counts, mode='denywrite')
    hduc.info()
    hduc.close()

if output != "none":
    hduo = fits.open(output, mode='denywrite')
    hduo.info()
    hduo.close()

exit()

