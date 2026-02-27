#!/usr/bin/env python
#
# Pupose:
# Similar to paf-scan.py but reads all beam data (for one subband) simultaneously (rather than per beam) in order to facilitate >2D SVD analysis

# Example useage: 
# paf-scan.py -s paf_250224_064528.hdf5 -A 71 -a 1 -h paf_250223_233303_ON_cal.json -z

# Lister Staveley-Smith 12 April 2025

# Libraries (more may be imported on demand once options are selected)
import sys
import re
import csv
import numpy as np
from scipy import signal
from scipy import fftpack
#from scipy.interpolate import NearestNDInterpolator
from astropy.cosmology import Planck18 as cosmo
from scipy import stats
import h5py
import getopt
import json
import os
import glob
import pkg_resources
from pylab import rcParams
import matplotlib.pyplot as plt
# conda install -c tensorly tensorly
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from l1csvd import l1csvd
from utils import *
import time

# HI rest frequency
f0 = 1420.4058

# Planck18 cosmology
hPlanck = 0.674

# Initialise calparams
calparams = {}

# Process command line parameters

sig = 'none'
stokes = False
qsub = False
asub = 0
bsub = 0
aBeam = 0
bBeam = 0
qBeam = False
start_ch = 0
end_ch = 0
uylim = 0.0
dylim = 0.0
poly = None
fit_range = "0,0"
qrange= False
smooth_par = "1,1"
qsmooth= False
calin = "none"
qcalin = True
nodisplay = False
block = True
output = "none"
qsvd = False
nsvd = 1
method = "nonenone"
methodlist = ['medi', 'mean', 'none']
smooth = 1
zoom = False
textind = 0
fake = "none"
fakestr = "none"
SVDmethodlist = ['CSVDstack', 'TuckerSVD', 'CPSVD', 'SVD', 'CSVD', 'L1SVD']
fakeamp = 1.0
history =""

optionshort = '?a:b:A:B:d:e:f:F:h:H:im:nNo:p:s:u:v:w:zS:x:X:'
optionlist = ['?','a=', 'b=', 'A=', 'B=', 'd=', 'e=', 'F=', 'h=', 'H=', 'i', 'method=', 'n', 'N', 'o=', 'p=', 'sig=', 'u=', 'v=', 'w=', 'zoom', 'SVD=', 'x=', 'X=']
optionhelp = ['help', 'start sub-band', 'end sub-band', 'start beam', 'end beam', 'lower plot limit', 'number of smoothing channels in frequency(,time)',
              'channel range for polynomial fit (comma-separated)', 'apply this calibration file', 'apply this calibration to other beams (first entry)', 'Stokes I',
              'method for freq-time pre-conditioning (meanmean/medimedi/nonemedi etc.)', 'no plot', 'unblocked plot', 'Output data file', 'polynomial baseline order (0 and -ve values applied pre-time-average)', 'source (signal) spectrum', 
              'upper plot limit', 'start channel number (within each sub-band)', 'end channel number (within each sub-band)', 
              'large font (for publication)', 'SVD method (CSVDstack-n/TuckerSVD-n/CPSVD-n, where n is number of singular values)', 'extra (fake) line (HI only)', 'extra fake line amplitude']

optionform = re.split("", optionshort)
na = 0
for a in optionform:
    if (a != "") and (a != ":"):
        optionform[na] = '-' + a
        na += 1

try:
    options, remainder = getopt.getopt(sys.argv[1:], optionshort, optionlist)
    history = " ".join(sys.argv)
except getopt.error as err:
    print(str(err))
    print('OPTION LIST:', optionlist)
    print("Use -? option for help")
    sys.exit(2)

for opt, arg in options:
    if opt in ('-?', '--?'):      # help
        print(" OPTION         DESCRIPTION")
        for i in range(len(optionlist)):
            print("{:6s} {:8s} {}".format(optionform[i], optionlist[i], optionhelp[i]))
        sys.exit(0)   
    elif opt in ('-s', '--sig'):    # filename for signal spectrum
        sig = arg
    elif opt in ('-h', '--h'):      # input calibration filename
        calin = arg
    elif opt in ('-H', '--H'):      # input calibration filename
        calin = arg
        qcalin = False
    elif opt in ('-k', '--k'):      # dashed vertical line location
        markfreq = float(arg)
    if opt in ('-F', '--F'):      # channel ranges for polynomial fit - e.g. -F 1000,2000,4000,5000
        qrange = True
        fit_range = arg
    elif opt in ('-a', '--a'):      # start sub-band
        qsub = True
        asub = int(arg)
    elif opt in ('-b', '--b'):      # end sub-band
        bsub = int(arg)
    elif opt in ('-A', '--A'):      # start beam
        qBeam = True
        aBeam = int(arg)
    elif opt in ('-B', '--B'):      # start beam
        bBeam = int(arg)
    elif opt in ('-e', '--e'):      # number of smoothing channels in frequency, time e.g. -e 5 or -e 5,2
        qsmooth = True
        smooth_par = arg
    elif opt in ('-p', '--p'):      # polynomial order
        poly = int(arg)
    elif opt in ('-u', '--u'):      # upper plot limit
        uylim = float(arg)
    elif opt in ('-d', '--d'):      # lower plot limit
        dylim = float(arg)
    elif opt in ('-v', '--v'):      # start channel
        start_ch = int(arg)
    elif opt in ('-w', '--w'):      # end channel
        end_ch = int(arg)
    elif opt in ('-i', '--i'):      # Average polarisations to Stokes I
        stokes = True    
    elif opt in ('-l', '--line'):      # Plot location of zero-redshift recombination lines and legend
        vline = True    
    elif opt in ('-n', '--n'):      # Skip display
        nodisplay = True    
    elif opt in ('-N', '--N'):      # display briefly
        block = False    
    elif opt in ('-m', '--method'):      # Pre-conditioning
        method = arg  
    elif opt in ('-o', '--out'):      # filename for output calibrated spectrum
        output = arg    
    elif opt in ('-z', '--zoom'):      # Skip display
        zoom = True    
    elif opt in ('-S', '--SVD'):      # Skip display
        qsvd = True
        SVDmethod = arg    
    elif opt in ('-x', '--x'):    # Fake line type
        fake = arg
    elif opt in ('-X', '--X'):    # Fake line amplitude factor
        fakestr = arg

if len(remainder) != 0:
    print('Unrecognised arguments:', remainder)
    exit()

# Certain option combinations not allowed

if sig == 'none':
  exit('ERROR: need signal file (-s <hdf name>)')

if bsub == 0:
    bsub=asub+1
if (bsub-asub) > 1:
    exit("ERROR: can only process a single subband")        
        
qmethod = False
if len(method) == 8:
    if (method[:4] in methodlist) and (method[4:] in methodlist):
        qmethod = True
if not qmethod:
    exit("ERROR: method needs to be \'XY\' where X and Y are one of {}".format(methodlist))

# Further option processing

BEAM = "beam_{:02d}".format(aBeam)

# Decode channel fitting limits for polynomial
if qrange and (poly == None):
    exit("ERROR: Option -F requires a polynomial order (-p) to be specified")
try:
    frange = [int(x) for x in fit_range.split(",")]
    pdim = len(frange)
except:
    exit("ERROR: invalid format for option F - use \"-F 0,100,0,100,0,10\" or similar")

if qrange:
    for i in range(pdim):
        if (start_ch != 0) or (end_ch != 0):
            if (frange[i] < start_ch) or (frange[i] > end_ch):
                exit("ERROR: channels in option -F outside range specified in options -v/w")
if (pdim % 2) != 0:
    exit("ERROR: even number of channels expected for option -F")

# Decode smoothing parameters
t = smooth_par.split(",")
try:
    if len(t) == 1:
        smooth = max(int(t[0]),1)
        tsmooth = 1
    elif len(t) == 2:
        smooth = max(int(t[0]),1)
        tsmooth = int(t[1])
    else:
        exit("ERROR: a maximum of 2 smoothing paramaters required - e.g. \"-e 2,2\"")
except:
    exit("ERROR: invalid smoothing parameters")

if qsvd:
    tmp = re.split('-', SVDmethod, maxsplit=2)
    if len(tmp) == 2:
        SVDmethod = tmp[0]
        nsvd = int(tmp[1])
    else:
        print("ERROR: use \"-S SVDmethod-n\", where n is the number of singular values.")
        exit("ERROR: valid SVD method are: {}".format(SVDmethodlist))
    t = False
    for m in SVDmethodlist:
        if SVDmethod == m:
            t = True
    if SVDmethod == '' or not t:
        exit("ERROR: valid SVD methods: {}".format(SVDmethodlist))    

if fake != "none":
    if (fake != "HI") and (fake != "IM"):
        exit("ERROR: only \"-x HI\" or \"-x IM\"supported")
    if fakestr != "none":
        try:
            fakeamp = float(fakestr)
        except:
            exit("ERROR: argument for -X needs to be the amplitude of the fake line")
    print("NOTIFICATION: fake line amplitude = {:.1e}".format(fakeamp))
elif fakestr != "none":
    exit("ERROR: option \"-X\" is redundant without option \"-x HI\" etc (extra fake line)")
fakeIMfile = 'none'
if (fake == "IM"):
    t = os.path.expanduser("~/Drive/PROJECTS/P1199/G23/G23_HIskymodel.npz")
    if os.path.isfile(t):
        fakeIMfile = t
        print("Intensity map found:", fakeIMfile)
    else:
        exit("ERROR: option \"-x IM\" needs an intensity map: {}".format(t))

if stokes:
    npol=1
else:
    npol=2
    
# Number of beams to read
if bBeam <= aBeam:
    bBeam = aBeam + 1
nbeam = bBeam - aBeam

poldict = 'npol_'+str(npol)

# Beam for fake source
fbeam = aBeam + int(nbeam/2)

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
    
# Figure size in x and y (supposedly in inches); white background

plt.rcParams['figure.figsize'] = [12, 6.75]
plt.rcParams['figure.facecolor'] = 'white'

# Output display resolution (determines sampling interval)
yres = 2048

# Sig filename handling        
sigfiles = filenameextractor(sig)
lsigfiles = len(sigfiles)

# Cal filename handling and duplication       
if calin != "none":
    calfiles = filenameextractor(calin)
    lcalfiles = len(calfiles)
    if (lcalfiles == 1) and (lsigfiles != 1):
        for i in range(1,lsigfiles):
            calfiles.append(calfiles[0])
        lcalfiles = len(calfiles)
    if (lcalfiles != lsigfiles):
        exit("ERROR: number of cal files must be one or be equal to the number of sig files")

# Output filename handling and duplication       
if output != "none":
    outfiles = filenameextractor(output)
    loutfiles = len(outfiles)
    if (loutfiles == 1) and (lsigfiles != 1):
        for i in range(1,lsigfiles):
            outfiles.append(outfiles[0])
        loutfiles = len(outfiles)
    if (loutfiles != lsigfiles):
        exit("ERROR: number of cal files must be one or be equal to the number of sig files")
 
for ifile in range(lsigfiles):
    # Open sig file    
    file = sigfiles[ifile]
    filecore = file.split('/')[-1]
    # open hdf file (tested on version 1.9)
    try:
        hdu = h5py.File(file, 'r')
    except:
        exit("ERROR: unable to read {}".format(file))
    try:
      version = hdu['metadata']['primary_header'][0]['HEADER_DEFINITION_VERSION']
      # python 3 compatibility
      if isinstance(version,bytes):
          version = version.decode()
      if float(version[:3]) < 1.9:
        print("WARNING: script may not work for SDHDF versions earlier than 5.0")
    except:
        version = "N/A"
    print("SDHDF version number: {}".format(version))

    # See if BEAM exists
    if qBeam:
        if BEAM in list(hdu.keys()):
            print("BEAM available:", BEAM)
        else:
            exit("ERROR: requested beam not available: "+BEAM)
    else:
        BEAM = list(hdu.keys())[0]
        print("BEAM not selected: using", BEAM)
        aBeam = int(BEAM[-2:])
        # Update bBEAM/nbeam
        bBeam = aBeam+1
        nbeam = bBeam - aBeam
        fbeam = aBeam + int(nbeam/2)
    endBEAM = 'beam_'+'{:02d}'.format(bBeam-1)
    if endBEAM not in list(hdu.keys()):
        exit("ERROR: requested end beam not available: "+endBEAM)

    # See if subband exists
    asubband = "band_SB{:d}".format(asub)
    if qsub:
        if asubband in list(hdu[BEAM].keys()):
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
        processed = True
        print("NOTIFICATION: processed data detected")
    except:
        try:
            source = hdu['metadata']['beam_parameters'][0]['SOURCE']
            if isinstance(source,bytes):
                source = source.decode()
        except:
            source='N/A'
    print("Signal:      {} ({})".format(file, source))
    
    # Open cal file
    if calin != "none":
        try:
            with open(calfiles[ifile], 'r') as fcal:
                calparams = json.load(fcal)
            fcal.close()
            print("Calibration:", calfiles[ifile])
        except:
            exit("ERROR: calibration file not found")

    # Loop over beams
    for ibeam in range(aBeam, bBeam):
        BEAM = "beam_{:02d}".format(ibeam)
        # Calibration beam should be same as beam unless forced otherwise
        if qcalin:
            CALBEAM = BEAM
        else:
            CALBEAM = list(calparams)[0]
            if BEAM != CALBEAM:
                print("WARNING: Applying {} calibration parameters to {}".format(CALBEAM, BEAM))

        # Can't loop over subbands here
        isub = asub
        csubband = "band_SB{:d}".format(isub)

    # Source metadata
        try:
            RA_STR = hdu[BEAM][asubband]['metadata']['source_params'][0,1]
            DEC_STR= hdu[BEAM][asubband]['metadata']['source_params'][0,2]
            DRIVE = "N/A"
        except:
            try:
                RA_STR = hdu[BEAM][asubband]['metadata']['observation_parameters'][0]['RIGHT_ASCENSION']
                DEC_STR = hdu[BEAM][asubband]['metadata']['observation_parameters'][0]['DECLINATION']
                DRIVE = hdu[BEAM][asubband]['metadata']['observation_parameters']['DRIVE_STATUS']
            except:
                RA_STR = "N/A"
                DEC_STR = "N/A"
                DRIVE = "N/A"

        # Contents
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
            int_times = np.array([DUMP_TIME]*data_dims[0])
        int_time = np.sum(int_times)
        if (len(int_times) > 1):
            tsint = np.sort(int_times)
            if (tsint[0]+tsint[-1]) != 0.0:
                if abs(2.0*(tsint[0]-tsint[-1])/(tsint[0]+tsint[-1])) > 0.1:
                    print("WARNING: spectra found with integration times differing by > 10%")        

        print('Dimensions of sub-band data:', data_dims)

        # Estimated data size (NB. second dimension was a dummy axis for SDHDF version numbers <~ 2.4)

        xpix = hdu[BEAM]['band_SB'+str(isub)]['astronomy_data']['data'].shape[-2]
        ypix = hdu[BEAM]['band_SB'+str(isub)]['astronomy_data']['data'].shape[0]
        ppix = hdu[BEAM]['band_SB'+str(isub)]['astronomy_data']['data'].shape[-3]
        tpix = hdu[BEAM]['band_SB'+str(isub)]['astronomy_data']['data'].shape[-1]
        #print('Pixels available: ({},{},{})'.format(xpix, ypix, ppix,tpix))
        if ppix < npol:
            print("WARNING: only {:d} polarisation available".format(ppix))

        # Channel range
        if end_ch == 0:
            end_ch = data_dims[-2]
        # Plot channel range (IFs can no longer be joined)
        plot_start_ch = start_ch
        plot_end_ch = end_ch #+xpix-xpix1
   
        # Skip pixels for display?
#        xskip = max(1,int((end_ch-start_ch)/xres))
        xskip = smooth
        yskip = tsmooth
        #yskip = max(1,int(ypix/yres))
        print('Smoothing and resampling by {:d} spectral channels'.format(xskip))
        carray = np.ones((yskip, xskip))/float(yskip*xskip)
        tarray = np.ones(yskip)

        # Read signal, looping over (or averaging) polarisations        
        apol = min(npol,ppix)
        for pol in range(apol):

            # Prepare slice arrays
            ind1 = [0]*ndims
            ind1[0] = slice(None)
            ind1[-3] = pol
            ind1[-2] = slice(start_ch,end_ch)
            ind2 = list(ind1)
            ind2[-3] = pol+1

             # Data / cal data
            tdata =  hdu[BEAM][csubband]['astronomy_data']['data'][tuple(ind1)]
            nnan = sum(np.isnan(tdata.flatten()))
            if nnan > 0:
                print("WARNING: nans detected *************************************************")
            if (npol == 1) and (ppix > 1):
                t2data = hdu[BEAM][csubband]['astronomy_data']['data'][tuple(ind2)]
                tdata = (tdata + t2data)/2.0
                print("Stokes I spectrum")
                rat = abs(np.median(tdata)/np.median(t2data))
                if (rat < 0.666) or (rat > 1.5):
                    print("WARNING: significant difference between levels of two pols (factor {:.2f})".format(rat))
                    print("WARNING: Stokes I processing may not be appropriate")
            elif npol == 2:
                if pol == 0:
                    print("Stokes XX spectrum")
                elif (pol == 1) and (ppix > 1):
                    print("Stokes YY spectrum")
                    
            # Average/convolve (boundary="sim" doesn't seem to work) and downsample for display only        
            tvdata = signal.convolve2d(tdata[DRIVE!=0], carray, mode="same", boundary="wrap")
            tdata = tvdata[::yskip,::xskip]
            cint_times = signal.convolve(int_times[DRIVE!=0], tarray, mode="same")
            print('Dimensions of selected data:', tdata.shape)

            
            # Frequencies, times (concatenate, then downsample)
            fval = hdu[BEAM][csubband]['astronomy_data']['frequency'][0][start_ch:end_ch]
            fval = fval[::xskip]
            lfval = len(fval)
            cint_times = cint_times[::yskip]
            print(tdata.shape)
            print(cint_times.shape)
            
            # Axis coordinate limits 

            extpix = [0, tdata.shape[0], 0, tdata.shape[1]]
            extcor = [fval[0], fval[-1], 0, ypix*cint_times[0]]
            #print 'Data array (pixels):', extpix
            #print '....in image units:', extcor
            
        # output hdf file?
            if output != "none":
                ind = 0
                print("Output file: {}".format(outfiles[ifile]))
                if (isub == asub) and (pol == 0):
                    if os.path.exists(outfiles[ifile]):
                        print('NOTE: appending/updating existing data output file')
                        hduo = h5py.File(outfiles[ifile], 'r+')
                        if BEAM in list(hduo.keys()):
                            hduo_0 = hduo[BEAM]
                        else:
                            hduo_0 = hduo.create_group(BEAM)
                            if 'metadata' in hdu[BEAM]:
                                hdu.copy(BEAM+'/metadata', hduo_0)
                    else:
                        hduo = h5py.File(outfiles[ifile], 'w')
                        hduo_0 = hduo.create_group(BEAM)
                        hdu.copy('metadata', hduo)
                        hdu.copy('configuration', hduo)
                        if 'metadata' in hdu[BEAM]:
                            hdu.copy(BEAM+'/metadata', hduo_0)
                    if ('proc_history' in hduo['metadata']):
                        lh = hduo['metadata']['proc_history'].shape[0]
                        hduo['metadata']['proc_history'].resize((lh+1,))
                        hduo['metadata']['proc_history'][lh] = history
                    else:
                        hduo['metadata'].create_dataset('proc_history', data=np.array([history],dtype='|S256'), maxshape=(None,), chunks=True)
                if pol == 0:
                    if (csubband in hduo_0):
                        print("ERROR: {} and {} already exist in {}".format(BEAM, csubband, outfiles[ifile]))
                        hduo.close()
                        exit()
                    else:
                        hduo_0SB = hduo_0.create_group(csubband)
                        # Pick off correct positions and other metadata
                        if 'metadata' in hdu[BEAM][csubband]:
                            if (ndrive == 0) and (yskip==1):
                                hdu.copy(BEAM+"/"+csubband+'/metadata', hduo_0SB)
                            else:
                                hduo_0SB.create_dataset('metadata/observation_parameters', data=hdu[BEAM][csubband]['metadata/observation_parameters'][DRIVE!=0][::yskip])
                        #hdu.copy(BEAM+'/'+csubband+'/metadata', hduo_0SB)
                        hduo_0SBm = hduo_0SB['metadata']
                        hduo_0SBa = hduo_0SB.create_group('astronomy_data')
                        # Use rows for frequency, rather than columns, for PAF data
                        hduo_0SBa.create_dataset('frequency', data=fval.reshape(1,lfval), maxshape=(1,None,), chunks=True)
                        # Prepare correct array dimensions (similar input and output dimensionality)
                        out_shape = [1]*ndims
                        out_shape[-3] = apol
                        out_shape[-2] = tdata.shape[1]
                        out_shape[0] = tdata.shape[0]
                        max_shape = [1]*ndims
                        max_shape[-2] = None
                        max_shape[-3] = 2
                        max_shape[0] = None
                        # NB won't be able to mix pre and post-2.4 SDHDF data in the same output file due to different dimensionality
                        hduo_0SBa.create_dataset('data', data=np.zeros(out_shape), maxshape=tuple(max_shape), chunks=True)

            # Centre frequency in MHz (can be changed to peak freq if using an HI Galactic calibrator)
            midfreq = (fval[0] + fval[-1]) / 2.0
                    
            # Determine if there are drive errors
            if (len(DRIVE) > 1):
                ndrive = np.sum(DRIVE==0)
                print("{} valid integrations found".format(tdata.shape[0]))
                print("{} integrations with drive errors (rejected)".format(ndrive))
                if ndrive == tdata.shape[0]:
                    exit("ERROR: no valid data")
            else:
                ndrive = 0
                DRIVE = 1

            print(BEAM)
            
            # Remove drive errors (already removed above)            
            qdata = tdata
            
            # Spectral baseline (compact sources)
            if method[:4]=='none':
                vref = np.average(qdata, axis=0, weights=cint_times)
            else:
                if method[:4]=='medi':
                    vref = np.median(qdata, axis=0)
                else:
                    vref = np.average(qdata, axis=0, weights=cint_times)
                # Grow ref spectrum back to 2D array
                vrefim = np.repeat(vref, repeats=qdata.shape[0])
                vrefim = np.reshape(vrefim, [qdata.shape[1], qdata.shape[0]])
                vrefim = np.transpose(vrefim)
                # Quotient (multiplicative variance)
                qdata = (np.divide(qdata, vrefim)-1.0)*np.average(vref)

            # Non-quotient time average baseline (compact sources)
            if method[4:]=='none':
                tref = np.average(qdata, axis=1)
            else:
                if method[4:]=='medi':
                    tref = np.median(qdata, axis=1)
                else:
                    tref = np.average(qdata, axis=1)
                # Grow ref spectrum back to 2D array of original size
                trefim = np.zeros(qdata.shape)
                trefim = np.repeat(tref, repeats=qdata.shape[1]).reshape(tref.shape[0], qdata.shape[1])
                # Subtract time baseline
                qdata -= trefim

                           
            # Plot label
            if ndrive == 1:
                derr = "; excl. {} drive errors".format(ndrive)
            elif ndrive > 1:
                derr = "; excl. {} drive errors".format(ndrive)
            else:
                derr = ""
            if qsvd:
                svdmode = "; SVD{}".format(nsvd)
            else:
                svdmode = ""
            rmode = " ("+svdmode+derr+")"

            #stats function
            def lstats(x):
                xstat = np.zeros(6)
                xstat[5] = np.argmax(x)
                xstat[4] = np.amax(x)
                xstat[3] = np.percentile(x,95)
                xstat[2] = np.percentile(x,50)
                xstat[1] = np.percentile(x,5)
                xstat[0] = np.amin(x)
    #            print 'Data (before calibration): min, h1, median, h3, max:\n', xstat
                return xstat

            # Apply calibration                    
            if calin != "none":
                # Check if BEAM exists
                if not (CALBEAM in calparams):
                    if qcalin:
                        exit("ERROR: missing calibration parameters for "+CALBEAM)
                    else:
                        CALBEAM = list(calparams)[0]
                        print("NOTIFICATION: forcing",CALBEAM,"calibration solution for",BEAM)
                try:
                    scale = calparams[CALBEAM][poldict]['band_SB'+str(isub)]['scale'][pol]
                    tcal = calparams[CALBEAM][poldict]['band_SB'+str(isub)]['tcal'][pol]
                except:
                    exit("ERROR: missing calibration parameters for this subband or polarisation")
            else:
                scale = 1.0
            if scale != 1.0:
                if 'fluxunit' in calparams[CALBEAM][poldict]['band_SB'+str(isub)]:
                    dunit = calparams[CALBEAM][poldict]['band_SB'+str(isub)]['fluxunit']
                else:
                    dunit = "Jy"
                print("Scaling factor to {}: {:.4e}".format(dunit, scale))
                    
            # Image and spectrum statistics      
            istat = lstats(qdata)*scale
            tstat = lstats(tdata)*scale

            # Multibeam cache
            if (ibeam == aBeam) and (pol == 0):
                ind0 = [0]*(ndims+1)
                ind0[0] = nbeam
                ind0[1] = qdata.shape[0]
                ind0[-3] = npol
                ind0[-2] = qdata.shape[1]
                ind0[-1] = tpix
                mtdata = np.zeros(tuple(ind0))  
                              
            # Prepare slice array
            ind0[0] = ibeam-aBeam
            ind0[2] = pol
            for i in [1,3,4]:
                ind0[i] = slice(None)
            if(tpix == 1):
                ind0[-1] = 0

            # Copy data
            mtdata[tuple(ind0)] = qdata*scale
    
        # end of polarisation loop
    # End of beam loop

    # Fake 2D Gaussian in centre of middle beam with this width (sigma as a percentage of axis dimension) and amplitude given by user
    if fake == "HI":
        print("Inserting fake source...")
        width = 0.03
        x0 = int(mtdata.shape[1]/2.0)
        y0 = int(mtdata.shape[3]/2.0)
        z0 = int(fbeam)
        sx = width*mtdata.shape[1]
        sy = width*mtdata.shape[3]
        sz = max(1,width*nbeam)
        ix = int(5*sx)
        iy = int(5*sy)
        iz = int(5*sz)
        #print(x0,y0,z0)
        #print(ix,iy,iz)
        #amp = fakeamp*stats.median_abs_deviation(mtdata[z0-iz:z0+iz,x0-ix:x0+ix,:,y0-iy:y0+iy,0],axis=None)
        for pol in range(apol):
            for j in range(-iz,iz):
                if (z0+j >= 0) and (z0+j < nbeam):
                    for i in range(-ix,ix):
                        mtdata[z0+j,x0+i,pol,y0-iy:y0+iy,0] += fakeamp*np.exp(-(j**2)/(2.0*sz**2))*np.exp(-(i**2)/(2.0*sx**2))*np.exp(-(np.arange(-iy,iy)**2)/(2.0*sy**2))
    # Fake intensity map
    if fake == "IM":
        immodel = np.zeros(mtdata.shape)
        x0 = int(mtdata.shape[1]/2.0)
        y0 = int(mtdata.shape[3]/2.0)
        IM = np.load(fakeIMfile)
        if IM['ddim'][2] < 0.0:
            exit("ERROR: frequency spacing needs to be positive for intensity map")
        # Artificially assign a slice at constant Dec to each beam, map the RA axis to the time axis, but map frequencies properly
        print("Intensity map dimensions (RA, Dec, Freq):", IM['sky'].shape)
        print("Adding intensity map...")
        imrati = IM['sky'].shape[1]/float(mtdata.shape[0])
        imratj = IM['sky'].shape[0]/float(mtdata.shape[1])
        #imdk = abs(fval[1]-fval[0]/IM['ddim'][2])
        #imzlim = IM['sky'].shape[2]
        fvalim = f0/(1.0+IM['zlim'][1])+IM['ddim'][2]*np.arange(IM['sky'].shape[2])
        # Performance measure
        start = time.time()
        for pol in range(apol):
            for i in range(mtdata.shape[0]):
                imy = int(i*imrati)
                for j in range(mtdata.shape[1]):
                    imx = int(j*imratj)
                    #dimz = [int(x) for x in (range(mtdata.shape[3])*imdk % imzlim)]
                    #immodel[i,j,pol,:,0] += fakeamp*IM['sky'][imx,imy,dimz]
                    immodel[i,j,pol,:,0] += fakeamp*np.interp(fval, fvalim, IM['sky'][imx,imy,:], left=0.0, right=0.0)
        mtdata += immodel
        end=time.time()
        print("IM addition time was {:.1f} sec".format(end-start))
            
    # Stacked singular value decomposition
    d = mtdata.shape
    print("Cached {:.1f} MB".format(64e-6*np.prod(d)))
    svdnorm = np.median(np.abs(mtdata))
    # Performance measure
    start = time.time()
    svdmedian = np.median(mtdata)
    svdMAD = np.median(np.abs(mtdata-svdmedian))
    if qsvd:
        nnan = sum(np.isnan(mtdata.flatten()))
        if nnan > 0:
            print("WARNING: {} nans detected - SVD may not be possible".format(nnan))
        if SVDmethod == "SVD":
            # Normal L2 SVD
            for j in range(mtdata.shape[0]):
                X = mtdata[j,:,:,:,:].reshape(np.prod(d[1:3]), np.prod(d[3:]))
                Xsvdmedian = np.median(X)
                XsvdMAD = np.median(np.abs(X-Xsvdmedian))
                U, S, Vt = np.linalg.svd((X-Xsvdmedian)/XsvdMAD, full_matrices=False)
                RI = np.matrix(U[:,:nsvd]) * np.diag(S[:nsvd]) * np.matrix(Vt[:nsvd,:])
                X -= np.array(RI)*XsvdMAD+Xsvdmedian
                mtdata[j,:,:,:,:] = X.reshape(d[1],d[2],d[3],d[4])
        elif SVDmethod == "CSVD":
            # Clipped L2 SVD
            for j in range(mtdata.shape[0]):
                X = mtdata[j,:,:,:,:].reshape(np.prod(d[1:3]), np.prod(d[3:]))
                Xsvdmedian = np.median(X)
                XsvdMAD = np.median(np.abs(X-Xsvdmedian))
                RI = np.zeros(X.shape)
                # Three rounds of 3-sigma clips / L2 SVD
                for i in range(3):
                    DIFF = (X-Xsvdmedian)/XsvdMAD - np.array(RI)
                    Du = np.percentile(DIFF,99.5)
                    Dl = np.percentile(DIFF,0.5)
                    X[DIFF>Du] = XsvdMAD*(Du + np.array(RI)[DIFF>Du])+Xsvdmedian
                    X[DIFF<Dl] = XsvdMAD*(Dl + np.array(RI)[DIFF<Dl])+Xsvdmedian
                    #print("Clipped L2 SVD, iteration {}...".format(i))
                    U, S, Vt = np.linalg.svd((X-Xsvdmedian)/XsvdMAD, full_matrices=False)
                    RI = np.matrix(U[:,:nsvd]) * np.diag(S[:nsvd]) * np.matrix(Vt[:nsvd,:])
                    X -= np.array(RI)*XsvdMAD+Xsvdmedian
                # Replace data array with clipped and SVD-subtracted version
                mtdata[j,:,:,:,:] = X.reshape(d[1],d[2],d[3],d[4])
        elif SVDmethod == "L1SVD":
            # Robust L1 SVD (v.slow) - https://ieeexplore.ieee.org/abstract/document/1467342
            print("Calculating robust (L1) SVD...")
            for j in range(mtdata.shape[0]):
                X = mtdata[j,:,:,:,:].reshape(np.prod(d[1:3]), np.prod(d[3:]))
                Xsvdmedian = np.median(X)
                XsvdMAD = np.median(np.abs(X-Xsvdmedian))
                U, S, V, Xlow = l1svd_error_minimization((X-Xsvdmedian)/XsvdMAD, nsvd, 10, 0.001, 1) # https://ieeexplore.ieee.org/abstract/document/1467342 (also see https://arxiv.org/pdf/2210.12097)
                X -= np.array(Xlow)*XsvdMAD+Xsvdmedian
                mtdata[j,:,:,:,:] = X.reshape(d[1],d[2],d[3],d[4])
        elif SVDmethod == 'CSVDstack':
            # Beam*time*pol v freq
            X = (mtdata.reshape(np.prod(d[0:3]), np.prod(d[3:]))-svdmedian)/svdMAD
            RI = np.zeros(X.shape)
            MASK = np.ones(X.shape)
            for i in range(2):
                print("Clipped 3D stacked SVD decomposition, iteration {}...".format(i))
                DIFF = X - np.array(RI)
                # Identify outliers 
                Du = np.percentile(DIFF,99.5)
                Dl = np.percentile(DIFF,0.5)
                X[DIFF>Du] = Du + np.array(RI)[DIFF>Du]
                X[DIFF<Dl] = Dl + np.array(RI)[DIFF<Dl]
                #X[np.logical_or(DIFF>np.percentile(DIFF,99.5), DIFF<np.percentile(DIFF,0.5))] = 0.0
                #remask(0.0, X)
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                RI = np.matrix(U[:,:nsvd]) * np.diag(S[:nsvd]) * np.matrix(Vt[:nsvd,:])
            # NB: this DOES NOT replace data array with clipped version. Better to use this during next edit:
            #X -= svdMAD*np.array(RI)+svdmedian
            #mtdata = X.reshape(d)
            mtdata -= svdMAD*np.array(RI).reshape(d) + svdmedian
        elif SVDmethod == 'TuckerSVD':
    # Tucker tensor decomposition with outlier rejection
            nt = nsvd
            dim = [d[0], np.prod(d[1:3]), np.prod(d[3:])]
            print(dim)
            rdim = [min(dim[0],nt), min(dim[1],nt), min(dim[2],nt)]
            X = (mtdata.reshape(dim[0], dim[1], dim[2])-svdmedian)/svdMAD
            Xtl = tl.tensor(X)
            TUCK = np.zeros(X.shape)
            MASK = np.ones(X.shape)
            random_state = 12345
            # Rank of the Tucker decomposition
            tucker_rank = [min(nt,dim[0]), min(nt,dim[1]), min(nt,dim[2])]
            print("Tucker core tensor rank = ", tucker_rank)
            # Rounds of 3-sigma clips - use Tucker mask to censor data
            for i in range(2):
                print("Censored Tucker tensor decomposition, iteration {}...".format(i))
                DIFF = X - np.array(TUCK)
                # Identify outliers on the Tucker mask
                Du = np.percentile(DIFF,99.5)
                Dl = np.percentile(DIFF,0.5)
                MASK[DIFF>Du] = 0.0
                MASK[DIFF<Dl] = 0.0
                Mtl = tl.tensor(MASK)
                # Tucker decomposition
                core, tucker_factors = tucker(Xtl, rank=tucker_rank, mask=Mtl, init="random", tol=1e-5, random_state=random_state)
                TUCK = tl.to_numpy(tl.tucker_to_tensor((core, tucker_factors)))
            # NB: this DOES NOT replace data array with clipped version. Better to use this during next edit:
            #X -= svdMAD*np.array(RI)+svdmedian
            #mtdata = X.reshape(d)
            mtdata -= svdMAD*TUCK.reshape(d) + svdmedian
        elif SVDmethod == 'CPSVD':
            dim = [d[0], np.prod(d[1:3]), np.prod(d[3:])]
            print(dim)
            X = (mtdata.reshape(dim[0], dim[1], dim[2])-svdmedian)/svdMAD
            Xtl = tl.tensor(X)
            CP = np.ones(X.shape)
            MASK = np.ones(X.shape)
            # Rounds of 3-sigma clips - use mask to censor data
            for i in range(2):
                print("Censored CP tensor decomposition, iteration {}...".format(i))
                DIFF = X - np.array(CP)
                # Identify outliers on the CP mask
                Du = np.percentile(DIFF,99.5)
                Dl = np.percentile(DIFF,0.5)
                MASK[DIFF>Du] = 0.0
                MASK[DIFF<Dl] = 0.0
                Mtl = tl.tensor(MASK)
                # Perform the CP decomposition
                weights, factors = parafac(Xtl, rank=nsvd, mask=Mtl, init="random", tol=10e-6)
                # Reconstruct the image from the factors
                CP = tl.to_numpy(tl.cp_to_tensor((weights, factors)))
            # NB: this DOES NOT replace data array with clipped version. Better to use this during next edit:
            #X -= svdMAD*np.array(CP)+svdmedian
            #mtdata = X.reshape(d)
            mtdata -= svdMAD*CP.reshape(d) + svdmedian
        else:
            exit("ERROR: no matching SVD method found")
        end=time.time()
        print("SVD execution (n={}) time was {:.3f} sec".format(nsvd, end-start))

    # If a fake signal was added, measure its post-SVD amplitude (for higher S/N ratios, this might be a useful measure of signal loss)
    if fake == "HI":
        msx = max(1, int(sx/4))
        msy = max(1, int(sy/4))
        mamp = np.average(mtdata[z0,x0-msx:x0+msx,:,y0-msy:y0+msy,0])
        mamp2 = mamp-np.median(mtdata)
        print("Post-SVD fake source amplitude = {:.3e}".format(mamp))
        print("Post-SVD fake source amplitude-median = {:.3e}".format(mamp2))
        print("Fake output/input = {:.3}".format(mamp/fakeamp))
        print("Fake output-median/input = {:.3}".format(mamp2/fakeamp))
    if fake == "IM":
        # Correlation coefficient
        mthi=np.percentile(mtdata, 99.5)
        mtlo=np.percentile(mtdata, 0.5)
        mtcopy=mtdata*1.0
        mtcopy[mtcopy>mthi] = mthi
        mtcopy[mtcopy<mtlo] = mtlo
        mtmean=np.average(mtcopy)
        immean=np.average(immodel)
        rho = np.sum((mtcopy-mtmean)*(immodel-immean))/np.sqrt(np.sum(np.square(mtcopy-mtmean))*(np.sum(np.square(immodel-immean))))
        print("Post-SVD 99.5% partial Winsorized correlation = {:.4f}".format(rho))

        # Model power spectrum
        print("Data dimensions:", mtcopy.shape)
        ftmodel = np.fft.fftn(immodel, axes=[0,1,3])
        # Auto power spectrum
        ftauto = np.fft.fftn(mtcopy, axes=[0,1,3])
        # Cross-power spectrum
        ftcross = ftauto*ftmodel
        # Shift to centre
        ftmodel = np.fft.fftshift(ftmodel)
        ftauto = np.fft.fftshift(ftauto)
        ftcross = np.fft.fftshift(ftcross)
        # Square the model and data to give power
        ftmodel = np.square(ftmodel)
        ftauto = np.square(ftauto)
        a = ftcross.shape
        print("Complex cross-power spectrum dimensions:", a)

        # Calculate pixel scale at central redshift in Mpc in each dimension, assuming intensity map fills beam and time dimensions
        zmean = f0/np.average(fval)-1
        if zmean < 0:
            zmean = np.average(IM['zlim'])
        zmean=0.12
        fdelta = f0/(1+zmean)+fval[0]-fval[1]
        zdelta = np.abs(zmean - f0/fdelta +1)
        print("Estimating pixel sizes in cMpc at z={:.3f}:".format(zmean))
        phdim = np.ones(5)
        # Frequency axis
        phdim[3] = np.abs(cosmo.comoving_distance(zmean).value-cosmo.comoving_distance(zmean+zdelta).value)
        # Time/RA axis
        phdim[1] = cosmo.comoving_distance(zmean).value*np.sin(np.pi*np.abs(IM['xlim'][1]-IM['xlim'][0])*np.cos(np.pi*np.average(IM['xlim'])/180.0)/ (180.0*mtcopy.shape[1])) #/(1+zmean)
        # Beam/Dec axis [dividing comoving distance by (1+z) again means this is the physical distance, same as above, not comoving] <- CHANGED BACK TO COMOVING
        phdim[0] = cosmo.comoving_distance(zmean).value*np.sin(np.pi*np.abs(IM['ylim'][1]-IM['ylim'][0])/(180.0*mtcopy.shape[0])) #/(1+zmean)
        print(phdim)
        # Create an array of pixel coordinates (adds an extra dimension of length 5)
        icross = np.mgrid[:a[0],:a[1],:a[2],:a[3],:a[4]]
        # Centre pixels (integer truncation to emulate Fourier transform)
        icen = np.int_(np.array(list(a))/2)
        print("Zero-k voxel: {}".format(icen))
        # Change back to float
        xcross = icross.astype(float)
        # Distance of each pixel from centre pixel
        for i in range(icross.shape[0]):
            xcross[i,:,:,:,:,:] -= icen[i]
            xcross[i,:,:,:,:,:] *= 2.0*np.pi/(hPlanck*phdim[i]*float(a[i])) # Convert distances from comoving Planck18 to comoving h=1.
        # Euclidean distances
        dpix = np.sqrt(np.sum(np.square(xcross), axis=0))
        print("Minimum/maximum wavenumbers (log(h/cMpc) at mean redshift): {:.4f}, {:.4f}".format(np.log10(np.min(dpix[dpix!=0])), np.log10(np.max(dpix[dpix!=0]))))
        # Total power
        print("Gridded model flux = {:.4e}".format(np.sum(immodel)))
        mtmax = np.max(np.abs(ftmodel))
        ftmax = np.max(np.abs(ftcross))
        mtwhere = np.argwhere(np.abs(ftmodel)==mtmax)
        ftwhere = np.argwhere(np.abs(ftcross)==ftmax)
        mtmax /= fakeamp**2
        ftmax /= fakeamp**2
        print("Maximum model power is {:.4f} at voxel {}".format(mtmax, mtwhere))
        print("Maximum cross-power is {:.4f} at voxel {}".format(ftmax, ftwhere))
        # Use the real component for the histogram weights (mtcopy and immodel are real, so the -ve and +ve imaginary components in their FTs will cancel anyway)
        mweights = np.real(ftmodel[dpix!=0.0])/fakeamp**2
        weights = np.real(ftcross[dpix!=0.0])/fakeamp**2
        aweights = np.real(ftauto[dpix!=0.0])/fakeamp**2
        # Histogram (in log space) the real part of the cross-power spectrum (remove dpix=0)
        bins = 10
        PSnum, edges = np.histogram(np.log10(dpix[dpix!=0.0]), bins=bins)
        mPS, edges = np.histogram(np.log10(dpix[dpix!=0.0]), bins=bins, weights=mweights)
        PS, edges = np.histogram(np.log10(dpix[dpix!=0.0]), bins=bins, weights=weights)
        aPS, edges = np.histogram(np.log10(dpix[dpix!=0.0]), bins=bins, weights=aweights)
        mPS /= PSnum
        PS /= PSnum
        aPS /= PSnum
        mPSerr, edges = np.histogram(np.log10(dpix[dpix!=0.0]), bins=bins, weights=np.square(mweights))
        mPSerr = np.sqrt(mPSerr)/PSnum
        PSerr, edges = np.histogram(np.log10(dpix[dpix!=0.0]), bins=bins, weights=np.square(weights))
        PSerr = np.sqrt(PSerr)/PSnum
        aPSerr, edges = np.histogram(np.log10(dpix[dpix!=0.0]), bins=bins, weights=np.square(aweights))
        aPSerr = np.sqrt(aPSerr)/PSnum
        # Setup and normalise for plot and file output
        if qsvd:
            label = "HI power spectrum ({}{})".format(SVDmethod, nsvd)
        else:
            label = "HI power spectrum (no t-SVD)"
        xaxis = 0.5*(edges[1:]+edges[:-1])
        xzero = 0.5*(3*edges[0]-edges[1])
        dPS = 1.0e6*np.abs(PS)*10**(3*xaxis)/(2*np.pi**2)
        dmPS = 1.0e6*np.abs(mPS)*10**(3*xaxis)/(2*np.pi**2)
        daPS = 1.0e6*np.abs(aPS)*10**(3*xaxis)/(2*np.pi**2)
        dPSerr = 1.0e6*PSerr*10**(3*xaxis)/(2*np.pi**2)
        dmPSerr = 1.0e6*mPSerr*10**(3*xaxis)/(2*np.pi**2)
        daPSerr = 1.0e6*aPSerr*10**(3*xaxis)/(2*np.pi**2)
        # Plot
        if not nodisplay:
            plt.rcParams['lines.linewidth'] = 2
            # x offset for model power
            delta = 0.02
            # Plot y limits
            ylim = [6e-1, 200000] # [6e-3, 20]
            plt.figure('Cross-power spectrum', figsize=[6.4, 4.8])
            plt.rc('font',family='serif')
            colour = np.array(['blue']*len(PS))
            # Power spectrum (cross-power, model power and auto power)
            #plt.scatter(10.0**(xaxis+2*delta), daPS, color='green', edgecolors='none', label=label)
            plt.scatter(10.0**(xaxis+delta), dmPS, color='red', edgecolors='none', alpha=0.3, label="G23 power spectrum")
            plt.scatter(10.0**xaxis, dPS, color='blue', label="HI-G23 cross-power")
            # Error estimates for above (from sample variance)
            #plt.errorbar(10.0**(xaxis+2*delta), daPS, yerr=daPSerr, capsize=6, fmt="none", color='g')
            plt.errorbar(10.0**(xaxis+delta), dmPS, yerr=dmPSerr, capsize=6, fmt="none", color='r', alpha=0.3)
            plt.errorbar(10.0**xaxis, dPS, yerr=dPSerr, capsize=6, fmt="none", color='b')
            # Fill between
            ylo = dmPS-dmPSerr
            yhi = dmPS+dmPSerr
            ylo[np.argwhere(ylo<ylim[0])] = ylim[0]
            xs = np.arange(xaxis[0]+delta, xaxis[-1]+delta, 0.01)
            cs = np.polyfit(xaxis+delta, np.log10(yhi), 7)
            yhi = np.polyval(cs, xs)
            cs = np.polyfit(xaxis+delta, np.log10(ylo), 8)
            ylo = np.polyval(cs, xs)
            plt.fill_between(10**xs, 10**ylo, 10**yhi, color='r', alpha=0.02)
            # Plot zero-k power to left of leftmost point
            plt.scatter(10.0**xzero, 1.0e6*ftmax*10**(3*xzero)/(2*np.pi**2), color='blue', marker="<")
            plt.errorbar(10.0**xzero, 1.0e6*ftmax*10**(3*xzero)/(2*np.pi**2), xerr=0.1, xuplims=True, color='b')
            plt.scatter(10.0**(xzero+delta), 1.0e6*mtmax*10**(3*xzero)/(2*np.pi**2), color='red', marker="<", edgecolors='none', alpha=0.3)
            plt.errorbar(10.0**(xzero+delta), 1.0e6*mtmax*10**(3*xzero)/(2*np.pi**2), xerr=0.1, xuplims=True, color='r', alpha=0.3)
            # Labels
            plt.xlabel(r'$k$ ($h$ Mpc$^{-1}$)', fontsize=18)
            plt.ylabel(r'$\Delta^2 (k)$ (mK$^2$)', fontsize=18)
            plt.ylim(ylim)
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(loc=2, prop={'size': 12})
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.show()
            plt.close()
        # Dump some results in a csv file (append)
        g = open('hosvd_IM_results_tmp.csv', 'a')
        gcsv = csv.writer(g, delimiter=';', quotechar='"')
        g.write("sep=;\n")
        gcsv.writerow([history])
        gcsv.writerow(["Frequency range:", fval[0], fval[-1]])
        gcsv.writerow(["fakeamp:", fakeamp])
        gcsv.writerow([label])
        gcsv.writerow(["Cross-power spectrum:"])
        gcsv.writerow(dPS)
        gcsv.writerow(["Cross-power error:"])
        gcsv.writerow(dPSerr)
        gcsv.writerow(["Model power spectrum:"])
        gcsv.writerow(dmPS)
        gcsv.writerow(["Model power error:"])
        gcsv.writerow(dmPSerr)
        gcsv.writerow(["log k values:"])
        gcsv.writerow(xaxis)
        g.close()
         
    # Measure bespoke rms if no fake source (dodges local HI in SB_0 and NGC 6744 in SB_1)
    if (fake == "HI") and (fakeamp == 0.0):
        if isub == 0:
            clo = y0-22
            chi = y0+3
        else:
            clo = y0-3
            chi = y0+22
        rms = np.std(mtdata[z0,x0,:,clo:chi,0])
        print("rms for beam_{}, cycle_{}, frequency range {:.1f} to {:.1f} MHz = {:.3e} Jy".format(fbeam, x0, fval[clo], fval[chi], rms))
    
    # Write data
    if output != "none":
        print("Writing data...")
        # Prepare slice array
        indo = [0]*ndims
        indo[0] = slice(None)
        indo[-3] = slice(None)
        indo[-2] = slice(None)
        indo[-1] = slice(None)
        for ibeam in range(aBeam, bBeam):
            BEAM = "beam_{:02d}".format(ibeam)
            hduo[BEAM][csubband]['astronomy_data']['data'][tuple(indo)] = mtdata[ibeam,:,:,:].reshape(out_shape)

    # Histogram and waterfall images (post-HOSVD)  
    if not nodisplay:
        if zoom:
            plt.rcParams['font.size'] = 20
            plt.rcParams['axes.labelsize'] = 'x-large'
            plt.rcParams['axes.titlesize'] = 'x-large'
            plt.rcParams['legend.fontsize'] = 'x-large'
            plt.rcParams['lines.linewidth'] = 2
        # rms
        print("rms = {:.3g}".format(np.std(mtdata.flatten())))
        # Save the data for tests if size small enough
        pixlimit = 100e6
        tmpsave = "mtdata_tmp.npy"
        if np.size(mtdata) < pixlimit:
            print("Saving data in temporary file:", tmpsave)
            np.save(tmpsave, mtdata)
        else:
            print("Data size over pixel limit ({}) for temporary saves...".format(pixlimit))
        # Histogram the flux density
        plt.figure('Flux density histogram', figsize=[6.4, 4.8])
        plt.rc('font',family='serif')
        plt.hist(mtdata.flatten(), bins=100)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
        plt.close()
        # Waterfalls
        for ibeam in range(aBeam,bBeam):
            for pol in range(apol):
                # Just display fake beam
                if (fake == "none") or (ibeam == fbeam):
                    istat = lstats(mtdata[ibeam-aBeam,:,pol,:,0])
                    plt.figure(file+':'+str(pol))
                    plt.rc('font',family='serif')
                    if (uylim != 0.0):
                        istat[3] = uylim
                    if (dylim != 0.0):
                        istat[1] = dylim
                    plt.imshow(mtdata[ibeam-aBeam,:,pol,:,0],cmap=plt.cm.afmhot,aspect='auto',origin='lower', extent=[fval[0],fval[-1],0,cint_times[0]*mtdata.shape[1]], vmin=istat[1],vmax=istat[3], interpolation='gaussian')
                    plt.xlabel('Frequency (MHz)')
                    plt.ylabel('Time (s)')
                    if zoom:
                        plt.tight_layout()
                    else:
                        ax = plt.twiny()
                        plt.xlim(plot_start_ch, plot_end_ch)
                        plt.colorbar(location="right")
                        plt.title("Waterfall (beam_{} {} SVD{})".format(ibeam, csubband, nsvd))
                    plt.show(block=block)
                    if not block:
                        plt.pause(0.5)
                    plt.close()
        for pol in range(apol):
            # Just display fake beam
            if (fake != "none"):
                istat = lstats(mtdata[:,x0,pol,:,0])
                plt.figure(file+':'+str(pol))
                plt.rc('font',family='serif')
                if (uylim != 0.0):
                    istat[3] = uylim 
                if (dylim != 0.0):
                    istat[1] = dylim
                plt.imshow(mtdata[:,x0,pol,:,0],cmap=plt.cm.afmhot,aspect='auto',origin='lower', extent=[fval[0],fval[-1],aBeam,bBeam], vmin=istat[1],vmax=istat[3], interpolation='gaussian')
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('Beam')
                if zoom:
                    plt.tight_layout()
                else:
                    ax = plt.twiny()
                    plt.xlim(plot_start_ch, plot_end_ch)
                    plt.colorbar(location="right")
                    plt.title("Waterfall (cycle_{} {} SVD{})".format(x0, csubband, nsvd))
                plt.show(block=block)
                if not block:
                    plt.pause(0.5)
                plt.close()
        for pol in range(apol):
            # Just display fake beam
            if (fake != "none"):
                istat = lstats(mtdata[:,:,pol,y0,0])
                plt.figure(file+':'+str(pol))
                plt.rc('font',family='serif')
                if (uylim != 0.0):
                    istat[3] = uylim 
                if (dylim != 0.0):
                    istat[1] = dylim
                plt.imshow(mtdata[:,:,pol,y0,0],cmap=plt.cm.afmhot,aspect='auto',origin='lower', extent=[0,cint_times[0]*mtdata.shape[1],aBeam,bBeam], vmin=istat[1],vmax=istat[3], interpolation='gaussian')
                plt.xlabel('Time (s)')
                plt.ylabel('Beam')
                if zoom:
                    plt.tight_layout()
                else:
                    ax = plt.twiny()
                    plt.xlim(plot_start_ch, plot_end_ch)
                    plt.colorbar(location="right")
                    plt.title("Waterfall (frequency {:.1f}MHz {} SVD{})".format(fval[y0], csubband, nsvd))
                plt.show(block=block)
                if not block:
                    plt.pause(0.5)
                plt.close()
    
    hdu.close()
    if output != "none":
        hduo.close()
# End of file loop
exit()

