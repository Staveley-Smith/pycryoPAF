#!/usr/bin/env python
#
# Example useage: 
# python paf-scan.py -s testsvd.hdf5 -i CSVD-5

# Purpose:
# Flattens sky scans taken using the Parkes cryopaf using various optional techniques, including a 
# reference spectral bandpass, mean/median residual bandpass removal, time series flattening, polynomials,
# and various SVD options.

# Lister Staveley-Smith

# Libraries
import sys
import re
import numpy as np
from scipy import signal
from scipy import fftpack
from scipy.interpolate import interp1d
import h5py
import getopt
import json
import os
import glob
import pkg_resources
# NEED conda install -c conda-forge cvxpy
from l1csvd import l1csvd
from utils import *
import time

# HI rest frequency
f0 = 1420.4058

# Initialise calparams
calparams = {}

# Process command line parameters

ref = 'none'
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
calin = "none"
qcalin = True
nodisplay = False
block = True
output = "none"
nsvd = 1
smooth_par = "1,1"
qsmooth= False
zoom = False
textind = 0
method = "nonenone"
methodlist = ['medi', 'mean', 'none']
qsvd = False
SVDmethodlist = ['SVD-n', 'CSVD-n', 'L1SVD-n']
fake = "none"
fakestr = "none"
fakeamp = 1.0
history =""

optionshort = '?a:b:A:B:d:e:F:h:H:im:nNo:p:r:s:u:v:w:zS:x:X:'
optionlist = ['?','a=', 'b=', 'A=', 'B=', 'd=', 'e=', 'F=', 'h=', 'H=', 'i', 'method=', 'n', 'N', 'o=', 'p=', 'ref=', 'sig=', 'u=', 'v=', 'w=', 'zoom', 'SVD=', 'x=', 'X=']
optionhelp = ['help', 'start sub-band', 'end sub-band', 'start beam', 'end beam', 'lower plot limit', 'number of smoothing channels in frequency(,time)', 'channel range for polynomial fit (comma-separated)', 'apply this calibration file', 'apply this calibration to other beams (first entry)', 'Stokes I', 'method for freq-time pre-conditioning (meanmean/medimedi/nonemedi etc.)', 'no plot', 'unblocked plot', 'Output data file', 'polynomial baseline order (0 and -ve values applied pre-time-average)', 'reference spectrum', 'source (signal) spectrum', 'upper plot limit', 'start channel number (within each sub-band)', 'end channel number (within each sub-band)', 'large font (for publication)', 'SVD method (SVD-n/CSVD-n,L1SVD-n, where n is number of singular values)', 'extra (fake) line (HI only)', 'extra fake line amplitude']

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
    elif opt in ('-r', '--ref'):      # filename for reference spectrum
        ref = arg
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
    elif opt in ('-x', '--x'):    # Fake line type
        fake = arg
    elif opt in ('-X', '--X'):    # Fake line amplitude factor
        fakestr = arg
    elif opt in ('-S', '--SVD'):      # SVD method
        qsvd = True
        SVDmethod = arg    
    elif opt in ('-z', '--zoom'):      # Large font
        zoom = True    

if len(remainder) != 0:
    print('Unrecognised arguments:', remainder)
    exit()

# Certain option combinations not allowed

if sig == 'none':
  exit('ERROR: need signal file (-s <hdf name>)')

if bsub == 0:
    bsub=asub+1        

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
        exit("ERROR: valid SVD methods: {}".format(SVDmethodlist))
    t = False
    for m in SVDmethodlist:
        if SVDmethod in m:
            t = True
    if SVDmethod == '' or not t:
        exit("ERROR: valid SVD methods: {}".format(SVDmethodlist))
        
if fake != "none":
    if (fake != "HI"):
        exit("ERROR: only \"-x HI\"supported")
    if fakestr != "none":
        try:
            fakeamp = float(fakestr)
        except:
            exit("ERROR: argument for -X need to be the amplitude of the fake line")
    print("NOTIFICATION: fake line amplitude = {}".format(fakeamp))
elif fakestr != "none":
    exit("ERROR: option \"-X\" is redundant without option \"-x HI\" etc (extra fake line)")

if nodisplay == False:
    from pylab import rcParams
    import matplotlib.pyplot as plt
        
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

if not nodisplay:
    plt.rcParams['figure.figsize'] = [12, 6.75]
    #plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['figure.facecolor'] = 'white'
    if zoom:
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.labelsize'] = 'x-large'
        plt.rcParams['axes.titlesize'] = 'x-large'
        plt.rcParams['legend.fontsize'] = 'x-large'
        plt.rcParams['lines.linewidth'] = 2

# Output display resolution (determines sampling interval)
yres = 2048

# Sig filename handling        
sigfiles = filenameextractor(sig)
lsigfiles = len(sigfiles)

# Ref filename handling and duplication       
if ref != "none":
    reffiles = filenameextractor(ref)
    lreffiles = len(reffiles)
    if (lreffiles == 1) and (lsigfiles != 1):
        for i in range(1,lsigfiles):
            reffiles.append(reffiles[0])
        lreffiles = len(reffiles)
    if (lreffiles != lsigfiles):
        exit("ERROR: number of ref files must be one or be equal to the number of sig files")

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
    if bBeam <= aBeam:
        # Update bBEAM/nbeam/fbeam
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
    
    # Open ref file
    if ref != 'none':
        hdur=h5py.File(reffiles[ifile], 'r')
        rsource = hdur['metadata']['beam_parameters'][0]['SOURCE']
        if isinstance(rsource,bytes):
            rsource = rsource.decode()
        print("Reference:   {} ({})".format(reffiles[ifile], rsource))
        DRIVEref = hdur[BEAM][asubband]['metadata']['observation_parameters']['DRIVE_STATUS']
        print("DRIVEref length:", len(DRIVEref))
    else:
        reffiles = [""]

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

        # Loop over subbands
        for isub in range(asub, bsub):
            csubband = "band_SB{:d}".format(isub)
            print(BEAM, csubband)
    
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
                    XXX = hdu[BEAM][asubband]['metadata']['observation_parameters']
                except:
                    RA_STR = "N/A"
                    DEC_STR = "N/A"
                    DRIVE = "N/A"

            # Reference metadata
            if ref != "none":
                try:
                    DRIVEref = hdur[BEAM][asubband]['metadata']['observation_parameters']['DRIVE_STATUS']
                except:
                    DRIVEref = "N/A"
            
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
                int_times = hdu[BEAM][csubband]['metadata']['integ_times']
                print("Processed")
            except:
                DUMP_TIME = hdu[BEAM]['metadata']['band_parameters'][0]['REQUESTED_INTEGRATION_TIME']
                N_DUMPS = hdu[BEAM]['metadata']['band_parameters'][0]['NUMBER_OF_INTEGRATIONS']
                int_times = np.array([DUMP_TIME]*data_dims[0])
            int_time = np.sum(int_times)
            print("Raw")
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
            #print('Pixels available: ({},{},{})'.format(xpix, ypix, ppix))
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
                    exit("WARNING: {} nans detected *************************************************".format(nnan))
                if (npol == 1) and (ppix > 1):
                    print("Stokes I spectrum")
                    t2data = hdu[BEAM][csubband]['astronomy_data']['data'][tuple(ind2)]
                    nnan = sum(np.isnan(t2data.flatten()))
                    if nnan > 0:
                        exit("WARNING: {} nans detected *************************************************".format(nnan))
                    tdata = (tdata + t2data)/2.0
                    rat = abs(np.median(tdata)/np.median(t2data))
                    if (rat < 0.666) or (rat > 1.5):
                        print("WARNING: significant difference between levels of two pols (factor {:.2f})".format(rat))
                        print("WARNING: Stokes I processing may not be appropriate")
                elif npol == 2:
                    if pol == 0:
                        print("Stokes XX spectrum")
                    elif (pol == 1) and (ppix > 1):
                        print("Stokes YY spectrum")

                if ref != 'none':
                    tref =  hdur[BEAM][csubband]['astronomy_data']['data'][tuple(ind1)]
                    if (npol == 1) and (ppix > 1):
                        print("Stokes I spectrum")
                        t2ref = hdur[BEAM][csubband]['astronomy_data']['data'][tuple(ind2)]
                        tref = (tref + t2ref)/2.0

                    # Figure out integration time parameters (raw data v processed data)
                    try:
                        int_times_ref = hdur[BEAM][csubband]['metadata']['integ_times'][:]
                    except:
                        DUMP_TIME_ref = hdur[BEAM]['metadata']['band_parameters'][0]['REQUESTED_INTEGRATION_TIME']
                        N_DUMPS_ref = hdur[BEAM]['metadata']['band_parameters'][0]['NUMBER_OF_INTEGRATIONS']
                        int_times_ref = np.array([DUMP_TIME_ref]*tref.shape[0])
                    int_time_ref = np.sum(int_times_ref)
                    if (len(int_times_ref) > 1):
                        tsint = np.sort(int_times_ref)
                        if (tsint[0]+tsint[-1]) != 0.0:
                            if abs(2.0*(tsint[0]-tsint[-1])/(tsint[0]+tsint[-1])) > 0.1:
                                print("WARNING: spectra (ref) found with integration times differing by > 10%")
                    
                # Average/convolve (boundary="sim" doesn't seem to work) and downsample for display only        
                tvdata = signal.convolve2d(tdata[DRIVE!=0], carray, mode="same", boundary="wrap")
                tdata = tvdata[::yskip,::xskip]
                cint_times = signal.convolve(int_times[DRIVE!=0], tarray, mode="same")
                print('Dimensions of selected data:', tdata.shape)
                if ref != 'none':
                    tvref = signal.convolve2d(tref[DRIVEref!=0], carray, mode="same", boundary="wrap")
                    tref = tvref[::yskip,::xskip]
                    cint_times_ref = signal.convolve(int_times_ref[DRIVEref!=0], tarray, mode="same")
                    
                
                # Frequencies (concatenate, then downsample)
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
                print('Dimensions of selected data:', tdata.shape)

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
                        
                qdata = tdata
                
                # Divide through by a reference if this exists (though this would not be normal for a scan)
                if ref != 'none':
                    # Determine if there are drive errors
                    if (len(DRIVEref) > 0):
                        ndriveref = np.sum(DRIVEref==0)
                        print("{} valid integrations found".format(tref.shape[0]))
                        print("{} integrations with drive errors (not averaged)".format(ndriveref))
                        if ndriveref == tref.shape[0]:
                            exit("ERROR: no valid reference data to average")
                    else:
                        ndriveref = 0
                        DRIVEref = 1
                    vref = np.average(tref, axis=0, weights=cint_times_ref)
                    vref_av = np.average(vref)
                    vtref = np.average(tref, axis=1)
                    if qdata.shape[1] != vref.shape[0]:
                        exit("ERROR: sig and ref spectra have different numbers of channels")
                    qdata = (np.divide(qdata, vref)-1.0)*vref_av
                        
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

 
                # Fake 2D Gaussian in centre of middle beam with this width (sigma as a percentage of axis dimension) and this amplitude (as a percentage of mean flux density)
                if fake == "HI":
                    if ibeam == fbeam:
                        print("Inserting fake source into this beam...")
                        width = 0.03
                        print("WARNING: applying fake signal of amp {:.1e} and time/chan width {:.1f}%".format(fakeamp, width*100))
                        x0 = int(tdata.shape[0]/2.0)
                        y0 = int(tdata.shape[1]/2.0)
                        sx = width*tdata.shape[0]
                        sy = width*tdata.shape[1]
                        ix = int(5*sx)
                        iy = int(5*sy)
                        #z0 = amp*np.mean(tdata[x0-ix:x0+ix,y0-iy:y0+iy])
                        for i in range(-ix,ix):
                             qdata[x0+i,y0-iy:y0+iy] += (fakeamp/scale)*np.exp(-(i**2)/(2.0*sx**2))*np.exp(-(np.arange(-iy,iy)**2)/(2.0*sy**2))
                
                # Clipping routine to replace masked pixels with local median (2D only) < REDUNDANT
                def remask(mask_value, DATA):
                    dshape = DATA.shape
                    if len(dshape) != 2:
                        exit("ERROR: can only unmask 2D data")
                    # identify mask indices
                    mask_indices = np.argwhere(DATA == mask_value)
                    #print("Masking {} voxels".format(len(mask_indices)))
                    for ix, iy in mask_indices:
                        # Grow boundaries if no valid pixels
                        for j in [2,4,8,16]:
                            ix1 = max(0, ix-j)
                            ix2 = min(dshape[0], ix+j)
                            iy1 = max(0, iy-j)
                            iy2 = min(dshape[1], iy+j)
                            med = np.median(DATA[ix1:ix2,iy1:iy2] != mask_value)
                            if med != 0:
                                DATA[ix,iy] = med
                                break

                # Singular value decomposition
                if qsvd:
                    nnan = sum(np.isnan(qdata.flatten()))
                    if nnan > 0:
                        print("WARNING: {} nans detected - SVD may not be possible".format(nnan))
                    # Performance measure
                    start = time.time()
                    #svdnorm = np.median(np.abs(qdata))
                    svdmedian = np.median(qdata)
                    svdMAD = np.median(np.abs(qdata-svdmedian))
                    if SVDmethod == "SVD":
                        # Normal L2 SVD
                        U, S, Vt = np.linalg.svd((qdata-svdmedian)/svdMAD, full_matrices=False)
                        print("USVt dimensions:", U.shape,S.shape,Vt.shape)
                        RI = np.matrix(U[:,:nsvd]) * np.diag(S[:nsvd]) * np.matrix(Vt[:nsvd,:])
                        qdata -= np.array(RI)*svdMAD+svdmedian
                    elif SVDmethod == "CSVD":
                        # Clipped L2 SVD
                        X = (qdata-svdmedian)/svdMAD
                        RI = np.zeros(X.shape)
                        # Three rounds of 3-sigma clips / L2 SVD
                        for i in range(3):
                            DIFF = X - np.array(RI)
                            Du = np.percentile(DIFF,99.5)
                            Dl = np.percentile(DIFF,0.5)
                            X[DIFF>Du] = Du + np.array(RI)[DIFF>Du]
                            X[DIFF<Dl] = Dl + np.array(RI)[DIFF<Dl]
                            #X[:,:][np.logical_or(DIFF[:,:]>np.percentile(DIFF[:,:],99.5), DIFF[:,:]<np.percentile(DIFF[:,:],0.5))] = 0.0
                            #remask(0.0, X)
                            print("Clipped L2 SVD, iteration {}...".format(i))
                            U, S, Vt = np.linalg.svd(X, full_matrices=False)
                            RI = np.matrix(U[:,:nsvd]) * np.diag(S[:nsvd]) * np.matrix(Vt[:nsvd,:])
                        print("USVt dimensions:", U.shape,S.shape,Vt.shape)
                        # Replace data array with clipped version.
                        qdata = svdMAD*(X-np.array(RI))+svdmedian
                        #qdata -= np.array(RI)*svdMAD+svdmedian
                    elif SVDmethod == "L1SVD":
                        # Robust L1 SVD (v.slow) - https://ieeexplore.ieee.org/abstract/document/1467342
                        print("Calculating robust (L1) SVD...")
                        U, S, V, Xlow = l1svd_error_minimization((qdata-svdmedian)/svdMAD, nsvd, 10, 0.001, 1) # https://ieeexplore.ieee.org/abstract/document/1467342 (also see https://arxiv.org/pdf/2210.12097)
                        qdata -= Xlow*svdMAD+svdmedian
                    end=time.time()
                    print("SVD execution (n={}) time was {:.3f} sec".format(nsvd, end-start))

                # If a fake signal was added, measure its post-SVD amplitude (for higher S/N ratios, this might be a useful measure of signal loss)
                if (fake == "HI") and (ibeam == fbeam):
                    msx = max(1, int(sx/4))
                    msy = max(1, int(sy/4))
                    mamp = np.average(qdata[x0-msx:x0+msx,y0-msy:y0+msy])*scale
                    mamp2 = mamp-np.median(qdata)*scale
                    print("Post-SVD fake source amplitude = {:.3e}".format(mamp))
                    print("Post-SVD fake source amplitude-median = {:.3e}".format(mamp2))
                    print("Fake output/input = {:.3}".format(mamp/fakeamp))
                    print("Fake output-median/input = {:.3}".format(mamp2/fakeamp))
                if (fake != "none") and (fakeamp == 0.0) and (ibeam == fbeam):
                    if isub == 0:
                        clo = y0-22
                        chi = y0+3
                    else:
                        clo = y0-3
                        chi = y0+22
                    rms = np.std(qdata[x0,clo:chi])*scale
                    print("rms for beam_{}, cycle_{}, frequency range {:.1f} to {:.1f} MHz = {:.3e} Jy".format(fbeam, x0, fval[clo], fval[chi], rms))
                    
                # Plot label
                if ndrive == 1:
                    derr = "; excl. {} drive errors".format(ndrive)
                elif ndrive > 1:
                    derr = "; excl. {} drive errors".format(ndrive)
                else:
                    derr = ""
                if qsvd:
                    svdmode = "; {}-{}".format(SVDmethod, nsvd)
                else:
                    svdmode = ""
                
                rmode = " ("+method+svdmode+derr+")"

                #stats function
                def stats(x):
                    xstat = np.zeros(6)
                    xstat[5] = np.argmax(x)
                    xstat[4] = np.amax(x)
                    xstat[3] = np.percentile(x,95)
                    xstat[2] = np.percentile(x,50)
                    xstat[1] = np.percentile(x,5)
                    xstat[0] = np.amin(x)
                    return xstat

                # Image and spectrum statistics      
                istat = stats(qdata)
                tstat = stats(tdata)

                        
                # Write data
                if output != "none":
                    # Prepare slice array
                    indo = [0]*ndims
                    indo[0] = slice(None)
                    indo[-3] = pol
                    indo[-2]=slice(None)
                    indo[-1] = 0
                    #hduo_0SBa['data'][] = qdata*scale
                    hduo_0SBa['data'][tuple(indo)] = qdata*scale
                # Temporary LMC match for spectrum 257 beam 6 IF 1 [-A 6 -a 1 -e 7 -h paf_241118_061543_ON_cal.json -o lmc_mbmatch_2pol.hdf5] in paf_241118_100007.hdf5 or [-i -n] in lmc_mbmatch_2pol.hdf5 
                #if (apol==1) and (ibeam==6) and (isub==1) and (file=='lmc_mbmatch_2pol.hdf5'):
                # Temporary LMC match for spectrum 441 beam 6 IF 1 [-A 19 -a 1 -e 7 -h paf_241118_061543_ON_cal.json -o lmc_mbmatch2_2pol.hdf5] in paf_241118_100007.hdf5 or [-i -n] in lmc_mbmatch2_2pol.hdf5 [shifts from #192 to #143 after excluding DRIVE errors]
                if (apol==1) and (ibeam==19) and (isub==1) and (file=='lmc_mbmatch2_2pol.hdf5'):
                    tmpsave = "lmc_mbmatch.npz"
                    print("Saving data in temporary file:", tmpsave)
                    np.savez(tmpsave, stokesI=scale*qdata[143,:], freq=fval)
                    exit()

                # Spectrum plot function
                
                def splot(spec):
                    plt.figure(file+':'+str(pol))
                    plt.rc('font',family='serif')
                    qqspec = spec*scale
                    cmult = 1.0
                    ylab = "Flux density (Jy)"
                    if np.amax(qqspec) < 1.0e-1:
                        cmult = 1.0e3
                        ylab = "Flux density (mJy)"
                    if np.amax(qqspec) < 1.0e-4:
                        cmult = 1.0e6
                        ylab = "Flux density (uJy)"
                    plt.plot(fval, qqspec*cmult, label=BEAM+" "+csubband)
                    plt.xlim([fval[0], fval[-1]])
                    if (uylim != 0.0) or (dylim !=0.0):
                        plt.ylim([cmult*dylim, cmult*uylim])
                    ymin,ymax = plt.ylim()
                    plt.xlabel('Frequency (MHz)')
                    plt.ylabel(ylab)
                    plt.title("Bandpass"+rmode)
                    plt.legend()
                    if zoom:
                        plt.tight_layout()
                    else:
                        ax = plt.twiny()
                        plt.xlim(plot_start_ch, plot_end_ch)
                    plt.show(block=block)
                    if not block:
                        plt.pause(0.5)
                    plt.close()

                # Plot input spectra
                if not nodisplay:
                    splot(vref)
        
                # Time series plot function        
                def tplot(ts):
                    vtspec = ts*scale
                    plt.figure(file+':'+str(pol))
                    plt.rc('font',family='serif')
                    plt.plot(cint_times[0]*np.arange(len(vtspec)), vtspec, label=BEAM+" "+csubband)
                    plt.xlim(0, cint_times[0]*len(vtspec))
                    if (uylim != 0.0) or (dylim !=0.0):
                        plt.ylim([dylim, uylim])
                    plt.xlabel('Time (sec)')
                    plt.ylabel('Flux density (Jy)')
                    plt.title("Time series"+rmode)
                    plt.legend()
                    if zoom:
                        plt.tight_layout()
                    plt.show(block=block)
                    if not block:
                        plt.pause(0.5)
                    plt.close()

                # Plot input time series       
                if (ypix > 1) and (not nodisplay):
                    tplot(tref+np.average(vref))
                            
                # Waterfall image (pre-flattening)        
                if (ypix > 1) and (not nodisplay): 
                    plt.figure(file+':'+str(pol))
                    plt.rc('font',family='serif')
                    if (uylim == 0.0):
                        uylimd = tstat[3]*scale
                    else:
                        uylimd = uylim
                    if (dylim == 0.0):
                        dylimd = tstat[1]*scale
                    else:
                        dylimd = dylim
                    plt.imshow(tdata*scale, cmap=plt.cm.afmhot, aspect='auto', origin='lower', extent=[fval[0],fval[-1],0,cint_times[0]*len(tref)], vmin=dylimd, vmax=uylimd, interpolation='gaussian')
                    plt.xlabel('Frequency (MHz)')
                    plt.ylabel('Time (s)')
                    if zoom:
                        plt.tight_layout()
                    else:
                        plt.title("Waterfall ("+BEAM+" "+csubband+" unflattened)")
                        ax = plt.twiny()
                        plt.xlim(plot_start_ch, plot_end_ch)
                        plt.colorbar(location="right")
                    plt.show(block=block)
                    if not block:
                        plt.pause(0.5)
                    plt.close()

                # Histogram (post-flattening)        
                if (ypix > 1) and (not nodisplay): 
                    # rms
                    print("rms = {:.3g}".format(np.std(scale*qdata.flatten())))
                    # Save the data for tests if size small enough
                    pixlimit = 100e6
                    tmpsave = "qdata_tmp.npy"
                    if np.size(qdata) < pixlimit:
                        print("NOT saving data in temporary file:", tmpsave)
                        #np.save(tmpsave, scale*qdata)
                    else:
                        print("Data size over pixel limit ({}) for temporary saves...".format(pixlimit))
                    # Histogram the flux density
                    plt.figure('Flux density histogram', figsize=[6.4, 4.8])
                    plt.rc('font',family='serif')
                    plt.hist(scale*qdata.flatten(), bins=100)
                    plt.yscale('log')
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    
                # Waterfall image (post-flattening)        
                if (ypix > 1) and (not nodisplay): 
                    plt.figure(file+':'+str(pol))
                    plt.rc('font',family='serif')
                    if (uylim == 0.0):
                        uylimd = istat[3]*scale 
                    else:
                        uylimd = uylim
                    if (dylim == 0.0):
                        dylimd = istat[1]*scale
                    else:
                        dylimd = dylim
                    plt.imshow(qdata*scale, cmap=plt.cm.afmhot, aspect='auto', origin='lower', extent=[fval[0],fval[-1],0,cint_times[0]*len(tref)], vmin=dylimd, vmax=uylimd, interpolation='gaussian')
                    plt.xlabel('Frequency (MHz)')
                    plt.ylabel('Time (s)')
                    if zoom:
                        plt.tight_layout()
                    else:
                        plt.title("Waterfall ("+BEAM+" "+csubband+" flattened)"+rmode)
                        ax = plt.twiny()
                        plt.xlim(plot_start_ch, plot_end_ch)
                        plt.colorbar(location="right")
                    plt.show(block=block)
                    if not block:
                        plt.pause(0.5)
                    plt.close()
        
        # End of suband loop
    # End of beam loop
    hdu.close()
    if ref != 'none':
        hdur.close()
    if output != "none":
        hduo.close()
# End of file loop
exit()

