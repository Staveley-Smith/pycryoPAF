#!/usr/bin/env python
#
# Purpose:
# Averages and calibrates SDHDF data from the Murriyang/Parkes cryoPAF receiver
#
# Example useage: 
# python paf-aver.py -s /Volumes/LaCie8T/LOCALDATA/P967/SHDFS/uwl_190319_165745.hdf -r /Volumes/LaCie8T/LOCALDATA/P967/SHDFS/uwl_190319_170309.hdf -i
# Tested on sdhdf version 1.9 and  python 3.11

# Lister Staveley-Smith
# 2025 April 21: renamed from paf.py to emphasise that it averages spectral data in time.

# python2 print compatibility
from __future__ import print_function

# Libraries (more are imported on demand once options are selected)

import sys
import re
import numpy as np
from scipy import signal
from scipy import fftpack
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import h5py
import getopt
import json
import os
import glob
import pkg_resources

# Python version
#if int(sys.version_info.major) > 2:
#    print("WARNING: python version > 2 detected")
#    exit("ERROR: uwl1.9 not yet working with python3 (h5py tweeks needed)")

# Read scipy version number
scivers = pkg_resources.get_distribution("scipy").version
sciflag = "new"
scint = list(map(int, scivers.split(".")))
if (scint[0] == 0) and (scint[1] < 18):
    sciflag = "old"
    print("WARNING: scipy version number ({}) too old for some options".format(scivers))
    
# Rydberg-related constants

Rc = 10973731.568508*299792458.0
Rhc = Rc*1836.15267343/(1.0+1836.15267343)
Rpc = Rc/2.0

# List of potential calibrators

callist = {
 '1934-638': {'f0': -30.7667, 'f1': 26.4908, 'f2': -7.0977, 'f3': 0.605334, 'fluxunit': 'Jy'}, 
 'Hydra A': {'f0': 4.497, 'f1': -0.91, 'f2': 0.0, 'f3': 0.0, 'fluxunit': 'Jy'},
 'Hydra_A': {'f0': 4.497, 'f1': -0.91, 'f2': 0.0, 'f3': 0.0, 'fluxunit': 'Jy'},
 '0407-658': {'f0': 5.179, 'f1': -1.280, 'f2': 0.0, 'f3': 0.0, 'fluxunit': 'Jy'},
 '0823-500': {'f0': -4.33508236, 'f1': 3.86117215, 'f2': -0.71842839, 'f3': 0.0, 'fluxunit': 'Jy'},
 'S8': {'f0': 1.881, 'f1': 0.0, 'f2': 0.0, 'f3': 0.0, 'fluxunit': 'K'}, # 76K at 14 arcmin resolution according to Kalberla+ (1982)
 'S9': {'f0': 1.919, 'f1': 0.0, 'f2': 0.0, 'f3': 0.0, 'fluxunit': 'K'} # 83K at 14 arcmin resolution according to Bruens+ (2005)
}

# Flux density / Temperature calculation
def flux(cal, farr):
    lfarr = np.log10(farr)
    return 10.0**(callist[cal]['f0']+callist[cal]['f1']*lfarr+callist[cal]['f2']*lfarr**2+callist[cal]['f3']*lfarr**3)

# Initialise calparams
calparams = {}

# Process command line parameters

ref = 'none'
sig = 'none'
cont = 'none'
stokes = False
vline = False
fit = False
qsub = False
asub = 0
bsub = 0
aBeam = 0
bBeam = 0
qBeam = False
#Beam = 0
start_ch = 0
end_ch = 0
uylim = 0.0
dylim = 0.0
n_notch = 0
poly = None
fit_range = "0,0"
qrange= False
calout = False
calin = "none"
qcalin = True
nodisplay = False
block = True
output = "none"
join = False
median = False
fake = "none"
fakestr = "none"
fakeamp = 1.0
smooth = 1
zoom = False
markfreq=0.0
text = "none"
textind = 0
history =""

# testing
noav = True

optionshort = '?a:b:A:B:c:d:e:f:F:gh:H:ijk:lmnNo:p:r:s:tu:v:w:x:X:z'
optionlist = ['?','a=', 'b=', 'A=', 'B=', 'cont=', 'd=', 'e=', 'fft=', 'F=', 'g', 'h=', 'H=', 'i', 'j', 'k=', 'line', 'median', 'n', 'N', 'o=', 'p=', 'ref=', 'sig=', 't', 'u=', 'v=', 'w=', 'x=', 'X=', 'zoom']
optionhelp = ['help', 'start sub-band', 'end sub-band', 'start beam', 'end beam', 'continuum data filename (for better bandpass calibration)', 'lower plot limit', 'number of smoothing channels',
              'number of fft notches (excluding total power)', 'channel range for polynomial fit (comma-separated)', 'write calibration (json) data', 'apply this calibration file', 'apply this calibration to other beams (first entry)', 'Stokes I',
              'concatenate (join) sub-bands', 'location of dashed vertical line', 'plot location of zero-redshift recombination lines', 'use medians for all time averages', 
              'no plot', 'unblocked plot', 'Output data file', 'polynomial baseline order (0 and -ve values applied pre-time-average)', 'reference spectrum', 'source (signal) spectrum', 
              'text spectrum', 'upper plot limit', 'start channel number (within each sub-band)', 'end channel number (within each sub-band)', 
              'extra fake lines (-x HII or -x PsII or -x HI)', 'extra fake line amplitude (pre-calibration)', 'large font (for publication)']

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
    elif opt in ('-x', '--x'):    # Add fake lines
        fake = arg
    elif opt in ('-X', '--X'):    # Add fake lines
        fakestr = arg
    elif opt in ('-g', '--g'):      # write calibration file?
        calout = True
    elif opt in ('-h', '--h'):      # input calibration filename
        calin = arg
    elif opt in ('-H', '--H'):      # input calibration filename
        calin = arg
        qcalin = False
    elif opt in ('-k', '--k'):      # dashed vertical line location
        markfreq = float(arg)
    elif opt in ('-f', '--fft'):      # number of notched FFT channels (exc total power)
        n_notch = int(arg)
    elif opt in ('-F', '--F'):      # channel ranges for polynomial fit - e.g. -F 1000,2000,4000,5000
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
    elif opt in ('-e', '--e'):      # number of smoothing channels
        smooth = int(arg)
    elif opt in ('-p', '--p'):      # polynomial order
        poly = int(arg)
    elif opt in ('-j', '--j'):   # concatenate?
        join = True    
    elif opt in ('-u', '--u'):      # upper plot limit
        uylim = float(arg)
    elif opt in ('-d', '--d'):      # lower plot limit
        dylim = float(arg)
    elif opt in ('-v', '--v'):      # start channel
        start_ch = int(arg)
        print("WARNING: scipy version number ({}) may cause crash with option -v".format(scivers))
    elif opt in ('-w', '--w'):      # end channel
        end_ch = int(arg)
        print("WARNING: scipy version number ({}) may cause crash with option -w".format(scivers))
    elif opt in ('-i', '--i'):      # Average polarisations to Stokes I
        stokes = True    
    elif opt in ('-l', '--line'):      # Plot location of zero-redshift recombination lines and legend
        vline = True    
    elif opt in ('-n', '--n'):      # Skip display
        nodisplay = True    
    elif opt in ('-N', '--N'):      # display briefly
        block = False    
    elif opt in ('-m', '--median'):      # Skip display
        median = True    
    elif opt in ('-o', '--out'):      # filename for output calibrated spectrum
        output = arg    
    elif opt in ('-c', '--cont'):   # filename for continuum spectrum
        cont = arg
    elif opt in ('-t', '--text'):   # text spectrum
        text = ""
    elif opt in ('-z', '--zoom'):      # Skip display
        zoom = True    

if len(remainder) != 0:
    print('Unrecognised arguments:', remainder)
    exit()

# Certain option combinations not allowed

if sig == 'none':
  exit('ERROR: need signal file (-s <hdf name>)')

if (ref == 'none') and (cont != 'none'):
    exit('ERROR: need reference file (-r <hdf name>)')

if calout:
    if calin != "none":
        exit("ERROR: can read cal file or write cal file but not both")

if bsub == 0:
    bsub=asub+1        
if join:
    srange = [asub, bsub]
    bsub = asub+1

if fake != "none":
    if (fake != "PsII") and (fake != "HII") and (fake != "HI"):
        exit("ERROR: only \"-x PsII\" or \"-x HII\" or \"-x HI\"supported")
    if fakestr != "none":
        try:
            fakeamp = float(fakestr)
        except:
            exit("ERROR: argument for -X need to be the amplitude of the fake line")
    print("NOTIFICATION: fake line amplitude = {}".format(fakeamp))
elif fakestr != "none":
    exit("ERROR: option \"-X\" is redundant without option \"-x HI\" etc (extra fake line)")
    
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

if nodisplay == False:
    from pylab import rcParams
    import matplotlib.pyplot as plt
    
if cont != 'none':
    fit = True
    
if stokes:
    npol=1
else:
    npol=2

poldict = 'npol_'+str(npol)

# Fitting function - initial parameters set to zero
   
maxpar = 20
coeffs = np.array([0.0]*maxpar)

# Residual function for least squares routine
def residual(coeffs, data, ref1, ref2, ref3):
    ref1av=np.average(ref1)
    ref2av=np.average(ref2)
    ref3av=np.average(ref3)
    norm1=ref1/ref1av
    norm2=ref2/ref2av
    if ref3av != 0.0:
        norm3=ref3/ref3av
    else:
        norm3=ref3
    model = coeffs[0]*ref1+coeffs[1]*ref2+coeffs[2]*ref3
    # Extra quadratic terms not useful?
#+coeffs[3]*norm1**2+coeffs[4]*norm2**2+coeffs[5]*norm3**2+coeffs[6]*norm1*norm2+coeffs[7]*norm1*norm3+coeffs[8]*norm2*norm3
    return data-model

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
    
# Pre-load Halpha recombination frequencies from n=115 to 211 and Ps alpha frequencies from n=93 to 167
Hstart = 115
Hend = 211
Pstart = 93
Pend = 167
def rydberg(R, level):
    freq = R*(1.0/level**2 - 1.0/(level+1)**2)/1.0e6
    return freq
Hfreq = [rydberg(Rhc, Hstart)]    
Pfreq = [rydberg(Rpc, Pstart)]    
for i in range(Hstart, Hend):
    Hfreq.append(rydberg(Rhc, i+1))
for i in range(Pstart, Pend):
    Pfreq.append(rydberg(Rpc, i+1))

# Figure size in x and y (supposedly in inches); white background

if not nodisplay:
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['figure.facecolor'] = 'white'
    if zoom:
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.labelsize'] = 'x-large'
        plt.rcParams['axes.titlesize'] = 'x-large'
        plt.rcParams['legend.fontsize'] = 'x-large'
        plt.rcParams['lines.linewidth'] = 2

# Output resolution (determines sampling interval)

#xres = 4096
yres = 2048

# Subband range (python range; [5,6] for 1420 MHz)
#srange = [asub, bsub]


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

# Cont filename handling        
if cont != "none": 
    contfiles = filenameextractor(cont)
    lcontfiles = len(contfiles)
    if (lcontfiles == 1) and (lsigfiles != 1):
        for i in range(1,lsigfiles):
            contfiles.append(contfiles[0])
        lcontfiles = len(contfiles)
    if (lcontfiles != lsigfiles):
        exit("ERROR: number of cont files must be one or be equal to the number of sig files")

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

    # See if first requested BEAM exists
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
        bBeam = aBeam + 1
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
    if cont != 'none':
        hduc=h5py.File(contfiles[ifile], 'r')
        csource = hduc['metadata']['beam_parameters'][0]['SOURCE']
        if isinstance(csource,bytes):
            csource = csource.decode()
        print("Continuum:   {} ({})".format(contfiles[ifile], csource))
        DRIVEcont = hduc[BEAM][asubband]['metadata']['observation_parameters']['DRIVE_STATUS']

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

        #calparams = {CALBEAM:{poldict:{}}}
        # Loop over subbands
        for isub in range(asub, bsub):
            csubband = "band_SB{:d}".format(isub)
            print(BEAM, csubband)
            if join:
                print("{} band_SB{}-SB{}".format(str(srange[0]),str(srange[1])))
            else:
                srange = [isub, isub+1]
                print("{} {}".format(BEAM, csubband))
    
            # Check if this is a cal source

            calwrite = "none"
            if calout:
                for cal in callist:
                    if re.search(cal, source):
                        calwrite = filecore.split(".hdf")[0] + "_cal" + ".json"
                        break
                if calwrite != "none":
                    print("Calibrator", cal, "recognised")
                    print("Cal filename:", calwrite)
                    if os.path.exists(calwrite):
                        print('NOTIFICATION: appending/updating existing cal file')
                        with open(calwrite, 'r+') as json_file:
        	                calparams=json.load(json_file)
                        json_file.close()

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
                    
            # Reference metadata
            if ref != "none":
                try:
                    DRIVEref = hdur[BEAM][asubband]['metadata']['observation_parameters']['DRIVE_STATUS']
                except:
                    DRIVEref = "N/A"
                    
            # Continuum metadata
            if cont != "none":
                try:
                    DRIVEcont = hduc[BEAM][asubband]['metadata']['observation_parameters']['DRIVE_STATUS']
                except:
                    DRIVEcont = "N/A"
            
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
        #    
            calavail = False
            calravail = False
            calcavail = False
            if 'calibrator_data' in hdu[BEAM]['band_SB'+str(srange[0])]:
                if ('cal_data_on' in hdu[BEAM]['band_SB'+str(srange[0])]['calibrator_data']) and ('cal_data_off' in hdu[BEAM]['band_SB0']['calibrator_data']):
                    cal_dims = hdu[BEAM]['band_SB'+str(srange[0])]['calibrator_data']['cal_data_on'].shape
                    ncdims = len(cal_dims)
                    calavail = True
            if not calavail:
                print("WARNING: no cal data available for source")
            if ref != 'none':
                if 'calibrator_data' in hdur[BEAM]['band_SB'+str(srange[0])]:
                    if ('cal_data_on' in hdur[BEAM]['band_SB'+str(srange[0])]['calibrator_data']) and ('cal_data_off' in hdur[BEAM]['band_SB0']['calibrator_data']):
                        cal_dims = hdur[BEAM]['band_SB'+str(srange[0])]['calibrator_data']['cal_data_on'].shape
                        calravail = True
                if not calravail:
                    print("WARNING: no cal data available for reference")
            if cont != 'none':
                if 'calibrator_data' in hduc[BEAM]['band_SB'+str(srange[0])]:
                    if ('cal_data_on' in hduc[BEAM]['band_SB'+str(srange[0])]['calibrator_data']) and ('cal_data_off' in hduc[BEAM]['band_SB0']['calibrator_data']):
                        cal_dims = hduc[BEAM]['band_SB'+str(srange[0])]['calibrator_data']['cal_data_on'].shape
                        calcavail = True
                if not calcavail:
                    print("WARNING: no cal data available for continuum")
        
            print('Dimensions of sub-band data:', data_dims)
            if calavail:
                print('Dimensions of sub-band cal_data:', cal_dims)
            #print 'Dimensions of first subband frequencies:', freq_dims

            # Estimated data size (NB. second dimension was a dummy axis for SDHDF version numbers <~ 2.4)

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
            if calavail or calravail or calcavail:
                squish = data_dims[-2]/cal_dims[-2]
                start_ch_cal = int(start_ch/squish)
                end_ch_cal = int(end_ch/squish)
                # Interpolation of cal spectrum requires monotonically increasing frequencies, so remove last cal frequency channel
                if join and (end_ch==data_dims[-2]):
                    end_ch_cal = int(end_ch/squish)-1
            # Plot channel range (IFs can be joined)
            plot_start_ch = start_ch
            plot_end_ch = end_ch+xpix-xpix1
       
            # Skip pixels for display?
    #        xskip = max(1,int((end_ch-start_ch)/xres))
            xskip = smooth
            yskip = 1
            #yskip = max(1,int(ypix/yres))
            print('Smoothing and resampling by {:d} spectral channels'.format(xskip))
            carray = np.ones((yskip, xskip))/float(yskip*xskip)
    
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
                if calavail:
                    cind1 = [0]*ncdims
                    cind1[0] = slice(None)
                    cind1[-3] = pol
                    cind1[-2] = slice(start_ch_cal,end_ch_cal)
                    cind2 = list(cind1)
                    cind2[-3] = pol+1
                # Data / cal data
                tdata =  np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['astronomy_data']['data'][tuple(ind1)] for subband in range(srange[0],srange[1])]), axis=1)
                #print(ind1)
                #print(tdata.shape)
                if calavail:
                    calondata = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_data_on'][tuple(cind1)] for subband in range(srange[0],srange[1])]), axis=1)
                    caloffdata = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_data_off'][tuple(cind1)] for subband in range(srange[0],srange[1])]), axis=1)
                if (npol == 1) and (ppix > 1):
                    t2data = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['astronomy_data']['data'][tuple(ind2)] for subband in range(srange[0],srange[1])]), axis=1)
                    tdata = (tdata + t2data)/2.0
                    if median:
                        rat = abs(np.median(tdata)/np.median(t2data))
                    else:
                        rat = abs(np.average(tdata)/np.average(t2data))
                    if (rat < 0.666) or (rat > 1.5):
                        print("WARNING: significant difference between levels of two pols (factor {:.2f})".format(rat))
                        print("WARNING: Stokes I processing may not be appropriate")
                    if calavail:
                        cal2ondata = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_data_on'][tuple(cind2)] for subband in range(srange[0],srange[1])]), axis=1)
                        cal2offdata = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_data_off'][tuple(cind2)] for subband in range(srange[0],srange[1])]), axis=1)
                        calondata = (calondata + cal2ondata)/2.0
                        caloffdata = (caloffdata + cal2offdata)/2.0
                if ref != 'none':
                    tref = np.concatenate(([hdur[BEAM]['band_SB'+str(subband)]['astronomy_data']['data'][tuple(ind1)] for subband in range(srange[0],srange[1])]), axis=1)
                    if calravail:
                        calonref = np.concatenate(([hdur[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_data_on'][tuple(cind1)] for subband in range(srange[0],srange[1])]), axis=1)
                        caloffref = np.concatenate(([hdur[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_data_off'][tuple(cind1)] for subband in range(srange[0],srange[1])]), axis=1)
                    if (npol == 1) and (ppix > 1):
                        t2ref = np.concatenate(([hdur[BEAM]['band_SB'+str(subband)]['astronomy_data']['data'][tuple(ind2)] for subband in range(srange[0],srange[1])]), axis=1)
                        tref = (tref + t2ref)/2.0
                        if calravail:
                            cal2onref = np.concatenate(([hdur[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_data_on'][tuple(cind2)] for subband in range(srange[0],srange[1])]), axis=1)
                            cal2offref = np.concatenate(([hdur[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_data_off'][tuple(cind2)] for subband in range(srange[0],srange[1])]), axis=1)
                            calonref = (calonref + cal2onref)/2.0
                            caloffref = (caloffref + cal2offref)/2.0
                        
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
                
                if cont != 'none':
                    cref = np.concatenate(([hduc[BEAM]['band_SB'+str(subband)]['astronomy_data']['data'][tuple(ind1)] for subband in range(srange[0],srange[1])]), axis=1)
                    if (npol == 1) and (ppix > 1):
                        c2ref = np.concatenate(([hduc[BEAM]['band_SB'+str(subband)]['astronomy_data']['data'][tuple(ind2)] for subband in range(srange[0],srange[1])]), axis=1)
                        cref = (cref + c2ref)/2.0
                    
                    # Figure out integration time parameters (raw data v processed data)

                    try:
                        int_times_cont = hduc[BEAM][csubband]['metadata']['integ_times'][:]
                    except:
                        DUMP_TIME_cont = hduc[BEAM]['metadata']['band_parameters'][0]['REQUESTED_INTEGRATION_TIME']
                        N_DUMPS_cont = hduc[BEAM]['metadata']['band_parameters'][0]['NUMBER_OF_INTEGRATIONS']
                        int_times_cont = np.array([DUMP_TIME_cont]*N_DUMPS_cont)
                    int_time_cont = np.sum(int_times_cont)
                    if (len(int_times_cont) > 1):
                        tsint = np.sort(int_times_cont)
                        if (tsint[0]+tsint[-1]) != 0.0:
                            if abs(2.0*(tsint[0]-tsint[-1])/(tsint[0]+tsint[-1])) > 0.1:
                                print("WARNING: spectra (cont) found with integration times differing by > 10%")
                

                # Average/convolve (boundary="sim" doesn't seem to work) and downsample
        
                tvdata = signal.convolve2d(tdata, carray, mode="same", boundary="wrap")
                tdata = tvdata[::yskip,::xskip]
                if ref != 'none':
                    tvref = signal.convolve2d(tref, carray, mode="same", boundary="wrap")
                    tref = tvref[::yskip,::xskip]
                if cont != 'none':
                    tvcont = signal.convolve2d(cref, carray, mode="same", boundary="wrap")
                    tcont = tvcont[::yskip,::xskip]
                
                # Frequencies (concatenate, then downsample)

                fval = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['astronomy_data']['frequency'][0][start_ch:end_ch] for subband in range(srange[0],srange[1])]))
                #print(fval.shape)
                fval = fval[::xskip]
                lfval = len(fval)
        
                if calavail:
                    fcval = np.concatenate(([hdu[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_frequency'][start_ch_cal:end_ch_cal] for subband in range(srange[0],srange[1])]))
                elif calravail:
                    fcval = np.concatenate(([hdur[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_frequency'][start_ch_cal:end_ch_cal] for subband in range(srange[0],srange[1])]))
                elif calcavail:
                    fcval = np.concatenate(([hduc[BEAM]['band_SB'+str(subband)]['calibrator_data']['cal_frequency'][start_ch_cal:end_ch_cal] for subband in range(srange[0],srange[1])]))
        
                # Axis coordinate limits 

                extpix = [0, tdata.shape[0], 0, tdata.shape[1]]
                extcor = [fval[0], fval[-1], 0, ypix*int_times[0]]
                #print 'Data array (pixels):', extpix
                #print '....in image units:', extcor
                
            # Output text file - prepare
                if text != "none":
                    textind += 1
                    textfile = filecore.split(".hdf")[0] + "_" + str(textind).zfill(3) + ".txt"
                    print("Output text file: {}".format(textfile))
                    if os.path.exists(textfile):
                        exit('ERROR: text file already exists')
                    tf = open(textfile, 'w')
                    if 'proc_history' in hdu['metadata']:
                        lh = hdu['metadata']['proc_history'].shape[0]
                        for ilh in range(lh):
                            print("#", hdu['metadata']['proc_history'][ilh].decode(), file=tf)
                    print("#", history, file=tf)
                    print("# Frequency (MHz)      Flux Density (Jy)", file=tf)
                        
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
                    if (csubband in hduo_0):
                        hduo_0SB = hduo_0[csubband]
                        hduo_0SBa = hduo_0SB['astronomy_data']
                        hduo_0SBm = hduo_0SB['metadata']
                        if ('averaged_filenames' not in hduo_0SBm):
                            print("WARNING: metadata in output file does not appear to be compatible")
                            lt = hduo_0SBa['data'].shape[0]
                            tr = np.arange(lt)
                            tstr = tr.astype(str)
                            print("NOTIFICATION: creating averaged_filenames with length", lt)
                            # Unlikely to reach here as arrays need to be re-sizeable
                            hduo_0SBm.create_dataset('averaged_filenames', data=np.array(tstr,dtype='|S256'), maxshape=(None,), chunks=True)
                        if ('integ_times' not in hduo_0SBm):
                            print("WARNING: metadata in output file does not appear to be compatible")
                            lt = hduo_0SBa['data'].shape[0]
                            print("NOTIFICATION: creating integ_times with length", lt)
                            # Unlikely to reach here as arrays need to be re-sizeable
                            hduo_0SBm.create_dataset('integ_times', data=np.zeros(lt), maxshape=(None,), chunks=True)
                        if ('source_params' not in hduo_0SBm):
                            print("WARNING: metadata in output file does not appear to be compatible")
                            lt = hduo_0SBa['data'].shape[0]
                            print("NOTIFICATION: creating source_params with length", lt)
                            # Unlikely to reach here as arrays need to be re-sizeable
                            hduo_0SBm.create_dataset('source_params', data=np.full((lt,3),fill_value="",dtype='|S256'), maxshape=(None,3), chunks=True)
                        if pol == 0:
                            names = hduo_0SBm['averaged_filenames'][:]
                            times = hduo_0SBm['integ_times'][:]
                            lnames = names.shape[0]
                            ltimes = times.shape[0]
                            if lnames != ltimes:
                                exit("ERROR: mismatch between metadata array lengths averaged_filenames and integ_times")
                            if filecore in names:
                                ind = np.where(names==filecore)[0][0]
                                rnames = lnames
                                print('NOTIFICATION: updating output data for', names[ind])
                                hduo_0SBm['integ_times'][ind] = int_time
                            else:
                                ind = lnames
                                rnames = lnames+1
                                hduo_0SBm['averaged_filenames'].resize((rnames,))
                                hduo_0SBm['integ_times'].resize((rnames,))
                                hduo_0SBm['source_params'].resize((rnames,3))
                                hduo_0SBm['averaged_filenames'][ind] = filecore
                                hduo_0SBm['integ_times'][ind] = int_time
                                hduo_0SBm['source_params'][ind,0] = source
                                hduo_0SBm['source_params'][ind,1] = RA_STR
                                hduo_0SBm['source_params'][ind,2] = DEC_STR
                            nrfval = len(hduo_0SBa['frequency'])
                            nrnpol = hduo_0SBa['data'].shape[2]
                            if nrfval != lfval:
                                try:
                                  # Use rows for frequency, rather than columns, for PAF data
                                  hduo_0SBa['frequency'].resize((1,lfval,))
                                except: 
                                  exit("ERROR: unable to resize frequency array")  
                            if (nrfval != lfval) or (nrnpol != npol) or (lnames != rnames):    
                                print("NOTIFICATION: re-sizing frequency and/or polarisation and/or time (file) axis")
                                try:
                                    if len(out_shape) == 5:
                                      hduo_0SBa['data'].resize((rnames,1,npol,lfval,1))
                                    elif len(out_shape) == 4:
                                      hduo_0SBa['data'].resize((rnames,npol,lfval,1))
                                    else:
                                      exit("ERROR: unable to resize data array")
                                except:
                                    exit("ERROR: unable to resize data array")
                            # Use rows for frequency, rather than columns, for PAF data
                            hduo_0SBa['frequency'][0,:] = fval
                    else:
                        hduo_0SB = hduo_0.create_group(csubband)
                        hdu.copy(BEAM+'/'+csubband+'/metadata', hduo_0SB)
                        hduo_0SBm = hduo_0SB['metadata']
                        # if reading from a file that this script has written, 'averaged_filenames' should exist
                        if ('averaged_filenames' in hduo_0SBm):
                            hduo_0SBm['averaged_filenames'].resize((1,))
                            hduo_0SBm['averaged_filenames'][0] = filecore
                        else:
                            hduo_0SB['metadata'].create_dataset('averaged_filenames', data=np.array([filecore],dtype='|S256'), maxshape=(None,), chunks=True)
                        # if reading from a file that this script has written, 'integ_times' should exist
                        if ('integ_times' in hduo_0SBm):
                            hduo_0SBm['integ_times'].resize((1,))
                            hduo_0SBm['integ_times'][0] = int_time
                        else:
                            hduo_0SB['metadata'].create_dataset('integ_times', data=np.array([int_time]), maxshape=(None,), chunks=True)
                        # if reading from a file that this script has written, 'source_params' should exist
                        if ('source_params' in hduo_0SBm):
                            hduo_0SBm['source_params'].resize((1,3))
                            hduo_0SBm['source_params'][0,0] = source
                            hduo_0SBm['source_params'][0,1] = RA_STR
                            hduo_0SBm['source_params'][0,2] = DEC_STR
                        else:
                            hduo_0SB['metadata'].create_dataset('source_params', data=np.array([[source, RA_STR, DEC_STR]],dtype='|S256'), maxshape=(None,3), chunks=True)
                        hduo_0SBa = hduo_0SB.create_group('astronomy_data')
                        # Use rows for frequency, rather than columns, for PAF data
                        hduo_0SBa.create_dataset('frequency', data=fval.reshape(1,lfval), maxshape=(1,None,), chunks=True)
                        # Prepare correct array dimensions (similar input and output dimensionality)
                        out_shape = [1]*ndims
                        out_shape[-3] = apol
                        out_shape[-2] = tdata.shape[1]
                        max_shape = [1]*ndims
                        max_shape[-2] = None
                        max_shape[-3] = 2
                        max_shape[0] = None 
                        # NB won't be able to mix pre and post-2.4 SDHDF data in the same output file due to different dimensionality
                        hduo_0SBa.create_dataset('data', data=np.zeros(out_shape), maxshape=tuple(max_shape), chunks=True)

                # Centre frequency in MHz (can be changed to peak freq if using an HI Galactic calibrator)
                midfreq = (fval[0] + fval[-1]) / 2.0
                
                # Keep a baseline value if a fit is done prior to calibration
                baseT = 0.0

                # Fake spectral data 
            
                def gaussian(xloc, mu, sigma):
                    return np.exp(-np.power(xloc - mu, 2.) / (2 * np.power(sigma, 2.)))
            
                if fake == "HII":
                     for vf in Hfreq:
                        if (vf > fval[0]) and (vf < fval[-1]):
                            tdata += fakeamp*gaussian(fval, vf, 3.e-5*vf)
            
                if fake == "PsII":
                    for vf in Pfreq:
                        if (vf > fval[0]) and (vf < fval[-1]):
                            tdata += fakeamp*gaussian(fval, vf, 1.e-3*vf)
                        
                if fake == "HI":
                    # Place blue/red-shifted HI line in centre of band
                    vf = 0.5*(fval[0] + fval[-1])
                    wf = 0.03*abs(fval[0] - fval[-1])
                    tdata += fakeamp*gaussian(fval, vf, wf)
                    
                # Apply pre-average polynomial
                
                if poly != None:
                    if poly < 1:
                        if qrange:
                            for ir in range(int(pdim/2)):
                                i1 = int(frange[0+2*ir]/smooth)
                                i2 = int(frange[1+2*ir]/smooth)
                                if i1 != 0:
                                    i1 -= int(start_ch/smooth)
                                if i2 != 0:
                                    i2 -= int(start_ch/smooth)
                                if ir == 0:
                                    chans = np.arange(i1,i2)
                                else:
                                    chans = np.append(chans,np.arange(i1,i2))
                        else:
                            chans = np.arange(tdata.shape[1])
                        # polyfit allows 2d (transposed) arrays, but polyval doesn't, so iterate
                        for ispec in range(tdata.shape[0]):
                            qsol = np.polyfit(chans, tdata[ispec,chans], deg=abs(poly))
                            polybase = np.polyval(qsol,np.arange(tdata.shape[1]))
                            baseT += polybase[np.argmax(tdata[ispec,:])]/tdata.shape[0]
                            tdata[ispec,:] = tdata[ispec,:] - polybase
                            # Sneaky median (need to have robust fit option)
                            if (poly == 0) and median:
                                tdata[ispec,:] = tdata[ispec,:] - np.median(tdata[ispec,chans])
                # Determine if there are drive errors
                if (len(DRIVE) > 0):
                    ndrive = np.sum(DRIVE==0)
                    print("{} valid integrations found".format(tdata.shape[0]))
                    print("{} integrations with drive errors (not averaged)".format(ndrive))
                    if ndrive == tdata.shape[0]:
                        exit("ERROR: no valid data to average")
                else:
                    ndrive = 0
                    DRIVE = 1
                # Average data along time axis
                if median:
                    vspec = np.median(tdata[DRIVE!=0,:], axis=0)
                else:
                    if ypix > 1:
                        print("NOTIFICATION: average spectrum is being weighted by integration time")
                        vspec = np.average(tdata[DRIVE!=0,:], axis=0, weights=int_times[DRIVE!=0])
                    else:
                        vspec = np.average(tdata[DRIVE!=0,:], axis=0)
                vspec_av = np.average(vspec)

                # Average cal data along time axis

                if calavail:
                    if median:
                        calonspec = np.median(calondata, axis=0)
                        caloffspec = np.median(caloffdata, axis=0)
                    else:
                        calonspec = np.average(calondata, axis=0, weights=None)
                        caloffspec = np.average(caloffdata, axis=0, weights=None)
                    dcalspec = calonspec - caloffspec
                    if sciflag == "new":
                        fsint = interp1d(fcval, dcalspec, kind='quadratic',fill_value='extrapolate')
                    else:
                        fsint = interp1d(fcval, dcalspec, kind='quadratic')
                    dcalspeci = fsint(fval)
                if calravail:
                    if ref != 'none':
                        if median:
                            calonvref = np.median(calonref, axis=0)
                            caloffvref = np.median(caloffref, axis=0)
                        else:
                            calonvref = np.average(calonref, axis=0, weights=None)
                            caloffvref = np.average(caloffref, axis=0, weights=None)
                        dcalref = calonvref - caloffvref
                        if sciflag == "new":
                            frint = interp1d(fcval, dcalref, kind='quadratic',fill_value='extrapolate')
                        else:
                            frint = interp1d(fcval, dcalref, kind='quadratic')
                        dcalrefi = frint(fval)
                else:
                    # Dummy spectrum for fit function
                    dcalspeci = np.zeros(lfval)
                
                # Average data along frequency axis
                vtspec = np.average(tdata, axis=1)
        
                # Divide spectrum and image by reference and normalise
        
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
                    if median:
                        vref = np.median(tref[DRIVEref!=0,:], axis=0)
                    else:
                        vref = np.average(tref[DRIVEref!=0,:], axis=0, weights=int_times_ref[DRIVEref!=0])
                    vref_av = np.average(vref)
                    vtref = np.average(tref[DRIVEref!=0,:], axis=1)
                    if vspec.shape[0] != vref.shape[0]:
                        exit("ERROR: sig and ref spectra have different numbers of channels")
                    if fit:
                        if median:
                            wref = np.median(tcont, axis=0)
                        else:
                            wref = np.average(tcont, axis=0, weights=int_times_cont)
                        wrtef = np.average(tcont, axis=1)
                        if vspec.shape[0] != wref.shape[0]:
                            exit("ERROR: sig and cont spectra have different numbers of channels")
                    
                        # Perform fit
                    
                        print("Fitting spectrum...")
                        # loss='soft_l1' is robust; 'linear' is least squares
                        out = least_squares(residual, coeffs, loss='soft_l1', args=(vspec, vref, wref, dcalspeci))
                        print("Optimized model parameters:")
                        print(out.x)
                        final_res = residual(out.x, vspec, vref, wref, dcalspeci)                
                        nref = vspec-final_res
                        nref_av = np.average(nref)
                        nref = nref*vref_av/nref_av
                        nref_av = np.average(nref)
                        qspec = (np.divide(vspec, nref)-1.0)*nref_av   
                        #print nref_av           
                    else:
                        qspec = (np.divide(vspec, vref)-1.0)*vref_av
                        if calavail:
                            tmp1 = np.divide(vspec,dcalspeci)
                        if calravail:
                            tmp2 = np.divide(vref,dcalrefi)
        #                    qspec2 = (np.divide(tmp1, tmp2)-1.0)*vref_av
        #                    qspec3 = vspec-dcalspeci*np.average(vspec)/np.average(dcalspeci)
        #                    qspec3 = tmp1*vref_av/np.average(tmp1)
                else:
                    qspec = vspec
                    
                # If we have only read in a single spectrum, adjust the array back to 1-D
                if len(qspec.shape) == 2:
                    qspec = qspec.flatten()
                
                # Fourier filter
        
                if n_notch > 0:
                    sig_fft = fftpack.fft(qspec)
                    power = np.abs(sig_fft)**2
                    notch_freq_fft = sig_fft.copy()
                    max_power = np.sort(power)[-n_notch]
                    notch_freq_fft[power >= max_power] = 0.0
                    # Put the DC term back!
                    if notch_freq_fft[0] == 0.0:
                        notch_freq_fft[0] = sig_fft[0]
                        print("Zero-frequency FFT notch override...")
                    qspec = np.real(fftpack.ifft(notch_freq_fft))
                
                # Polynomial fit
            
                if poly != None:
                    if qrange:
                        for ir in range(int(pdim/2)):
                            i1 = int(frange[0+2*ir]/smooth)
                            i2 = int(frange[1+2*ir]/smooth)
                            if i1 != 0:
                                i1 -= int(start_ch/smooth)
                            if i2 != 0:
                                i2 -= int(start_ch/smooth)
                            if ir == 0:
                                chans = np.arange(i1,i2)
                            else:
                                chans = np.append(chans,np.arange(i1,i2))
                    else:
                        chans = np.arange(len(qspec))
                    qsol = np.polyfit(chans, qspec[chans], abs(poly))
                    polybase = np.polyval(qsol,np.arange(len(qspec)))
                    if poly > 0:
                        baseT = polybase[np.argmax(qspec)]
                    qspec = qspec - polybase
                    # Sneaky median (need to have robust fit option)
                    if (poly == 0) and median:
                        qspec = qspec - np.median(qspec)

                # Fake spectral data - code moved earlier
            
                #def gaussian(xloc, mu, sigma):
                #    return np.exp(-np.power(xloc - mu, 2.) / (2 * np.power(sigma, 2.)))
            
                #if fake == "HII":
                #     for vf in Hfreq:
                #        if (vf > fval[0]) and (vf < fval[-1]):
                #            qspec = qspec + fakeamp*gaussian(fval, vf, 3.e-5*vf)
            
                #if fake == "PsII":
                #    for vf in Pfreq:
                #        if (vf > fval[0]) and (vf < fval[-1]):
                #            qspec = qspec + fakeamp*gaussian(fval, vf, 1.e-3*vf)
                        
                #if fake == "HI":
                    # Place blue/red-shifted HI line in centre of band
                #    vf = 0.5*(fval[0] + fval[-1])
                #    wf = 0.03*abs(fval[0] - fval[-1])
                #    qspec = qspec + fakeamp*gaussian(fval, vf, wf)

                # Prep waterfall data
        
                if ref != 'none':
                    if fit:
                        # Grow ref spectrum back to 2D array
                        denom = np.divide(qspec,nref_av)+1.0
                        nref = np.divide(vspec, denom) 
                        vrefim = np.repeat(nref, repeats=extpix[1])
                    else:
                        # Grow ref spectrum back to 2D array
                        denom = np.divide(qspec,vref_av)+1.0
                        vref = np.divide(vspec, denom) 
                        vrefim = np.repeat(vref, repeats=extpix[1])
                    vrefim = np.reshape(vrefim, [extpix[3],-1])
                    vrefim = np.transpose(vrefim)
                    qdata = (np.divide(tdata, vrefim)-1.0)*np.average(vref)
                else:
                    qdata = tdata
            

                def stats(x):
                    xstat = np.zeros(6)
                    xstat[5] = np.argmax(x)
                    xstat[4] = np.amax(x)
                    xstat[3] = np.percentile(x,95)
                    xstat[2] = np.percentile(x,50)
                    xstat[1] = np.percentile(x,5)
                    xstat[0] = np.amin(x)
        #            print 'Data (before calibration): min, h1, median, h3, max:\n', xstat
                    return xstat

                # Image and spectrum statistics
        
                istat = stats(qdata)
                sstat = stats(qspec)

                # Cal diode stats
        
                if calavail:
                    calonstat = stats(calonspec)
                    caloffstat = stats(caloffspec)
                if ref != "none":
                    vstat = stats(vref)
                    if calravail:
                        calonvrstat = stats(calonvref)
                        caloffvrstat = stats(caloffvref)
                        tcalrat = (calonvrstat[2] - caloffvrstat[2])/caloffvrstat[2]
        
                # Calibration
        
                if calwrite != "none" and ((ref != "none") or (callist[cal]['fluxunit'] == "K")):
                    calflux = flux(cal, midfreq)
                    if callist[cal]['fluxunit'] == "Jy":
                        snorm = sstat[2]
                        vnorm = vstat[2]
                    else:
                        snorm = sstat[4] # Peak temperature
                        midfreq = fval[int(sstat[5])]
                        if ref == "none":
                            if poly != None:
                                vnorm = baseT
                            else:
                                vnorm = sstat[2]
                        else:
                            vnorm = vstat[2]
                    scale = calflux/snorm
                    tsys = vnorm*scale
                    tsrc = snorm*scale
                    if calravail:
                        tcal = tcalrat*tsys
                    else:
                        tcal = 0.0
                    if pol == 0:
                        scalex = scale
                        scaley = 0.0
                        tsysx = tsys 
                        tsysy = 0.0
                        tcalx =tcal
                        tcaly = 0.0
                    else:
                        scaley = scale
                        tsysy = tsys
                        tcaly = tcal
              
                    newparams = {
                        'band_SB'+str(srange[0]): {
                            'scale': (scalex, scaley), 'tsys': (tsysx, tsysy), 'tcal': (tcalx, tcaly), 'freq': midfreq, 'frequnit': 'MHz', 'source': cal, 'flux': calflux, 'fluxunit': callist[cal]['fluxunit'], 'sig': file, 'ref': reffiles[ifile]
                        }
                    }
                    with open(calwrite, 'w') as json_file:
                        if BEAM in calparams:
                            print(poldict)
                            if poldict in calparams[BEAM]:
                                calparams[BEAM][poldict].update(newparams)
                            else:
                                calparams[BEAM].update({poldict: newparams})
                        else:
                            calparams.update({BEAM: {poldict: newparams}})
                        json.dump(calparams, json_file, indent=1)
                    json_file.close()
                    print("System temperature: {:.2f} {} ({:.3f}MHz)".format(tsys, callist[cal]['fluxunit'], midfreq))
                    print("Source temperature: {:.2f} {} ({:.3f}MHz)".format(tsrc, callist[cal]['fluxunit'], midfreq))             
                    
                elif calin != "none":
                    # Check if BEAM exists
                    if not (CALBEAM in calparams):
                        if qcalin:
                            exit("ERROR: missing calibration parameters for "+CALBEAM)
                        else:
                            CALBEAM = list(calparams)[0]
                            print("NOTIFICATION: forcing",CALBEAM,"calibration solution for",BEAM)
                    try:
                        #print(BEAM,CALBEAM,poldict,srange[0],pol)
                        #print(calparams)
                        scale = calparams[CALBEAM][poldict]['band_SB'+str(srange[0])]['scale'][pol]
                        #print("scale ok")
                        tcal = calparams[CALBEAM][poldict]['band_SB'+str(srange[0])]['tcal'][pol]
                        if ref != "none":
                            if tcal == 0.0:
                                print("WARNING: no value for tcal present in cal file")
                                tsys = vstat[2]*scale
                                tsrc = sstat[2]*scale
                            else:
                                tsys = tcal*caloffvrstat[2]/(calonvrstat[2]-caloffvrstat[2])
                                tsrc = tsys*(caloffstat[2]/caloffvrstat[2]-1.0)
                            scale = tsys/vstat[2]  # information only
                    except:
                        exit("ERROR: missing calibration parameters for this subband or polarisation")
                else:
                    scale = 1.0
                if scale != 1.0:
                    if 'fluxunit' in calparams[CALBEAM][poldict]['band_SB'+str(srange[0])]:
                        dunit = calparams[CALBEAM][poldict]['band_SB'+str(srange[0])]['fluxunit']
                    else:
                        dunit = "Jy"
                    print("Scaling factor to {}: {:.4e}".format(dunit, scale))
                    if ref != "none":
                        if calin != "none":
                            print("APPLYING EXISTING CALIBRATION PARAMETERS:")
                        else:
                            print("WRITING NEW CALIBRATION PARAMETERS:")
                            print("Cal temperature: {:.2f} {}".format(tcal, callist[cal]['fluxunit']))
                        #print("System temperature: {:.2f} {}".format(tsys, callist[cal]['fluxunit']))
                        #print("Source temperature: {:.2f} {}".format(tsrc, callist[cal]['fluxunit']))             

                # Fake recombination data - code moved earlier
            
                #def gaussian(xloc, mu, sigma):
                #    return np.exp(-np.power(xloc - mu, 2.) / (2 * np.power(sigma, 2.)))
            
                #if fake == "HII":
                #    if uylim != 0.0:
                #        amp = uylim
                #    else:
                #        amp = 1.0
                #    for vf in Hfreq:
                #        if (vf > fval[0]) and (vf < fval[-1]):
                #            qspec = qspec + amp*gaussian(fval, vf, 3.e-5*vf)/scale
             
                #if fake == "PsII":
                #    if uylim != 0.0:
                #        amp = uylim
                #    else:
                #        amp = 1.0
                #    for vf in Pfreq:
                #        if (vf > fval[0]) and (vf < fval[-1]):
                #            qspec = qspec + amp*gaussian(fval, vf, 1.e-3*vf)/scale
                        
                # Write data
        
                if output != "none":
                    # Prepare slice array
                    indo = [0]*ndims
                    indo[0]=ind
                    indo[-3] = pol
                    indo[-2]=slice(None)
                    indo[-1] = 0
    #                hduo_0SBa['data'][ind,0,pol,:,0] = qspec*scale
                    #print(indo)
                    #print(qspec.shape)
                    hduo_0SBa['data'][tuple(indo)] = qspec*scale

                if text != "none":
                    for it in range(len(fval)):
                        print(fval[it], qspec[it]*scale, file=tf)
                    tf.close()
                
                # Spectrum plot function
        
                def splot(nspec):
                    plt.figure(file+':'+str(pol))
                    plt.rc('font',family='serif')
                    cmult = 1.0
                    ylab = ''
                    if nspec == 1:
                        qqspec = qspec*scale
                        ylab = "Flux density (Jy)"
                        if np.amax(qqspec) < 1.0e-1:
                            cmult = 1.0e3
                            ylab = "Flux density (mJy)"
                        if np.amax(qqspec) < 1.0e-4:
                            cmult = 1.0e6
                            ylab = "Flux density (uJy)"
                        plt.plot(fval, qqspec*cmult, label=BEAM+" "+csubband)
        #                plt.plot(fval, qspec3*scale, label='cal-scaled quotient')
        #                plt.plot(fval, final_res, label='residual')
                    else:
                        #qqspec = vspec*scale # removed 21/03/2025
                        qqspec = qspec*scale
                        plt.plot(fval, qqspec, label=BEAM+" "+csubband)
                        if calavail:
                            cnew = dcalspec*scale*vspec_av/np.average(dcalspec)
                            plt.plot(fcval, cnew, label='scaled cal on-off (sig)')
                        if ref != 'none':
                            plt.plot(fval, vref*scale*vspec_av/vref_av, label='scaled reference')
                            if calravail:
                                crnew = dcalref*scale*vspec_av/np.average(dcalref)
                                plt.plot(fcval, crnew, label='scaled cal on-off (ref)')
                        if cont != 'none': 
                            plt.plot(fval, wref*vspec_av*scale/np.average(wref), label='scaled continuum')
                            plt.plot(fval, (vspec-final_res)*scale, label='reference + scaled continuum-reference')
                    plt.xlim([fval[0], fval[-1]])
                    if (uylim != 0.0) or (dylim !=0.0):
                        plt.ylim([cmult*dylim, cmult*uylim])
                    ymin,ymax = plt.ylim()
                    if markfreq != 0.0:
                        plt.plot([markfreq,markfreq], [ymin,ymax], '--')
                        dmark = 2.9
                        plt.fill([markfreq-dmark,markfreq+dmark,markfreq+dmark,markfreq-dmark], [ymin,ymin,ymax,ymax],alpha=0.2, color="orange")
                    plt.xlabel('Frequency (MHz)')
                    plt.ylabel(ylab)
                
                
                    rms = np.std(qqspec)
                    lqc = int(len(qqspec)/4)
                    rmsc = np.std(qqspec[lqc:3*lqc])
                    print("RMS of plotted spectrum: {:.4e}".format(rms))
                    print("RMS of central half    : {:.4e}".format(rmsc))
                    overm = (len(qqspec[qqspec>ymax])+len(qqspec[qqspec<ymin]))/(1.0*len(qqspec))
                    print("Fraction of spectra above and below requested vertical plot range:", overm)
                

                    # Overplotting
            
                    if vline:
                
                        # Standing waves

                        sw =5.55
                        nsw = int(abs(fval[-1]-fval[0])/sw)            
                        #for i in range(nsw):
                        #    plt.axvline(x=(fval[0]+(i+1)*sw), linestyle='--', color='lightgrey')
            
                        # Recombination lines

                        ttxtH = ymin+(ymax-ymin)/30.0
                        ttxtP = ymin+(ymax-ymin)/60.0
                        for vf in Hfreq:
                            if (vf > fval[0]) and (vf < fval[-1]):
                                plt.axvline(x=vf, linestyle='--', color='pink')
                                plt.text(vf, ttxtH, "H"+str(Hfreq.index(vf)+Hstart)+r"$\alpha$", horizontalalignment='center',fontsize='small')
                        for vf in Pfreq:
                            if (vf > fval[0]) and (vf < fval[-1]):
                                plt.axvline(x=vf, linestyle='--', color='lime')
                                plt.text(vf, ttxtP, "Ps"+str(Pfreq.index(vf)+Pstart)+r"$\alpha$", horizontalalignment='center',fontsize='small')
            
                        # Force vertical labels
            
    #                    plt.axvline(x=fval[0], linestyle='--', color='lightgrey', label='standing wave period')
                        plt.axvline(x=Hfreq[0], linestyle='--', color='pink', label=r'Hn$\alpha$ recombination lines')
                        plt.axvline(x=Pfreq[0], linestyle='--', color='lime', label=r'Psn$\alpha$ recombination lines')
            
                    plt.legend()
                    if not zoom:
                        ax = plt.twiny()
                        plt.xlim(plot_start_ch, plot_end_ch)
                    plt.show(block=block)
                    if not block:
                        plt.pause(0.5)
    #                plt.savefig("test.png")
                    plt.close()

                # Plot input spectra

                if not nodisplay:
                    if processed == False:
                        splot(0)
        
                # Time series plot function
        
                def tplot(nspec):
                    plt.figure(file+':'+str(pol))
                    plt.rc('font',family='serif')
                    if nspec == 1:
                        plt.plot(int_times[0]*range(ypix), vtspec*scale, label='quotient')
                    else:
                        plt.plot(int_times[0]*range(ypix), vtspec*scale, label='signal')
                        #if ref != 'none': plt.plot(range(ypix), vtref*np.average(vtspec)/np.average(vtref), label='scaled reference')
                        #if cont != 'none': plt.plot(range(wtref.shape[0]), wtref*np.average(vtspec)/np.average(wtref), label='scaled continuum')
                    plt.xlim([0, int_times[0]*ypix])
                    plt.xlabel('Time (sec)')
                    plt.ylabel('Flux density (Jy)')
                    plt.legend()
                    plt.show(block=block)
                    if not block:
                        plt.pause(0.5)
                    plt.close()

                # Plot input time series
        
                if (ypix > 1) and (not nodisplay):
                    tplot(0)
        
                # Plot calibrated spectrum
        
                if (ref != 'none') or (processed == True):
                    if not nodisplay:
                        splot(1)

                # Waterfall image
        
                if (ypix > 1) and (not nodisplay): 
                    plt.figure(file+':'+str(pol))
                    plt.rc('font',family='serif')
                    if (uylim != 0.0):
                        istat[4] = uylim/scale 
                    if (dylim != 0.0):
                        istat[0] = dylim/scale
                    plt.imshow(qdata,cmap=plt.cm.afmhot,aspect='auto',origin='lower',extent=extcor,vmin=istat[1],vmax=istat[3])
                    plt.xlabel('Frequency (MHz)')
                    plt.ylabel('Time (s)')
                    ax = plt.twiny()
                    plt.xlim(plot_start_ch, plot_end_ch)
                    plt.show(block=block)
                    if not block:
                        plt.pause(0.5)
                    plt.close()
        
        # End of suband loop
    # End of beam loop
    hdu.close()
    if ref != 'none':
        hdur.close()
    if cont != 'none':
        hduc.close()
    if output != "none":
        hduo.close()
# End of file loop
exit()

