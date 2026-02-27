#!/usr/bin/env python
#
# Lister Staveley-Smith
# 2024 Nov 25: subset hdfs and/or split hdf's by position (RA/DEC) near calibrator and removes integrations with drive errors; beam and subband selection permitted

import h5py
import sys
import re
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import numpy as np
import os
import getopt
import glob


# Split when position of first beam encountered changes by this much (deg)
trackerr = 0.05 # degrees

# Minimum offsource radius
trackoff = 0.5 # degrees

# Maximum offsource radius (probably need time limit too)
maxtrackoff = 2.0 # degrees

# Process command line parameters
sig = 'none'
qsub = False
asub = 0
bsub = 0
aBeam = 0
bBeam = 0
qBeam = False
history =""

optionlist = ['?','a=', 'b=', 'A=', 'B=', 'C', 'sig=']

optionhelp = ['help', 'start sub-band', 'end sub-band', 'start beam', 'end beam', 'force cal to be in group', 'input file']

try:
    options, remainder = getopt.getopt(sys.argv[1:], '?r:a:b:A:B:Cr:s:', optionlist)
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
        bBeam = int(arg)
    elif opt in ('-C', '--C'):      # start beam
        wcal = True
    elif opt in ('-s', '--sig'):    # input filenames
        sig = arg

if len(remainder) != 0:
    print('Unrecognised arguments:', remainder)

# Certain option combinations not allowed
if sig == 'none':
  exit('ERROR: need input file(s) (-s <hdf filename>)')

# Further option processing
if bsub == 0:
    bsub=asub+1
if bBeam == 0:
    bBeam=aBeam+1

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

for f in sigfiles:
    filecore = ".".join(f.split(".")[:-1])
    # Generate output filenames
    fileon = filecore+"_ON.hdf5"
    fileoff = filecore+"_OFF.hdf5"
    # Check that output files don't exist
    if os.path.exists(fileon):
        exit('ERROR: output file exists - {}'.format(fileon))
    if os.path.exists(fileoff):
        exit('ERROR: output file exists - {}'.format(fileoff))
    # Open readonly
    try:
        hdu = h5py.File(f, 'r')
    except:
        exit("ERROR: unable to read {}".format(f))
    BEAMS = [x for x in list(hdu.keys()) if "beam" in x]
    BEAM = BEAMS[0]
    SBS = [x for x in list(hdu[BEAM].keys()) if "band" in x]
    # Test if requested beams and subbands available
    sbeam = []
    if qBeam:
        for i in range(aBeam,bBeam):
            tbeam = "beam_{0:02d}".format(i)
            if tbeam in BEAMS:
                sbeam.append(tbeam)
            else:
                exit("ERROR: {} not present in {}".format(tbeam, f))
        BEAMS = sbeam
    ssub = []
    if qsub:
        for i in range(asub,bsub):
            tband = "band_SB{}".format(i)
            if tband in SBS:
                ssub.append(tband)
            else:
                exit("ERROR: {} not present in {}".format(tband, f))
        SBS = ssub
        
    # Figure out source name
    if "metadata" in list(hdu.keys()):
        SOURCE = hdu['metadata']['beam_parameters'][0]['SOURCE'].decode()
    else:
        exit("ERROR: no source name available")
    if "1934-638" in SOURCE:
        POS_CAL = SkyCoord('19:39:25.02' ,'-63:42:45.6', unit=(u.hourangle, u.deg), frame='icrs')
    elif "hydra" in SOURCE.lower():
        POS_CAL = SkyCoord('09:18:05.67' ,'-12:05:43.8', unit=(u.hourangle, u.deg), frame='icrs')
    elif "s8" in SOURCE.lower():
        POS_CAL = SkyCoord('207.0' ,'-15.0', unit=(u.deg, u.deg), frame='galactic')
    elif "s9" in SOURCE.lower():
        POS_CAL = SkyCoord('356.0' ,'-4.0', unit=(u.deg, u.deg), frame='galactic')
    elif "0407-658" in SOURCE.lower():
        POS_CAL = SkyCoord('04:08:20.38' ,'-65:45:09.1', unit=(u.hourangle, u.deg), frame='icrs')
    elif "0823-500" in SOURCE.lower():
        POS_CAL = SkyCoord('08:25:26.87' ,'-50:10:38.5', unit=(u.hourangle, u.deg), frame='icrs')
    else:
        exit("ERROR: no recognised calibrator source name")
    print("Calibration source {} recognised {}".format(SOURCE, POS_CAL))
    
    SB0 = SBS[0]
    j2on = []
    j2off = []
    for BEAM in BEAMS:
        try:
            TMP = hdu[BEAM][SB0]
        except:
            exit("ERROR: file can't be read: "+f)
        rf = f.split("/")[-1]
        # Read position metadata
        RA = hdu[BEAM][SB0]['metadata']['observation_parameters']['RIGHT_ASCENSION']
        DEC = hdu[BEAM][SB0]['metadata']['observation_parameters']['DECLINATION']
        DRIVE = hdu[BEAM][SB0]['metadata']['observation_parameters']['DRIVE_STATUS']
        # Convert to degrees
        RADEC = SkyCoord([x.decode() for x in RA], [x.decode() for x in DEC], unit=(u.hourangle, u.deg), frame='icrs')
        # Separations
        SEP = RADEC.separation(POS_CAL)
        # List of a list of data indices we wish to keep
        j1on=[]
        j1off = []
        nderr = 0
        for i in range(len(SEP)):
            if DRIVE[i] == 1:
                if SEP[i].value < trackerr:
                    j1on.append(i)
                if (SEP[i].value > trackoff) and (SEP[i].value < maxtrackoff):
                    j1off.append(i)
            else:
                nderr += 1
        j2on.append(j1on)
        j2off.append(j1off)
        print(BEAM)
        print("Will reject {} integrations with drive errors".format(nderr))
        print("Will keep {} ON integrations out of {}".format(len(j1on), len(SEP)))
        print("Will keep {} OFF integrations out of {}".format(len(j1off), len(SEP)))
    print("Writing {}...".format(fileon))
    # Open read/write!
    hduo = h5py.File(fileon, 'w')
    i = 0
    for BEAM in BEAMS:
        hduoB = hduo.create_group(BEAM)
        for SB in SBS:
            hduBS = hduoB.create_group(SB)
            hduBS.create_dataset('astronomy_data/data', data=hdu[BEAM][SB]['astronomy_data/data'][j2on[i]])
            hdu.copy(BEAM+'/'+SB+'/astronomy_data/frequency', hduBS['astronomy_data'])
            hduBS.create_dataset('metadata/observation_parameters', data=hdu[BEAM][SB]['metadata/observation_parameters'][j2on[i]])
        hdu.copy(BEAM+'/metadata', hduoB)
        # Also change metadata used by paf.py
        NMETA = hduoB['metadata']['band_parameters']['NUMBER_OF_INTEGRATIONS'].shape[0]
        META = []
        for j in range(NMETA):
            META.append(len(j2on[i]))
        hduoB['metadata']['band_parameters']['NUMBER_OF_INTEGRATIONS'] = META
        i += 1
    hdu.copy('configuration', hduo)
    hdu.copy('metadata', hduo)
    hduo.close()
    
    print("Writing {}...".format(fileoff))
    # Open read/write!
    hduo = h5py.File(fileoff, 'w')
    i = 0
    for BEAM in BEAMS:
        hduoB = hduo.create_group(BEAM)
        for SB in SBS:
            hduBS = hduoB.create_group(SB)
            hduBS.create_dataset('astronomy_data/data', data=hdu[BEAM][SB]['astronomy_data/data'][j2off[i]])
            hdu.copy(BEAM+'/'+SB+'/astronomy_data/frequency', hduBS['astronomy_data'])
            hduBS.create_dataset('metadata/observation_parameters', data=hdu[BEAM][SB]['metadata/observation_parameters'][j2off[i]])
        hdu.copy(BEAM+'/metadata', hduoB)
        # Also change metadata used by paf.py
        NMETA = hduoB['metadata']['band_parameters']['NUMBER_OF_INTEGRATIONS'].shape[0]
        META = []
        for j in range(NMETA):
            META.append(len(j2off[i]))
        hduoB['metadata']['band_parameters']['NUMBER_OF_INTEGRATIONS'] = META
        i += 1
    hdu.copy('configuration', hduo)
    hdu.copy('metadata', hduo)
    hduo.close()
    hdu.close()
exit()