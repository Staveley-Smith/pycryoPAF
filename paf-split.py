#!/usr/bin/env python
#
# Lister Staveley-Smith
# 2024 Nov 22: subset hdfs and/or split hdf's by position (RA/DEC) and removes integrations with drive errors; beam and subband selection permitted

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

# Minimum number of integrations at the same position to keep
trackmin = 2

# Process command line parameters
sig = 'none'
qsub = False
asub = 0
bsub = 0
aBeam = 0
bBeam = 0
qBeam = False
xbeam = 0
history =""

optionlist = ['?','a=', 'b=', 'A=', 'B=', 'sig=', 'X=']

optionhelp = ['help', 'start sub-band', 'end sub-band', 'start beam', 'end beam', 'input file', 'optical axis beam']

try:
    options, remainder = getopt.getopt(sys.argv[1:], '?r:a:b:A:B:s:X:', optionlist)
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
    # Open readonly
    try:
        hdu = h5py.File(f, 'r')
    except:
        exit("ERROR: unable to read {}".format(f))
    BEAMS = [x for x in list(hdu.keys()) if "beam" in x]
    strbeam = "beam_{0:02d}".format(xbeam)
    if strbeam in BEAMS:
        ixbeam = BEAMS.index(strbeam)
        BEAM = BEAMS[ixbeam]
    else:
        exit("ERROR: optical axis {} not available".format(cbeam))
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
    SB0 = SBS[0]
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
    SEP = RADEC.separation(RADEC[0])
    lSEP = len(SEP)
    # List of a list of data indices we wish to keep
    j1=[] 
    j2=[]
    nderr = 0
    for i in range(lSEP):
        if DRIVE[i] == 1:
            if SEP[i].value < trackerr:
                j2.append(i)
                # Don't forget the last group
                if (i >= lSEP-1):
                    j1.append(j2)
            else:
                if len(j2) > 0:
                    j1.append(j2)
                    j2 = []
                SEP = RADEC.separation(RADEC[i])
        else:
            nderr += 1
            if len(j2) > 0:
                j1.append(j2)
                j2 = []
            if (i+1) < len(SEP):
                SEP = RADEC.separation(RADEC[i+1])
    # Reject splits which result in too few integrations
    nterr = 0
    if trackmin > 1:
        for i in range(len(j1)):
            k = len(j1)-i-1
            if (len(j1[k])) < trackmin:
                del j1[k]
                nterr += 1 

    print("Rejecting {} integrations with drive errors".format(nderr))
    print("Rejecting {} split files as too short (<{} integrations)".format(nterr,trackmin+1))
    print("Generating {} split files".format(len(j1)))
    
    for i in range(len(j1)):
        # Generate filename
        file = filecore+"_G{0:02d}.hdf5".format(i)
        print("Writing {}...".format(file))
        # Check that output file doesn't exist
        if os.path.exists(file):
            exit('ERROR: output file exists - {}'.format(file))
        # Open read/write!
        hduo = h5py.File(file, 'w')
        for BEAM in BEAMS:
            hduoB = hduo.create_group(BEAM)
            for SB in SBS:
                hduBS = hduoB.create_group(SB)
                hduBS.create_dataset('astronomy_data/data', data=hdu[BEAM][SB]['astronomy_data/data'][j1[i]])
                hdu.copy(BEAM+'/'+SB+'/astronomy_data/frequency', hduBS['astronomy_data'])
                hduBS.create_dataset('metadata/observation_parameters', data=hdu[BEAM][SB]['metadata/observation_parameters'][j1[i]])
            hdu.copy(BEAM+'/metadata', hduoB)
            # Also change metadata used by paf.py
            NMETA = hduoB['metadata']['band_parameters']['NUMBER_OF_INTEGRATIONS'].shape[0]
            META = []
            for j in range(NMETA):
                META.append(len(j1[i]))
            hduoB['metadata']['band_parameters']['NUMBER_OF_INTEGRATIONS'] = META
        hdu.copy('configuration', hduo)
        hdu.copy('metadata', hduo)
        hduo.close()
    hdu.close()
exit()