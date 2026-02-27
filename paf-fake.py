#!/usr/bin/env python
#
# Lister Staveley-Smith
# 2025 June 4: copy a datset, replacing data with Gaussian noise from radiometer equation (using information on integration time, channel width an polarisation for each subband)

import h5py
import sys
import re
import numpy as np
import getopt
import os

# Tsys
tsys = 75.0

# Process command line parameters
sig = 'none'
output = 'none'
dfreq = 0.0


optionshort = '?s:o:d:'
optionlist = ['?','sig=', 'o=', 'd=']
optionhelp = ['help', 'input file', 'output file', 'delta frequency (MHz)']

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
    elif opt in ('-s', '--sig'):      # input filename
        sig = arg    
    elif opt in ('-o', '--out'):      # output filename
        output = arg    
    elif opt in ('-d', '--delta'):      # output filename
        dfreq = float(arg)    

if len(remainder) != 0:
    print('Unrecognised arguments:', remainder)

# Certain option combinations not allowed
if sig == 'none':
  exit('ERROR: need input file (-s <hdf filename>)')

if output == 'none':
  exit('ERROR: need output file (-o <hdf filename>)')
    
# History
history = " ".join(sys.argv)

# Open input file
try:
    hdu=h5py.File(sig, 'r')
except:
    exit("File not found: {}".format(sig))
BEAMS = [x for x in list(hdu.keys()) if "beam" in x]        

# Check that output file doesn't exist
if os.path.exists(output):
    exit('ERROR: output file already exists: {}'.format(output))

# Open output file
try:
    hduo=h5py.File(output, 'w')
except:
    exit("ERROR: cannot open output file: {}".format(output))

# Copy beam data over
print("Copying data and metadata...")
for BEAM in BEAMS:
    hdu.copy(BEAM, hduo)

# Copy configuration and metadata
if "configuration" in list(hdu.keys()): 
    hdu.copy('configuration', hduo)
if "metadata" in list(hdu.keys()):
    hdu.copy('metadata', hduo)

# Close input file
hdu.close()

# Loop through beams and subbands
print("Overwriting data assuming T(sys)={} ...".format(tsys))
for BEAM in BEAMS:
    SBS = [x for x in list(hduo[BEAM].keys()) if "band" in x]
    for SB in SBS:
        tshape =  hduo[BEAM][SB]['astronomy_data']['data'].shape
        if tshape[1] == 1:
            stokesI = True
        else:
            stokesI = False
        FREQ =  hduo[BEAM][SB]['astronomy_data']['frequency']
        sfreq = FREQ[0][0]
        if dfreq != 0.0:
            FREQ[0] += dfreq
            hduo[BEAM][SB]['astronomy_data']['frequency'][0,:] = FREQ[0]
        df = np.abs(FREQ[0,0]-FREQ[0,-1])/FREQ.shape[1]
        dt =  hduo[BEAM][SB]['metadata']['observation_parameters'][0][1]
        delta = tsys/np.sqrt(1.0e6*df*dt)
        if stokesI:
            delta /= np.sqrt(2)
        if (BEAM == BEAMS[0]):
            print(BEAM, SB, tshape, ":")
            if stokesI:
                print("Stokes I detected")
            print("Frequency spacing = {:.6g} MHz; integration time = {:.3f} s".format(df, dt))
            if dfreq != 0.0:
                print("Start frequency changed from {} to {} MHz".format(sfreq,sfreq+dfreq))
        if len(tshape) == 4:
            hduo[BEAM][SB]['astronomy_data']['data'][:,:,:,:] = np.random.normal(0.0, delta, tshape)
        else:
            exit("ERROR: data is not 4-dimensional")
print("...{} beams found".format(len(BEAMS)))
    
# Close output file
hduo.close()

exit()