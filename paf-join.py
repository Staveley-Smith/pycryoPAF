#!/usr/bin/env python
#
# Lister Staveley-Smith
# 2024 Nov 23: joins select hdfs if beam keys are different

import h5py
import sys
import re
import os


files = sys.argv[1:]
n = len(files)

if n == 0:
    exit("ERROR: no files specified")
if re.search(r"\*", files[0]):
    exit("ERROR: no files found")

# Invent output filename
output = files[0].split("_B")[0]+".hdf5"

# Check that output file doesn't exist
if output != "none":
    if os.path.exists(output):
        exit('ERROR: output file exists - {}'.format(output))

# Initialise
hdu = [""]*n
BEAMS = [""]*n

# Loop over files
for i in range(n):
    # Open readonly
    hdu[i] = h5py.File(files[i], 'r')
    BEAMS[i] = [x for x in list(hdu[i].keys()) if "beam" in x]
    try:
        TMP = hdu[i][BEAMS[i][0]]
    except:
        exit("ERROR: file can't be read: "+files[i])
    for j in range(i):
        for k in range(len(BEAMS[i])):
            if BEAMS[i][k] in BEAMS[j]:
                exit("ERROR: duplicate {} in {}".format(BEAMS[i][k], files[i]))

# Output file
hduo = h5py.File(output, 'w')
for i in range(n):
    for BEAM in BEAMS[i]:
        print("Copying beam data from {}".format(files[i]))
        hdu[i].copy(BEAM, hduo)
        if i > 0:
            hdu[i].close()

# Copy the first file's configuration and metadata
print("Copying non-beam metadata from {}".format(files[0]))
if "configuration" in list(hdu[0].keys()): 
    hdu[0].copy('configuration', hduo)
if "metadata" in list(hdu[0].keys()):
    hdu[0].copy('metadata', hduo)
hduo.close()
print("Output file: {}".format(output))

exit()