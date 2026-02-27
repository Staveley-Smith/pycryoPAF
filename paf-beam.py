#!/usr/bin/env python
#
# Lister Staveley-Smith

import numpy as np
import matplotlib.pyplot as plt
import os

# Parset search paths
path = ["./", "/Users/00060351/Drive/PROJECTS/CRYOPAF/COMMISSIONING/", "/Users/lss/Drive/PROJECTS/CRYOPAF/COMMISSIONING/"]

# Parset name
#parset = "3x3.parset"
#parset = "3x3-indexing-error.parset"
parset = "closepack72.parset"
parset = "hextile72.parset"
#parset = "grid72.parset"    # Use this when doing continuous recording during scan (cal observation) - or use -o option

# Parset defaults
offsets = -9                # Az, El offset vectors
pitch = -9                  # Beam spacing 
beamlabelerror = -9         # First 9-beam data had a beam labelling offset
#overwriterefbeam = False    # Choose whether to re-calculate RA, Dec even for optical axis (reference) beam (e.g. grid72.parset)

# Find, open and read parset
success = False
for i in range(len(path)):
    if os.path.isfile(path[i]+parset):
        ps = open(path[i]+parset)
        print("parset found: {}".format(path[i]+parset))
        for line in ps:
            if (line[:7] == "offsets") or ("pitch" in line) or ("beamlabel" in line) or ("overwriterefbeam" in line):
                exec(line)
        success = True
        break

if not success:
    exit("ERROR: parset file not found or not able to read - "+parset)
    
# Check parset parameters present
if (offsets == -9) or (pitch == -9) or (beamlabelerror == -9):
    exit("ERROR: parset parameters missing")

# Correct beam offsets to compensate for beam labelling error in data gathered using 3x3.parset 
if beamlabelerror == True:
    newoffsets = list(offsets)   # stop python from stupidly referencing itself
    newoffsets[:-1] = offsets[1:]
    newoffsets[-1] = offsets[0]
    offsets = list(newoffsets) # Copy back again!

# Apply pitch
xoff = pitch*np.array(offsets)

# Apply direction error
xoff = -xoff

# Figure out the reference beam (zero offset) 
refbeam = -1
for i in range(len(xoff)):
    if (xoff[i,0] == 0) and (xoff[i,1] == 0):
        refbeam = i 
if refbeam != -1:
    print("NOTE: parset file indicates that optical axis coincides with beam", refbeam)
else:
    print("NOTE: no beam is on the optical axis")

plt.rc('font',family='serif',size=14)
plt.rcParams['axes.linewidth'] = 2

# Beam plot
fig, ax = plt.subplots(num=parset,figsize=[6,6])
for i in range(len(xoff)):
    circle = plt.Circle(xoff[i], 0.1, fill=False, color='blue')
    ax.add_patch(circle)
    plt.text(xoff[i,0],xoff[i,1], str(i), ha='center', va='center', fontsize='small', fontname='sans-serif')
# Big circle for beam FWHP
circle = plt.Circle([-0.6,-0.6], 0.12, hatch='//', fill=False, color='red')
ax.add_patch(circle)
plt.axis('equal')
plt.xlabel(r'X offset (deg)', size=18)
plt.ylabel(r'Y offset (deg)', size=18)
plt.tight_layout()
plt.show()
plt.close()

exit()