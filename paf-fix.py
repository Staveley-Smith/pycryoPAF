#!/usr/bin/env python
#
# Lister Staveley-Smith
# 2024 Nov 7: edit positions and frequencies keep history
# Assumes all UT in all beams is the same (should check this)
# Converts TOPO frequencies to BARY frequencies at middle UT/RA/DEC (only one frequency axis per subband)
# 2025 Feb 26: simplified to just convert from Az/El

import h5py
import sys
import re
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, LSR, ICRS, Angle
from astropy.time import Time
import numpy as np
import getopt
import glob
import os


# Velocity of light
c = 2.99792458e8

#V_lsr 20 km/s to RA(B1900)=270deg, Dec(B1900)=+30deg or RA(J2000)=269.040535, Dec(J2000)=30.004646
V_lsr = 20.0
RA_lsr = 269.040535*np.pi/180.0
DEC_lsr = 30.004646*np.pi/180.0
V_lsr_cart = V_lsr*np.array([np.cos(RA_lsr)*np.cos(DEC_lsr), np.sin(RA_lsr)*np.cos(DEC_lsr), np.sin(DEC_lsr)])

# Largest tracking error (computed vs reference beam)
trackerr = 0.10

# Parset search paths
path = ["./", "/Users/00060351/Drive/PROJECTS/CRYOPAF/COMMISSIONING/", "/Users/lss/Drive/PROJECTS/CRYOPAF/COMMISSIONING/"]

# Parset name
#parset = "3x3.parset"
#parset = "3x3-indexing-error.parset"
parset = "closepack72.parset"
#parset = "grid72.parset"    # Use this when doing continuous recording during scan (cal observation) - or use -o option


# Process command line parameters
sig = 'none'
overwriterefbeam = False
name = 'none'
dry = False
qframe = False
posfix = False
framelist = ["BARY", "LSRK"]

optionlist = ['?','sig=', 'overwriteref', 'name=', 'dryrun', 'frame=', 'posfix']

optionhelp = ['help', 'input file(s)', 'overwrite reference beam coordinates', 'name of source', 'dry run (changes nothing)', 'reference frame (BARY/LSR etc.)', 'position (RA/Dec) re-calculation']

try:
    options, remainder = getopt.getopt(sys.argv[1:], '?s:or:n:dr:f:pr', optionlist)
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
    elif opt in ('-s', '--sig'):    # input filenames
        sig = arg
    elif opt in ('-o', '--overwriteref'): # overwrite reference beam coordinates (also see parset file)
        overwriterefbeam = True
    elif opt in ('-n', '--name'): # overwrite source name
        name = arg
    elif opt in ('-d', '--dryrun'): # dry run
        dry = True
    elif opt in ('-f', '--frame'): # reference frame (BARY, LSR etc.)
        qframe = True
        frame = arg
    elif opt in ('-p', '--posfix'): # reference frame (BARY, LSR etc.)
        posfix = True

if len(remainder) != 0:
    print('Unrecognised arguments:', remainder)

# Certain option combinations not allowed
if sig == 'none':
  exit('ERROR: need input file(s) (-s <hdf filename>)')

if overwriterefbeam and (not posfix):
    exit("ERROR: can only overwrite reference beam coordinates with -p option")

if qframe and (frame not in framelist):
    exit("ERROR: only valid frames are {}".format(framelist))

if (not qframe) and (not posfix):
    exit("ERROR: no re-calculation of any positions or frequencies requested")

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
files = filenameextractor(sig)
n = len(files)

if n == 0:
    exit("ERROR: no files specified")
if re.search(r"\*", files[0]):
    exit("ERROR: no files found")


if dry:
    print("WARNING: dry run - nothing will be changed")
    
# History
history = " ".join(sys.argv)

print("parset: {}".format(parset))

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

# Open and read parset
#try:
#    with open(parset) as ps:
#        for line in ps:
#            if (line[:7] == "offsets") or ("pitch" in line) or ("beamlabel" in line) or ("overwriterefbeam" in line):
#                exec(line)
#except:
#    exit("ERROR: parset file not found or not able to read - "+parset)
    
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
#    if off:
#        print("NOTE: a coordinate shift will be applied (except for this beam)")    
else:
#    if off:
#        print("NOTE: no beam in this SDHDF is on the optical axis - no tracking shift can be applied")
#        off = False
#    else:
#        print("NOTE: no beam in this SDHDF is on the optical axis")
    print("NOTE: no beam in this SDHDF is on the optical axis")

if overwriterefbeam:
    print("NOTE: all coordinates (inc. ref beam) will be initialised from Az, El values")
else:
    print("NOTE: all coordinates (excl. ref beam) will be initialised from Az, El values")

for f in files:
    # Open read/write!
    try:
        if dry:
            hdu=h5py.File(f, 'r')
        else:
            hdu=h5py.File(f, 'r+')
            
    except:
        exit("File not found: {}".format(f))
    BEAMS = [x for x in list(hdu.keys()) if "beam" in x]

    if (refbeam != -1):
        REFBEAM = 'beam_{:02d}'.format(refbeam)
    else:
        REFBEAM = ""
    # Warning if no. beams in a multibeam sdhdf don't match parset 
    if len(BEAMS) > 1:
        if len(BEAMS) > len(xoff):
            print("ERROR: {} beams ({}) greater than {} beams ({})".format(f, len(BEAMS), parset, len(xoff)))
            exit()
        if len(BEAMS) != len(xoff):
            print("WARNING: {} beams ({}) mis-matched to {} beams ({})".format(f, len(BEAMS), parset, len(xoff)))
        if REFBEAM not in list(hdu.keys()):
            exit("ERROR: Optical axis beam absent (multibeam SDHDF)")
        BEAM = REFBEAM
    else:
        BEAM = BEAMS[0]
        jbeam = int(BEAM[-2:])
        if jbeam >= len(xoff):
            exit("ERROR: {} exceeds parset limit ({} beams)".format(BEAM, len(xoff)))
        
    SBS = [x for x in list(hdu[BEAM].keys()) if "band" in x]

    SB0 = SBS[0]
    try:
        TMP = hdu[BEAM][SB0]
    except:
        exit("ERROR: file can't be read: "+f)
    rf = f.split("/")[-1]
    # PAF
    if "metadata" in list(hdu.keys()):
        SOURCE = hdu['metadata']['beam_parameters'][0]['SOURCE'].decode()
        CAL_MODE = hdu['metadata']['primary_header'][0]['CALIBRATION_MODE'].decode()
        if name != 'none':
            SOURCE = name
        SKEEP = hdu['metadata']['beam_parameters'][0]
        SKEEP[2] = name
    else:
        SOURCE = "N/A"
        CAL_MODE = "N/A"
    N_DUMPS = hdu[BEAM]['metadata']['band_parameters'][0]['NUMBER_OF_INTEGRATIONS']
    int_times = np.array([hdu[BEAM]['metadata']['band_parameters'][0]['REQUESTED_INTEGRATION_TIME']]*N_DUMPS)
    ltimes = len(int_times)
    int_time = np.sum(int_times)
    # Read reference beam metadata
    RA = hdu[BEAM][SB0]['metadata']['observation_parameters']['RIGHT_ASCENSION']
    DEC = hdu[BEAM][SB0]['metadata']['observation_parameters']['DECLINATION']
    UTC = hdu[BEAM][SB0]['metadata']['observation_parameters']['UTC']
    AZ = hdu[BEAM][SB0]['metadata']['observation_parameters']['AZIMUTH_ANGLE']
    EL = hdu[BEAM][SB0]['metadata']['observation_parameters']['ELEVATION_ANGLE']
    DRIVE = hdu[BEAM][SB0]['metadata']['observation_parameters']['DRIVE_STATUS']
    if "configuration" in list(hdu.keys()):
        # Precision a bit low
        #ITX = hdu['configuration']['telescope_configuration']['ITRF_X_COORDINATE']
        #ITY = hdu['configuration']['telescope_configuration']['ITRF_Y_COORDINATE']
        #ITZ = hdu['configuration']['telescope_configuration']['ITRF_Z_COORDINATE']
        #C8 = EarthLocation(x=ITX*u.m, y=ITY*u.m, z=ITZ*u.m)
        C8 = EarthLocation.of_site('parkes')
    else:
        print("Missing telescope_configuration metadata - assume data is from Parkes/Murriyang")
        C8 = EarthLocation.of_site('parkes')
    # Read and prepare to update history
    if not dry:
        if "metadata" in list(hdu.keys()):
            if ('proc_history' in hdu['metadata']):
                lh = hdu['metadata']['proc_history'].shape[0]
                hdu['metadata']['proc_history'].resize((lh+2,))
                hdu['metadata']['proc_history'][lh] = history
                hdu['metadata']['proc_history'][lh+1] = parset
            else:
                hdu['metadata'].create_dataset('proc_history', data=np.array([history,parset],dtype='|S256'), maxshape=(None,), chunks=True)
    MJD = hdu[BEAM][SB0]['metadata']['observation_parameters']['MJD']
    #MASK = np.zeros(len(UTC)).astype(bool)
    TMP = [""]*len(UTC)
    for i in range(len(UTC)):
        if (UTC[i].decode() == "0") or (AZ[i] == 0) or (EL[i] == 0):
            print("NOTIFICATION: corrupt UTC or Az or El detected")
            # In case integration not already flagged
            DRIVE[i] = 0
            TMP[i]  = Time(MJD[i], format='mjd').iso[:11] + "00:00:00.00"
        else:
            TMP[i] = Time(MJD[i], format='mjd').iso[:11] + UTC[i].decode()
    OBSTIME = Time(TMP, format='iso')
    
    # Az,El object; convert to RADEC
    AZEL = SkyCoord(alt=EL*u.degree, az=AZ*u.degree, frame='altaz', obstime=OBSTIME, location=C8)
    RADEC = AZEL.icrs
    # Spherical_offsets_to spits an error without this silly call
    RADEC2 = SkyCoord(RADEC.to_string('hmsdms'), frame='icrs')    
    
    # System RA, Dec columns
    RADEC_SYS = SkyCoord([x.decode() for x in RA], [x.decode() for x in DEC], unit=(u.hourangle, u.deg), frame='icrs')
    
    # If one beam on the optical axis, determine RA, DEC offsets [offset typically a few arcmin]
    if (BEAM == REFBEAM) and (not overwriterefbeam):
        OFF = RADEC2.spherical_offsets_to(RADEC_SYS)
        DRA = np.mean(OFF[0].value[DRIVE!=0])
        DDEC = np.mean(OFF[1].value[DRIVE!=0])
        print(f)
        print("Reference beam spherical offsets in RA, Dec = ({:.4f}, {:.4f}) deg".format(DRA,DDEC))
        if (np.abs(DRA) > trackerr) or (np.abs(DDEC) > trackerr):
            exit("ERROR: computed coordinate errors greater than {:.4f} deg".format(trackerr))
    else:
        DRA = 0.0
        DDEC = 0.0
     
    # Alter metadata in other beams, assuming feedangle is zero.
    for BEAMX in BEAMS:
        ibeam = int(BEAMX[-2:])
        # Beam offset
        AZELX = AZEL.spherical_offsets_by(xoff[ibeam,0]*u.deg, xoff[ibeam,1]*u.deg)
        RADECX = AZELX.icrs
        # Tracking offset (assume reference beam coordinates are correct)
        RADECX_CORR = RADECX.spherical_offsets_by(DRA*u.deg, DDEC*u.deg)
        RA_C = RADECX_CORR.ra.to_string('hr',sep=":")
        DEC_C = RADECX_CORR.dec.to_string(sep=":")
        #RA_C[MASK==True] = "0"
        #DEC_C[MASK==True] = "0"       

        # Write altered metadata (has to be whole array, not individual values)
        if (not dry) and posfix:
            if (BEAMX != REFBEAM) or overwriterefbeam:
                # Iterate over sub-bands
                for SB in SBS:
                    hdu[BEAMX][SB]['metadata']['observation_parameters']['RIGHT_ASCENSION'] = RA_C
                    hdu[BEAMX][SB]['metadata']['observation_parameters']['DECLINATION'] = DEC_C

        # Frequencies into barycentric frame using central UT and beam position
        if qframe:
            for SB in SBS:
                FREQ = hdu[BEAMX][SB]['astronomy_data']['frequency']
                FRAME = FREQ.attrs['FRAME']
                FRAMEIN = FRAME[0][2].decode()
                if FRAMEIN.upper()[:len(frame)] in frame.upper():
                    print("WARNING: {} {} already \'{}\' - no FREQUENCY conversion needed".format(BEAMX,SB, frame))
                elif FRAMEIN != "topocentric":
                    exit("ERROR: can only convert from topocentric but frame is {}".format(FRAMEIN))
                else:
                    MSG = [""]*2
                    imsg = 0
                    vcorrms = 0.0
                    ic = int(len(UTC)/2)
                    BARY = RADECX_CORR[ic].radial_velocity_correction(kind='barycentric', obstime=OBSTIME[ic], location=C8)
                    if BEAMX == BEAMS[0]:
                        if frame in framelist:
                            vcorrms += BARY.to(u.m/u.s).value
                            MSG[imsg] =  "Average TOPO->BARY correction ({},{})= {:.2f}".format(BEAMX,SB,BARY.to(u.km/u.s))
                            imsg += 1
                        if (frame == "LSRK"):
                            # The commented code seems ok, but default LSR (U,V,W) transformation is not the LSRK convention.
                            #icrs = ICRS(ra=Angle(RADECX_CORR[ic].ra.deg*u.deg), dec=Angle(RADECX_CORR[ic].dec.deg*u.deg), pm_ra_cosdec=0*u.mas/u.yr, pm_dec=0*u.mas/u.yr, radial_velocity=0.0*u.km/u.s, distance=10e6*u.pc)
                            #vlsr = icrs.transform_to(LSR()).radial_velocity
                            #print(icrs.transform_to(LSR()))
                            #vcorrms += vlsr.to(u.m/u.s).value
                            #MSG[imsg] = "Average BARY->LSRK correction ({},{})= {:.2f}".format(BEAMX,SB,vlsr.to(u.km/u.s))
                            vlsr = V_lsr_cart[0]*np.cos(RADECX_CORR[ic].ra.rad)*np.cos(RADECX_CORR[ic].dec.rad) + V_lsr_cart[1]*np.sin(RADECX_CORR[ic].ra.rad)*np.cos(RADECX_CORR[ic].dec.rad) + V_lsr_cart[2]*np.sin(RADECX_CORR[ic].dec.rad)
                            vcorrms += vlsr
                            MSG[imsg] = "Average BARY->LSRK correction ({},{})= {:.2f} km / s".format(BEAMX,SB,vlsr)
                        # Add history
                        for msg in MSG:
                            if msg != '':
                                print(msg)
                                if not dry:
                                    lh = hdu['metadata']['proc_history'].shape[0]
                                    hdu['metadata']['proc_history'].resize((lh+1,))
                                    hdu['metadata']['proc_history'][lh] = msg
                    # Update hdf
                    if not dry:
                        if frame == "BARY":
                            FRAME[0][2] = "barycentric"
                        elif frame == "LSRK":
                            FRAME[0][2] = "LSR"
                        hdu[BEAMX][SB]['astronomy_data']['frequency'].attrs['FRAME'] = FRAME
                        hdu[BEAMX][SB]['astronomy_data']['frequency'][0,:] = FREQ[0]/(1.0+vcorrms/c)
                
                
    # Source name
    if not dry:
        if name != 'none':
            hdu['metadata']['beam_parameters'][0] = SKEEP
            
    # Read altered metadata
    RA_N = hdu[BEAM][SB0]['metadata']['observation_parameters']['RIGHT_ASCENSION']
    DEC_N = hdu[BEAM][SB0]['metadata']['observation_parameters']['DECLINATION']

    # Header
    if n != 1:
      nstr = ""
    if f == files[0]:
      if "metadata" in list(hdu.keys()):
          hdrv = hdu['metadata']['primary_header'][0]['HEADER_DEFINITION_VERSION'].decode()
      else:
          hdrv = "N/A"
      #print("HDR_DEFN_VERSION (first file):", hdrv)
      #print('{:21.21} {:20.20} {:} {:} {:10.10} {:9.9} {:3.3} {:5.5}'.format('  HDF file', ' Source', 'B', 'I','  RA', '  Dec', 'cal', '  sec'))
    # Print summary
    for i in range(ltimes):
      t1 = hdu[BEAM][SB0]['metadata']['observation_parameters'][i]['MJD']
      t2 = Time(t1, format='mjd')
      UT_DATE = t2.to_value(format='iso', subfmt='date')
      if (DEC_N[i].decode()[0] != "-") and (DEC_N[i].decode()[0] != "+"):
          DEC_N[i] = b"+"+DEC_N[i]
      if n == 1:
          it = int_times[i]
          nstr = "("+str(i)+")"
          rf = UT_DATE+"-"+UTC[i].decode()
      else:
          it = int_time
      #print('{:21.21} {:20.20} {:} {:} {:10.10} {:9.9} {:3.3} {:.1f} {:}'.format(rf, SOURCE, len(BEAMS), len(SBS), RA_N[i].decode(), DEC_N[i].decode(), CAL_MODE, it, nstr))
      if (not dry) and posfix:
          if overwriterefbeam or BEAM!=REFBEAM:
              if ((RA_N[i].decode()[:10] != RA_C[i][:10]) or (DEC_N[i].decode()[:9] != DEC_C[i][:9])) and (DRIVE[i] != 0):
                  print("Requested coords: {} {} vs retrieved coords: {} {}".format(RA_C[i][:10], DEC_C[i][:9], RA_N[i].decode()[:10], DEC_N[i].decode()[:9]))
                  print("ERROR: update failed at cycle {} {} {}".format(i,BEAM,SB0))
      if n > 1:
          break
              
    hdu.close()
exit()