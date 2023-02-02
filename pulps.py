# newgrp ugmrtpsr && umask 0007
# source /Data/bcj/INPTA/soft/Pulsar/pulsarbashrc.sh



_version_ = '19-11-2022' # Please update after each edits

# Importing Libraries
import psrchive

import sys
import time
import os
import os.path
import argparse
import math
import numpy as np
from numpy import array, exp, sin
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Enabling LaTeX
from scipy.ndimage import shift # ?
from scipy.interpolate import interp1d # returns the 1D interpolation function, input is x and y data in order(x,y)
from lmfit import Minimizer, Parameters, report_fit, Model # Least square fitting of DM time series with the given model
from astropy.io import fits # Module to load handle FITS files
from astropy.io.fits import update # To update the FITS file with data and header of a particular HDU extension

# Help message and CLI
parser = argparse.ArgumentParser(prog='pulps.py',
                            usage='''\033[96m python pulps.py [FLAG] [OPTION] [FILE]... \033[00m\nPlease use -h command for help
                            ''',
                            description =  ''' 
----------------------------------------------------------------------------------------
\033[93m #######  ##    ##  ##     ########  ####### \033[00m
\033[93m ##    #  ##    ##  ##      ##    #  ##	     \033[00m
\033[93m #######  ##    ##  ##      #######  ####### \033[00m
\033[93m ##       ##    ##  ##      ##            ## \033[00m
\033[93m ##       ########  ######  ##       ####### \033[00m
----------------------------------------------
(PULsar Profile Simulator) - Developed by InPTA (Indian Pulsar Timing Array)

This program simulates PSRFITS files for a given pulsar for testing different
DM determination schemes. The SNR determined from the given pulsar data is used
to generate a pulse of any shape that the user defines with the amount of noise
based on the data. The supplied DM time series is fitted by a polynomial model
and a distribution for DM noise is worked out for simulations. DM is generated
as a random variable for each epoch from this and is used to generate simulated
data for each epoch. This is repeated for a number of times as the user defines
and in the end simulated observations for analysis are obtained.

Additionally, measurements of frequency scaling index alpha for the given pulsar
are read from a file and a polynomial fit to model trend is done and a distribution
for alpha noise is worked out for simulations. alpha is generated as a random
variable for each epoch from this and is used to generate a simulated data for each
epoch. Furthermore, alpha variation of scatter broadening with frequency for all
epochs from simulated distribution is used to scatter broadened the pulse.

For more details and examples, refer to the SOP : \033[93m https://bit.ly/3wRu4DD \033[00m
Dependencies: PSRCHIVE, TEMPO-2, Astropy and all standard Python libraries.
\033[91mPiyush 14-6-2022 \033[00m
----------------------------------------------------------------------------------------
                            '''                            
                                 ,formatter_class=argparse.RawDescriptionHelpFormatter,
                            add_help=True
                            )
# Define all command line arguments

parser.add_argument('-V','-v','--version', action='version', version='%(prog)s '+_version_)

group1 = parser.add_argument_group('\033[96m Flags to control type of scatter-broadening \033[00m')
group2 = parser.add_argument_group('\033[96m Arguments initialised by the input file by default \033[00m')
group1.add_argument('-noscat', '-ns',
                            help = "Scatter-broadening will NOT be included in simulation",
                            action="store_true")
group1.add_argument('-conscat', '-cs',
                            help = "Constant Scatter-broadening will be included in simulation",
                            action="store_true")
group1.add_argument('-varscat', '-vs',
                            help = "Variable Scatter-broadening will be included in simulation",
                            action="store_true")
#1
group2.add_argument('-telescope',
                            help = 'Telescope name for TEMPO2',
                            type = str)
#2
group2.add_argument('-tempo2path',
                            help = 'Path to TEMPO2',
                            type = str)
#3
group2.add_argument('-dmseries',
                            help = "DM time series file, allowed inputs are .dm file/constant/POWER",
                            type = str)
#4
group2.add_argument('-dmerr',
                            help = "DM error-bound: maximum permissible error in DM time series",
                            type = float)
#5
group2.add_argument('-alphaseries',
                            help = "alpha time series file,allowed inputs are .alpha file/constant ",
                            type = str)
#6
group2.add_argument('-alphaerr',
                            help = "alpha error-bound: maximum permissible error in alpha time series",
                            type = float)
#7
group2.add_argument('-template',
                            help = 'Template .fits file',
                            type = str)
#8
group2.add_argument('-par',
                            help = 'Ephemeris .par file',
                            type = str)
#9
group2.add_argument('-MJDST',
                            help = 'Starting MJD for simulation',
                            type = float)
#10
group2.add_argument('-nmjd',
                            help = 'Number of simulated fits files',
                            type = int)
#11
group2.add_argument('-cadence',
                            help = 'Cadence of simulated observations in days',
                            type = int)
#12
group2.add_argument('-hf',
                            help = 'Higher cut-off frequency for observation band',
                            type = float)
#13
group2.add_argument('-lf',
                            help = 'Lower cut-off frequency for observation band',
                            type = float)
#14
group2.add_argument('-snr',
                            help = 'SNR of the simulated pulse profiles',
                            type = float)
#15
group2.add_argument('-nbin',
                            help = 'Number of phase-bins in simulated profiles',
                            type = int)
#16
group2.add_argument('-nchan',
                            help = 'Number of frequency channels',
                            type = int)
#17
group2.add_argument('-D',
                            help = 'Value of dispersion measure in parsec cm^-3 to be used',
                            type = float)
#18
group2.add_argument('-tau300mhz',
                           help = 'Constant pulse-broadening time scale at 300MHz',
                           type = float)
#19
group2.add_argument('-alpha',
                           help = 'Constant pulse-broadening frequency scaling index',
                           type = float)
#20
group2.add_argument('-PSR',
                           help = 'Pulsar\'s J name, e.g. J1643-1224',
                           type = float)
#21
group2.add_argument('-amp',help = 'amplitude in powerspectrum model, only considered when the dmseries is a power model, default = 1e-6', type = float)

#22
group2.add_argument('-beta',help = 'beta in powerspectrum model, only considered when the dmseries is a power model, default = -2.0', type = float)

#23
group2.add_argument('-WIDTH',help = 'width of pulse in pulsemaker, default = 1.0', type = float)

#24
group2.add_argument('-SIGWIDTH',help = 'width of signal from pulsemaker. Should be positive, default = 3.0', type = float)

# UPGRADE : ADD HERE MORE ARGUMENTS
args = parser.parse_args()                                                        

if (os.path.exists('./getperiodPredictor')):
        print("[INFO] : 'getperiodpredictor' is found in the working directory")
        pass
else:
        print("\033[91m [ERR] \033[00m : File 'getperiodPredictor' is not available, please generate/move the file in the same directory as this script.")
        exit()

if (os.path.exists('./getphasePredictor')):
        print("[INFO] : 'getphasepredictor' is found in the working directory")
        pass
else:
        print("\033[91m [ERR] \033[00m : File 'getphasePredictor' is not available, please generate/move the file in the same directory as this script.")
        exit()

# Read from Input file
inputfile = []
try:
    open("pulps.in", "r")
except EnvironmentError:
    print("\033[91m [ERR] \033[00m : File pulps.in is not available, please generate/move the file in the same directory as this script.")
    exit()
with open("pulps.in", "r") as arguments:
    for lines in arguments:
        if not lines.strip().startswith('#'):
            row = lines.split()
            if row != []:
                inputfile.append(row[1])
if len(inputfile) != 24: # Number of input parameters
    print('\033[91m [ERR] \033[00m : Some input parameter is missing... Please check pulps.in file in your working directory')
    exit()


#setting default values (fazal)
amp = 1e-6
beta = -2.0
width = 1.0
sigwidth = 3

# Initialise input files from command line or defaults from input file
if args.tempo2path:
    tempo2path = args.tempo2path
else:
    tempo2path = str(inputfile[0]) # Default
print('[INFO] : Path to TEMPO2 is {}'.format(tempo2path))
if args.dmseries:
    dmfile = args.dmseries    
else:
    dmfile = str(inputfile[1]) # Default
print('[INFO] : DM time-series file used is {}'.format(dmfile))
if dmfile == 'constant':
    print('[INFO] : Amplitude and beta will not be used')
if args.amp:
    amp = args.amp
elif inputfile[2] != 'NONE': #if inputfile is NONE then the default values will be taken
    amp = float(inputfile[2])
print('[INFO] : DM powerspectrum amplitude used is{}'.format(amp))   
if args.beta:
    beta = args.beta
elif inputfile[3] != 'NONE':  #if inputfile is NONE then the default values will be taken
    beta = float(inputfile[3]) # Default
print('[INFO] : DM powerspectrum beta used is{}'.format(beta))
if args.dmerr:
    dmerrbound = args.dmerr
else:
    dmerrbound = float(inputfile[4]) # Default
print('[INFO] : DM error-bound used is {}'.format(dmerrbound))
if args.alphaseries:
    alphafile = args.alphaseries
else:
    alphafile = str(inputfile[5]) # Default
print('[INFO] : Alpha time-series file used is {}'.format(alphafile))
if args.alphaerr:
    alphaerrbound = args.alphaerr
else:
    alphaerrbound = float(inputfile[6]) # Default
print('[INFO] : Alpha error-bound used is {}'.format(alphaerrbound))
if args.template:
    template = args.template
else:
    template = str(inputfile[7]) # Default
print('[INFO] : Template file used is {}'.format(template))
if args.par:
    par = args.par
else:
    par = str(inputfile[8]) # Default
print('[INFO] : Ephermeris file used is {}'.format(par))
if args.MJDST:
    MJDST = args.MJDST
else:
    MJDST = float(inputfile[9]) # Default
print('[INFO] : Starting MJD is {}'.format(MJDST))
if args.nmjd:
    nmjd = args.nmjd
else:
    nmjd  = int(inputfile[10]) # Default
print('[INFO] : Number of simulated fits files is {}'.format(nmjd))
if args.cadence:
    cadence = args.cadence
else:
    cadence = int(inputfile[11]) # Default
print('[INFO] : Cadence in days is {}'.format(cadence))
if args.hf:
    hf = args.hf
else:
    hf = float(inputfile[12]) # Default
print('[INFO] : Higher cut-off frequency is {} MHz'.format(hf))
if args.lf:
    lf = args.lf
else:
    lf = float(inputfile[13]) # Default
print('[INFO] : Lower cut-off frequency is {} MHz'.format(lf))
if args.snr:
    snr = args.snr
else:
    snr = int(inputfile[14]) # Default
print('[INFO] : SNR is {}'.format(snr))
if args.nbin:
    nbin = args.nbin
else:
    nbin = int(inputfile[15]) # Default
print('[INFO] : Number of phase-bins is {}'.format(nbin))
if args.nchan:
    nchan = args.nchan
else:
    nchan = int(inputfile[16]) # Default
print('[INFO] : Number of frequency channels is {}'.format(nchan))
if args.D:
    DM = args.D
else:
    DM = float(inputfile[17]) # Default
print('[INFO] : DM value used is {} parsec cm^-3'.format(DM))
if args.tau300mhz:
    tau300mhz = args.tau300mhz
else:
    tau300mhz = float(inputfile[18]) # Default
print('[INFO] : Pulse-broadening time scale at 300MHz is {}'.format(tau300mhz))
if args.alpha:
    alpha = args.alpha
else:
    alpha = float(inputfile[19]) # Default
print('[INFO] : Pulse-broadening frequency scaling index is {}'.format(alpha))
if args.telescope:
    telescope = args.telescope
else:
    telescope = str(inputfile[20]) # Default
print('[INFO] : Telescope name is {}'.format(telescope))
if args.PSR:
    PSR = args.PSR
else:
    PSR = str(inputfile[21]) #Default
print('[INFO] : Pulsar is {}'.format(PSR))
if args.WIDTH:
    width = args.WIDTH
elif inputfile[22] != 'NONE':  #if inputfile is NONE then the default values will be taken (1.0)
    width = float(inputfile[22])
print('[INFO] : Width of the Pulse is {}'.format(width))
if args.SIGWIDTH:
    sigwidth = args.SIGWIDTH
elif inputfile[23] != 'NONE':  #if inputfile is NONE then the default values will be taken(3)
    sigwidth = float(inputfile[23]) #Default
print('[INFO] : Width of the signal is {}'.format(sigwidth))


#UPGRADE : DEFINE HERE MORE ARGUMENTS

def line(x,sl,intcpt): # sl is slope and intcpt is intercept
    return x*sl+intcpt

# Creates a power law spectrum.
def power(freqs, amp, beta):
    return amp * freqs ** beta

# Creates a timeseries from the power spectrum.
def make_timeseries(amp, beta, ndays):
	
	"""
	Creates a timeseries using the given spectral index (beta) and amplitude
	(amp). Random noise is also added to the spectrum.
	"""
	nfreqs = 10000
	freqs = np.linspace(1e-3,2,nfreqs)
	powersp   = power(freqs,amp,beta) * np.random.rand(len(freqs))
	fouriersp = np.sqrt(powersp) * np.exp(1j*np.pi*np.random.rand(len(powersp)))
	Tseriesfft = np.concatenate(([0],fouriersp,np.conj(fouriersp[::-nfreqs])))
	Tseries = np.real(np.fft.ifft(Tseriesfft))[1:nfreqs+1]
	Tseries = Tseries[:-100]; Tseries = Tseries[100:]
	freqs = freqs[:-100]; freqs = freqs[100:]

	phase_points = np.linspace(1e-3,1,ndays)
	Tseries = np.interp(phase_points, freqs, Tseries)
	return (Tseries)


# Creates DM series based on the input
def make_dmseries(n):
    """
    Creates a DM time series based on the amplitude and power given in the input
    """
    # amp = 1e-6; power = -2.0
    print ("Using powerspectrum model (amp * freqs ** beta), amplitude is taken as", amp)
    print ("beta is taken as", beta)
    dmseries = np.zeros(n)
    dmseries = make_timeseries(amp, beta, n)
    return(dmseries)
# extra functions UPGRADE
# a) just constant DM (done) give option -dmseries constant.
# b) if user wants DM simulation
# c) if user wants DM from function (read from file OR generate)

def dmsimulate(dmfile):
    if dmfile == 'NONE':
        print('No DM time series given. Exiting...')
        exit()

    if dmfile == 'POWER':
        dm = make_dmseries(nmjd)
        MJD = np.arange(MJDST,MJDST+nmjd)
        print('DM file is = ', dm)
        print('MJD is', MJD)
        dmerr = nmjd * [dmerrbound]
        x = MJD
        y = dm
        err = dmerr

    if dmfile == 'constant':
        MJD = np.arange(MJDST,MJDST+nmjd)
        dm = nmjd * [DM]
        print('DM will be taken to be constant = ', DM )
        dmerr = nmjd * [dmerrbound]
        print('DM error will be taken to be constant = ', dmerrbound)
        print('DM file is = ', dm)
        print('MJD is', MJD)
        x = MJD
        y = dm
        err = dmerr
    else:
    # Read DM Time-series
        MJD, dm, dmerr = np.loadtxt(dmfile, comments='#', usecols=(0,1,2), unpack=True) # DM comes from 12.5 yr nangrav data
        # Throw out some DMs by this condition (because of bad measure and bias)
        condition = dmerr < dmerrbound
        x= np.extract(condition,MJD)
        y= np.extract(condition,dm)
        err= np.extract(condition, dmerr)

    # Plot1
    # plt.errorbar(x,y,yerr=err,fmt='o')
    # plt.xlabel("MJD")
    # plt.ylabel("DM ($pc\,cm^{-3}$)")
    #plt.savefig('dm_timeseries.png')

    # UPGRADE
    # linear for 1643 but not a general thing for all pulsars
    # thus we may want the user to specify some model (sum of several sinusoids - fourier series)
    # some command line argument for the fourier series

    pwrmod = Model(line) # Name of model object
    pars = pwrmod.make_params(sl=0.0, intcpt=0) # Parameters for the model (initial values and must be best possible guesses, e.g. peak of gaussian and position)
    result = pwrmod.fit(y, pars, weights=err,x=x) # Fitting, we may want to do exception handling IMPORTANT
    #print('')
    #print('Fitting report of DM Time-series:')
    #print(result.fit_report()) # result is printed as report in a file tempout

    dmtrendsl = result.params.get('sl').value # Value of fitted slope
    dmtrendintcpt = result.params.get('intcpt').value # Value of fitted intercept
    yfit=x*dmtrendsl+dmtrendintcpt # Linear Fit
    dmres = y - yfit # DM residuals

    # Plot2
    #plt.errorbar(x,y,yerr=err,fmt='o')
    #plt.plot(x,yfit)
    #plt.xlabel("MJD")
    #plt.ylabel("DM ($pc\,cm^{-3}$)") 
    #plt.savefig('dm_timeseries_fit.png')

    # Plot3
    #plt.errorbar(x,dmres,yerr=err,fmt='o')
    #plt.xlabel("MJD")
    #plt.ylabel("DM ($pc\,cm^{-3}$)")

    empcdfx,empcdfy = ecdf(dmres)
    inv_cdf = interp1d(empcdfy,empcdfx, bounds_error=False, assume_sorted=True)
    r = np.random.uniform(0, 1, nmjd)
    ynew = inv_cdf(r)
    ynew[np.isnan(ynew)] = np.nanmean(ynew)	# Piyush's implementation
        #print(ynew)
    xnew = MJDST+np.arange(nmjd)*cadence
    #gnew = xnew
    dmsimorig = xnew*dmtrendsl+dmtrendintcpt + ynew 
        #print(dmsimorig)
    dmsim = dmsimorig[np.logical_not(np.isnan(dmsimorig))] # Simulated DMs
    xnew = xnew[np.logical_not(np.isnan(dmsimorig))]
    #for i in range(520):
    #    print(xnew[i],gnew[i])
    #if (len(xnew)!=nmjd):
    #    exit() 		
    #nmjd=dmsim.shape[0]
    outarr = np.transpose([xnew,dmsim])
    np.savetxt('dmsim.dat',(outarr), fmt='%12.6f %9.5f', delimiter=' ', newline='\n')
    xxnew=xnew
    ddmsim=dmsim
    
    # Plot4
    #plt.plot(xxnew,ddmsim, 'ko')
    #plt.xlabel('Epoch (MJD)')
    #plt.ylabel('DM')
    #plt.title('Simulated DM ($pc\,cm^{-3}$)')
    #plt.tight_layout() # ISSUE : latex error in my system
    #plt.savefig("dmsim.pdf", format='pdf',bbox_inches='tight')
    
    # Plot5
    #plt.plot(xnew,dmsim)
    #plt.xlabel("MJD")
    #plt.ylabel("DM ($pc\,cm^{-3}$)") 
    #plt.savefig('dm_timeseries_simulated.png')
    
    return dmsim

	# Gaussian pulse generator
	# Generate a unity amplitude Gaussian pulse centered 64 at with width of 6
	# Convolve with exponential scatter broadening PBF with nu^-4.4 law
	# UPGRADE some other pulse (1. gaussian pulse, other arbitrary (combination of gaussian), or read from file)

def pulsemaker():
    pulse=np.zeros(nbin)  # If user wants pulse evolution then this section will be skipped, after this the pulse will be defined for profile evolution
    for ibin in range(nbin):
        xibin=float(ibin)
        pulse[ibin]=math.exp(-(((xibin-(nbin/2.0))/width)**2)) # the width (1.0 here) can also be a argument # for profile evolution this pulse will be func of ibin AND ichan ###changed by fazal to width argument
        signal=np.sum(pulse[int((nbin/2)-sigwidth):int((nbin/2)+sigwidth)]) # the width (3 here) can also be a argument ####fazal Changed it into int
        # Add noise for each band using SNR and make band pulse
    pulse = pulse + np.random.normal(0.0,signal/snr,nbin)
    return pulse

# Simulating alpha values from given data
def alphasimulate(alphafile):

    if alphafile == 'NONE':
        print('No alpha time series given. Exiting...')
        exit()

    if alphafile == 'CONSTANT':
        MJD = np.arange(MJDST,MJDST+nmjd)
        alphafile = nmjd * [alpha]
        print('Alpha will be taken to be constant = ', alpha )
        alphaerr = nmjd * [alphaerrbound]
        print('Alpha error will be taken to be constant = ', alphaerrbound)
        print('Alpha file is = ', alphafile)
        print('MJD is', MJD)
        x = MJD
        y = alphafile
        err = alphaerr
    else:
        # Read alpha time-series
        MJD, alphafile, alphaerr = np.loadtxt(alphafile, comments='#', usecols=(0,1,2), unpack=True)
        # Alpha comes from measurements with uGMRT
        condition = alphaerr < alphaerrbound
        x = np.extract(condition,MJD)
        y = np.extract(condition,alphafile)
        err = np.extract(condition,alphaerr)

    # Different modelling options for alphasim (sine etc)
    mean=np.mean(y)
    alphares = y - mean
    yyy=np.ones(x.shape[0])*mean

    # Plot6
    #plt.errorbar(x,y,yerr=err,fmt='o')
    #plt.plot(x,yyy)
    #plt.xlabel("MJD")
    #plt.ylabel("alpha") 
    #plt.savefig('alpha_timeseries.png')

    # Plot7
    #plt.errorbar(x,alphares,yerr=err,fmt='o')
    #plt.xlabel("MJD")
    #plt.ylabel("alpha") 
    #plt.savefig('alpha_timeseries_residuals.png')

    empcdfx,empcdfy = ecdf(alphares)
    inv_cdf = interp1d(empcdfy,empcdfx, bounds_error=False, assume_sorted=True) 
    r = np.random.uniform(0, 1, nmjd)
    ynew = inv_cdf(r)
    ynew[np.isnan(ynew)] = np.nanmean(ynew) # Piyush's implementation
    xnew = MJDST+np.arange(nmjd)*cadence
    alphasimorig = ynew  + mean
    alphasim = alphasimorig[np.logical_not(np.isnan(alphasimorig))]
    xnew = xnew[np.logical_not(np.isnan(alphasimorig))] 
    #nmjd=alphasim.shape[0]
    outarr = np.transpose([xnew,alphasim])
    np.savetxt('alphasim.dat',(outarr), fmt='%12.6f %9.5f', delimiter=' ', newline='\n')
    xxnew=xnew
    aalphasim=alphasim
    
    # Plot8
    #plt.plot(xxnew,aalphasim, 'ko')
    #plt.xlabel('Epoch (MJD)')
    #plt.ylabel('alpha')
    #plt.title('Simulated alpha')
    #plt.tight_layout()
    #plt.savefig("alphasim.pdf", format='pdf',bbox_inches='tight')
    
    return alphasim

# Container making function for arbitrary band
def container(PSR, nchan, nbin, DM, parfile, template):
    # Prepare container fits file
    # Read template file, set subint as 1 and nchan and nbin as specified set DM value and install ephemeris, all using pam. 
    cmd = "pam -e mod.fits -a PSRFITS --reverse_freqs --setnsub 1 --setnchn {} --setnbin {} -d {} -E {} {}".format(nchan,nbin, DM, parfile, template)
    print("[CMD] : {}".format(cmd))
    os.system(cmd)
    	
    newfits = template.rstrip('fits')
    nfits=newfits+'mod.fits'

    #fits_image_filename1 = template
    #hdul1 = fits.open(fits_image_filename1)
    #hdr1  = hdul1[3].header
    #data1 = hdul1[3].data 
    #print(hdul1.info())
    #print(hdr1)
    #print(data1)
    #print(len(data1))

    #fits_image_filename2 = nfits
    #hdul2 = fits.open(fits_image_filename2)
    #hdr2  = hdul2[3].header
    #data2 = hdul2[3].data 
    #print(hdul2.info())
    #print(hdr2)
    #print(data2)
    #print(len(data2))

    cmd = "psredit -m -c name={} {}".format(PSR, nfits)
    print("[CMD] : {}".format(cmd))
    os.system(cmd)

    cmd = "pam -e scr.fits -D {}".format(nfits)
    print("[CMD] : {}".format(cmd))
    os.system(cmd)

    newfits = template.rstrip('fits')
    nfits=newfits+'mod.scr.fits' # what is the usage of this file

    strasc = "{}\n".format(nchan)
    flist = open("band_freq","w")
    flist.write(strasc)
    flist.close()

    cmd = "pdv -T -A {} | grep MJD  | awk '{{print $6}}' >> band_freq ".format(nfits)
    print("[CMD] : {}".format(cmd))
    os.system(cmd)
	
# Empirical cumulative distribution function ?
def ecdf(x): 
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def convolve_funcs(func1, func2): # from KK's code

	"""
	Algorithm implementing circular convolution of two functions.
	Currently using the FFT based convolution as it is way faster
	than the usual way.
	"""

	# Checking the size of the arrays and zero padding if one of 
	# them is shorter than the other.
	n1 = np.size(func1)
	n2 = np.size(func2)

	if n2 < n1:
		zeros = np.zeros(n1-n2).tolist()
		func2 = func2.tolist() + zeros
		func2 = np.asarray(func2, dtype=float)

	if n1 < n2:
		zeros = np.zeros(n2-n1).tolist()
		func1 = func1.tolist() + zeros
		func1 = np.asarray(func1, dtype=float)

	# Taking the fft of both the functions
	f1_fft = np.fft.fft(func1)
	f2_fft = np.fft.fft(func2)
	
	# Multiplying the fft of two functions
	m = f1_fft * f2_fft
	
	# Gives out the final convolved function
	y = abs(np.fft.ifft(m))

	return(y)

	
# Scatter broadening function
def fscat(x, data, nbins, tau): 
    scatfn = shift((exp(-(x)/tau)),nbins/2) # adds a scatter broadening tail
    scatfn /= max(scatfn) # normalisation
    #y = np.convolve(data,scatfn,'same') / sum(scatfn) # convolution , KK will send me code to check  this line with cicular convolution
    y = convolve_funcs(data, scatfn) / sum(scatfn)
    y /= max(y) # normalisation
    return y
	
# Simulate fits files from starting MJD with given cadence
def simulate(Case, tempo2path, nmjd, MJDST, par, nchan, template, hf, lf, telescope, dmsim, pulse, alphasim):
    MJD = MJDST
    for imjd in range(nmjd):
        MJD1 = MJD - 0.02 # how to decide this range?
        MJD2 = MJD + 0.02 # 0.02 + 0.02 > 1800 sec (segment duration also called the prediction span is time over which predictor will be used to make predictions)
        print(MJD1,MJD2)
        # Here we need to create a new parfile with DM for this MJD for getting 
        # predictors with correct phase. Read DM from simulated DM and install 
        # it in temporary parfile
        fp = open(par,"r")
        count = len(fp.readlines())
        fp.close()
        fpout = open("tempparfile","w+")
        with open(par) as fp:
            for iline in range(count):
                line = fp.readline()
                fields = line.strip().split()
                if fields[0] == "DM":
                    fields[1] = str(dmsim[imjd])
                line='   '.join(fields)
                fpout.write(line+'\n')
        fpout.close()
        parfile3 = "tempparfile"
       	
	#os.system('cat pred.tim') 
        # Generate a predictor for Band 3 and current MJD 8 time coefficients and 
        # 64 frequency coefficients for 1800 s time and 300-500 MHz band. See Hobbs et al 2006 paper.
	# but how are we choosing this values, expecially the segment duration
        cmd = "{}/bin/tempo2 -f {} -pred \"{} {} {} {} {} 64 8 1800.0\"".format(tempo2path, parfile3, telescope, MJD1, MJD2, hf, lf)
        print("[CMD] : {}".format(cmd))
        os.system(cmd)
	
	#os.system('cat pred.tim')
        g=open('pred.tim','r')
        tim=g.readlines()
        g.close()
        print(len(tim))
        # Then obtain phase at each frequency. (C program1)
        cmd = "./getphasePredictor band_freq {}".format(MJD) # path to program should be an input
        print("[CMD] : {}".format(cmd))
        os.system(cmd)
	
	#[All above is same for all three cases]
	
        if Case==1:
            pass
        else:
	####################################### this much below doesnt exist in case1 ###########
            # Also obtain the period of pulsar at the current MJD. (C program2)
            cmd = "./getperiodPredictor band_freq {}".format(MJD)
            print("[CMD] : {}".format(cmd))
            os.system(cmd)
	
            # Use it to scale x axis in sec for later use in convolution with PBF.
            xphase = np.zeros(nbin)
            period=np.loadtxt("period.out")
            for ixphase in range(nbin) :
                xphase[ixphase]=(period/nbin)*ixphase
	
            # Calculate tau_sc for each frequency
            freqval=np.loadtxt("band_freq")
            freqval=freqval[1:nchan+1]
            tau=np.zeros(nchan)
        
            for ichan in range(nchan):
                if Case==3:
                    logtau = np.log10(tau300mhz)+alphasim[imjd]*np.log10(freqval[ichan]) - alphasim[imjd]*np.log10(300.0)
                elif Case==2:
                    logtau = np.log10(tau300mhz)+alpha*np.log10(freqval[ichan]) - alpha*np.log10(300.0)
                tau[ichan] = 10**logtau
        ###################################################################################
	
       	# UPGRADE: Put in the formula for aberation and retardation (phase correction as a function of freq)
	
        # Read predictors and fix these in predictor binary table #[same for all three cases]
        f=open('t2pred.dat','r')
# first reads the contents of the file t2pred.dat into a list of strings called pred. 
        pred=f.readlines()
	#print('pred is following')
	#print(pred)
        f.close()
        pr=[l.strip('\n\r') for l in pred]  # With this command the t2predict.dat is converted into an array of strings
# It then processes this list to remove newline characters and create a new list called pr. (done just above)
	#print('pr is following')
	#print(pr)
        nlen=len(pr)
	#print('nlen is following')
	#print(nlen)
        
        newfits=template.rstrip('fits')
        nfits=newfits+'mod.fits' # from here nfits is the .mod.fits extension file
	#print(nfits)	
        # astropy is called here
        hdul=fits.open(nfits)
# The code retrieves the data and header for the predictor data stored in the file (hdul['T2PREDICT'].data and hdul['T2PREDICT'].header)
# and stores them in the variables data and hdr, respectively.

        data=hdul['T2PREDICT'].data # This is the hdul[3] which has the predictor model
        hdr=hdul['T2PREDICT'].header # This is the header for the hdul[3]

	#print('"data" is following')
	#print(data)
	#print('below is copy')

        hdr['NAXIS2']=nlen # NAXIS2 is the number of rows in table, this command equates that to number of elements in 'pr'
        dataq=np.copy(data)
	#print('resized copy is following')
        dataq.resize(nlen)

 	#print(dataq)	## THIS HAS PROBLEM
	#print(data)
	#print(len(data))
	#print('BREAK')
	#print(dataq)
	#print(len(dataq))
	#os.system('psrstat '+nfits)	
	# At this point, .mod.fits is working BOTH IN V6 as well as V6.2 templates
 	# The cheby model error starts from here in the .mod.fits file, baecause the format is different

        for ielem in range(nlen):
	    #print(ielem)
	    #print(dataq['PREDICT'][ielem])
	    #print(pr[ielem])
            dataq['PREDICT'][ielem]=pr[ielem] # In header 'PREDICT' is the text file that contains row by row, ilem is the ith element of that text list
        update(nfits, dataq, hdr, 'T2PREDICT') # Here we are updating the chebymodel of the .mod.fits file by the new chebymodel params in data of hdul[3]
	
	#phdr=hdul['PRIMARY'].header
        #pdata=hdul['PRIMARY'].data
        #phdr['EXTEND'] = 'T'
	#phdr['STT_IMJD']=int(MJD) # Start MJD (UTC days) (J - long integer)
        #phdr['STT_SMJD']=int((MJD-int(MJD))*86400.0) # [s] Start time (sec past UTC 00h) (J)
        #timefract = (MJD-int(MJD))*86400.0
        #phdr['STT_OFFS']=timefract - int(timefract) # [s] Start time offset (D)
	#print(phdr)
	#print(pdata)
	#update(nfits, pdata, phdr, 'PRIMARY')
	
	#hdul.close()
	#os.system('psrstat '+nfits)
	
        hdul=fits.open(nfits)
        phdr=hdul['PRIMARY'].header
        pdata=hdul['PRIMARY'].data
        phdr['STT_IMJD']=int(MJD) # Start MJD (UTC days) (J - long integer)
	#print(int(MJD))
        phdr['STT_SMJD']=int((MJD-int(MJD))*86400.0) # [s] Start time (sec past UTC 00h) (J)
	#print(int((MJD-int(MJD))*86400.0))
        timefract = (MJD-int(MJD))*86400.0
	#print(timefract)
        phdr['STT_OFFS']=timefract - int(timefract) # [s] Start time offset (D)
	#print(timefract - int(timefract))

        if Case==1:
            nnfits = str(PSR)+'_'+str(MJD)+'_'+str(hf)+'.noscat.fits'
        elif Case==2:
            nnfits = str(PSR)+'_'+str(MJD)+'_'+str(hf)+'.conscat.fits'
        elif Case==3:
            nnfits = str(PSR)+'_'+str(MJD)+'_'+str(hf)+'.varscat.fits'
	
        hdul.writeto(nnfits)
	#[same for all three cases] in the above line The PRIMARY header and data from nfits is supplied to nnfits
	#keep in mind that the T2PREDICT is already updated in nfits
	
	# Read phases for each channel and frequency labels in an array #[same for all three cases]
        d=np.loadtxt("phase.out")
        frequency=d[:,0]
        phase=d[:,1]
	
        # Then make a data  file with pulse offset at phase predicted by 
        # PREDICTORS across all channels by simply sifting and then 
        # copying simulated data
        # [for only case 2 and 3:
    	# Convolve pulse with exponential PBF with Kolmogorov -4.4 or simulated alpha (case3) index
    	# for each frequency]
	
        if Case==1:
            nfits = str(PSR)+'_'+str(MJD)+'_'+str(hf)+'.noscat.fits'
        elif Case==2:
            nfits = str(PSR)+'_'+str(MJD)+'_'+str(hf)+'.conscat.fits'
        elif Case==3:
            nfits = str(PSR)+'_'+str(MJD)+'_'+str(hf)+'.varscat.fits'
        
        ar = psrchive.Archive_load(nfits) #[same for all three cases]
        subint = ar.get_Integration(0) #[same for all three cases]
	
        for ichan in range(nchan):
#  loop that processes multiple channels (nchan) of data. Each iteration of the loop processes a single channel.
            if Case==1:
                pass
            else:
                scatpulse = fscat(xphase, pulse, nbin, tau[ichan]) # this line doesnt exist in case1
            prof = subint.get_Profile(0,ichan) #  retrieves a profile for the current channel and call it prof
# Below the code determines how much to shift the pulse using the equation:
            nshift=int(np.floor((phase[ichan]-0.5)*nbin)) # UPGRADE here the phase correction will be added for aberation and retardation
            if nshift < 0 :
                for ibin in range(nbin+nshift) :
                    if Case==1:
                        prof[ibin+abs(nshift)] = pulse[ibin]*1000.0
                    else:
                        prof[ibin+abs(nshift)] = scatpulse[ibin]*1000.0 # scatpulse become pulse in case1
                for ibin in range(abs(nshift)) :
                    if Case==1:
                        prof[ibin] = pulse[ibin+nbin+nshift]*1000.0
                    else:
                        prof[ibin] = scatpulse[ibin+nbin+nshift]*1000.0 # scatpulse become pulse in case1
            else :
                for ibin in range(nbin-nshift) :
                    if Case==1:
                        prof[ibin] = pulse[ibin+nshift]*1000.0
                    else:
                        prof[ibin] = scatpulse[ibin+nshift]*1000.0 # scatpulse become pulse in case1
                for ibin in range(nshift) :
                    if Case==1:
                        prof[ibin+nbin-nshift] = pulse[ibin]*1000.0
                    else:
                        prof[ibin+nbin-nshift] = scatpulse[ibin]*1000.0 # scatpulse become pulse in case1
        #psrchive.Archive.set_dispersion_measure(ar, dmsim[imjd])
        ar.unload(nfits)
        MJD = MJD + cadence

if args.noscat:
    print('[INFO] : Scatter-broadening will NOT be included in simulation')
    dmsim = dmsimulate(dmfile)
    container(PSR, nchan, nbin, DM, par,template)
    pulse = pulsemaker()
    alphasim = 0
    simulate(1, tempo2path, nmjd, MJDST, par, nchan, template, hf, lf, telescope, dmsim, pulse, alphasim)
elif args.conscat:
    print('[INFO] : Constant scatter-broadening will be included in simulation')
    dmsim = dmsimulate(dmfile)
    container(PSR, nchan, nbin, DM, par,template)
    pulse = pulsemaker()
    alphasim = 0
    simulate(2, tempo2path, nmjd, MJDST, par, nchan, template, hf, lf, telescope, dmsim, pulse, alphasim)
elif args.varscat:
    print ('[INFO] : Variable scatter-broadening will be included in simulation')
    dmsim = dmsimulate(dmfile)
    alphasim = alphasimulate(alphafile)
    container(PSR, nchan, nbin, DM, par, template)
    pulse = pulsemaker()
    simulate(3, tempo2path, nmjd, MJDST, par, nchan, template, hf, lf, telescope, dmsim, pulse, alphasim)
else:
    print('\033[91m [ERR] \033[00m : Please choose a valid scatter-broadening option. Use -h for help')
    exit()

#cmd = "rm band_freq t2pred.dat pred.tim *.mod.* tempparfile phase.out period.out"
#print("[CMD] : {}".format(cmd))
#os.system(cmd)


