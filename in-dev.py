# Code Author : Piyush M

# Importing Libraries
import psrchive
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(prog='in-dev', #name of the program file should be finalised
                            usage='''python in-dev.py [FITS] [OPTION]...\nPlease use -h command for help
                            ''',
                            description =  '''
----------------------------------------------------------------------
   This Python script is under-development

Notes: 1. The nsub given in input is rounded to nearest possible
          subints (intrinsic to PSRCHIVE),
       2. Number of frequency channels should be a multiple of 2
          For major uses, calibration of profiles are not necessary,
       3. ALWAYS DO scrunch polarisation channels to ONE, if NOT the case
       4. Keep number of phase bins maximum
       5. Bounds on color plot of secondary spectrum are hardcoded
          and can be optimised
       6. if DM assigned is zero or negative, then that command is
          ignored and default is used
       7. All spectra generation requires number of peaks in the
          integrated pulse profile (current version handles one or
          two gaussian peaks)

-Piyush
----------------------------------------------------------------------
                            '''
                                 , formatter_class=argparse.RawDescriptionHelpFormatter,
                            add_help=True
                            )
group1 = parser.add_argument_group('Available plotting options')
group2 = parser.add_argument_group('Flux Calibration options')
group3 = parser.add_argument_group('PSRFITS metadata change options (Does not write it to disk)')


#positional argument
parser.add_argument('fits',
                    help = " '.fits' file of the pulsar: to be present in working directory")

#optional arguments

group3.add_argument('-DM',
                           help = 'value of dispersion measure to be used : sets DM in parsec per cubic cm, default is from FITS header',
                           type = float)

group2.add_argument('-Tsys',
                           help = 'system temperature in kelvin : sets Tsys value for calibration',
                           type = float)

group2.add_argument('-Tsky',
                           help = 'sky temperature in kelvin : sets Tsky value for calibration',
                           type = float)

group2.add_argument('-mode',
                           help = 'mode of observation : choose "PA" for phased-array and "IA" for incoherent-array',
                           type = str)

group2.add_argument('-Nant',
                           help = 'number of antennae used for observation',
                           type = int)

group3.add_argument('-npeak',
                           help = 'number of peaks in integrated profile',
                           type = int)

group3.add_argument('-Nbin',
                           help = 'number of phase-bins per profile : scrunches archive to have NBIN bins',
                           type = int)

group3.add_argument('-Nsub',
                           help = 'number of subintegrations : scrunches archive to have NSUB subintegrations',
                           type = int)

group3.add_argument('-Nchan',
                           help = 'number of frequency channnels  : scrunches archive to have NCHAN frequency channels',
                           type = int)

group3.add_argument('-Npol',
                           help = 'number of polarization states  : scrunches archive to have NPOL polarization states',
                           type = int)

group1.add_argument('-intpf',
                           help = 'show an integrated profile plot : flux density count as a function of phase bins, use it for bin indices of on-pulse windows and number of peaks',
                           action="store_true")

group1.add_argument('-dspec',
                           help = 'show the dynamic spectrum plot : flux density as a function of observation time and frequency',
			   action="store_true")

group1.add_argument('-sspec',
                           help = 'show the secondary spectrum plot : power spectrum of the dynamic spectrum',
			   action="store_true")

group1.add_argument('-acf',
                           help = 'show the autocorrelation spectrum plot : autocorrelation transform of the dynamic spectrum',
                           action="store_true")

group1.add_argument('-fspec',
                           help = 'show the frequency spectrum plot : flux density as a function of frequency channels (index of Nchan)',
                           action="store_true")

group3.add_argument('-onw',
                           help = 'onpulse-width : take 1/(ONW) th of maximum as onpulse width. ONW is +ve int, default is 10',
                           type = int, default = 10)

group1.add_argument('-tau',
                            help='compute scattering tau spectrum',
                            action="store_true")

args = parser.parse_args()

# FITS file as input
fits = args.fits
print(fits)

# Loading the archives
a = psrchive.Archive_load(fits)
b = psrchive.Archive_load(fits)

# Onpulse width parameter
N = args.onw if args.onw and args.onw > 0 else 10

# DM assignment
if args.DM is not None:
    if args.DM > 0.0:
        psrchive.Archive.set_dispersion_measure(a, args.DM)
    else:
        print("Please give valid DM value")

# Dedisperse
a.dedisperse()

# Optional scrunching BEFORE extracting numpy array
if args.Nsub:
    a.tscrunch_to_nsub(args.Nsub)

if args.Nchan:
    a.fscrunch_to_nchan(args.Nchan)

if args.Nbin:
    a.bscrunch_to_nbin(args.Nbin)

if args.Npol:
    a.pscrunch_to_npol(args.Npol)

# Extract numpy cube ONCE
data = a.get_data()

# Shape reporting
shape = data.shape
axes = ["Nsub (time)", "Npol", "Nchan (freq)", "Nbin (phase)"]

print("\nData shape:")
for i, (s, name) in enumerate(zip(shape, axes)):
    print(f" Axis {i}: {name} = {s}")

# Collapse polarization if already single state
if data.shape[1] == 1:
    data = data[:, 0, :, :]
    print("\nPolarization collapsed -> 3D cube:", data.shape)

# Derived dimensions (consistent downstream usage)
nsub = data.shape[0]
nchan = data.shape[1]
nbin = data.shape[2]

# Observation metadata
tobs = a.integration_length()
print("Observation time: {} minutes or {} seconds".format(tobs/60.0, tobs))

freq_lo = a.get_centre_frequency() - np.abs(a.get_bandwidth())/2.0
freq_hi = a.get_centre_frequency() + np.abs(a.get_bandwidth())/2.0

print("DM value used: {} pc cm^-3".format(a.get_dispersion_measure()))
print("Bandwidth: {} MHz ({} - {} MHz)".format(np.abs(a.get_bandwidth()), freq_lo, freq_hi))
print("Centre frequency: {} MHz".format(a.get_centre_frequency()))

tscope = str(a.get_telescope())
print("Telescope used:", tscope)

# Functions:
from scipy.optimize import curve_fit
def exp_decay(x, a, x0):
    return a * np.exp(-np.abs(x)/x0)
def gaussian(x, a, x0):
    return a * np.exp(-(x**2)/(2*x0**2))
def lorentzian(x, a, x0):
    return a / (1 + (x/x0)**2)

def dynamic_spectrum(self):
    """
    Plots the Dynamic spectrum
    """
    # Orientation correction
    self = np.rot90(self.reshape(a.get_nsubint(), a.get_nchan()))
    
    # Plotting
    plt.xlabel('Time (in min) with nsub=' + str(nsub))
    plt.ylabel('Frequency (in MHz) with nchan=' + str(nchan))
    plt.title(fits + "\nDynamic spectrum")
    plt.set_cmap('plasma')
    plt.imshow(self, extent=(0,tobs/60.0,freq_lo,freq_hi), aspect = 'auto')
    cbar2 = plt.colorbar()
    plt.savefig(fits+"_Dynamic_spectrum.png")
    #plt.show()

#def autocorrelation_spectrum(self):
def autocorrelation_spectrum(self, return_data=False):  
    """
    Plots the Autocorrelation  spectrum
    """
    # Orientation correction
    self = np.rot90(self.reshape(a.get_nsubint(), a.get_nchan()))

    # Definition of autocorrelation spectrum
	self = self - np.mean(self)
	fft = np.fft.fft2(self)
	acf = np.fft.fftshift(np.fft.ifft(fft * np.conjugate(fft))).real
	acf = acf / np.max(acf)
	acf_data = acf
	plot_acf = np.log10(acf_data + 1e-12)
	
    # Axes
    #time_lag = np.fft.fftshift(np.fft.fftfreq(nsub, d=tobs/nsub))
    #freq_lag = np.fft.fftshift(np.fft.fftfreq(nchan, d=np.abs(a.get_bandwidth())/nchan))

	dt = tobs / nsub
	#time_lag = np.arange(-nsub//2, nsub//2) * dt
	df = np.abs(a.get_bandwidth()) / nchan
	#freq_lag = np.arange(-nchan//2, nchan//2) * df

	time_lag = (np.arange(nsub) - nsub//2) * dt
	freq_lag = (np.arange(nchan) - nchan//2) * df
  
    # Plotting
    #plt.xlabel('Time (in min) with nsub=' + str(nsub))
    #plt.ylabel('Frequency (in MHz) with nchan=' + str(nchan))
	plt.figure()
	plt.xlabel('Time lag (s)')
	plt.ylabel('Frequency lag (MHz)')
    plt.title(fits + "\nAutocorrelation spectrum")
    plt.set_cmap('plasma')
	#self_plot = 10*np.log10(np.abs(np.fft.fftshift(np.fft.ifft(np.fft.fft2(self)*np.conjugate((np.fft.fft2(self)))))))
    #plt.imshow(self, extent=(0,tobs/60.0,freq_lo,freq_hi), aspect = 'auto')
	plt.imshow(plot_acf, extent=(min(time_lag), max(time_lag), min(freq_lag), max(freq_lag)), aspect='auto', origin='lower')
    cbar2 = plt.colorbar()
    plt.savefig(fits+"_Autocorrelation_spectrum.png")
    #plt.show()

    if return_data:
      return acf_data, time_lag, freq_lag

def scint_params(acf, time_lag, freq_lag):

    # central cuts
    #t_cut = acf[:, nchan//2]
    #f_cut = acf[nsub//2, :]

	mid_t = acf.shape[0] // 2
	mid_f = acf.shape[1] // 2

	t_cut = acf[:, mid_f]   # time cut at zero freq lag
	f_cut = acf[mid_t, :]   # freq cut at zero time lag
	
	t_cut = t_cut[mid_t:]
	time_lag = time_lag[mid_t:]

	f_cut = f_cut[mid_f:]
	freq_lag = freq_lag[mid_f:]

    # normalise
    #t_cut = t_cut / np.max(t_cut)
    #f_cut = f_cut / np.max(f_cut)

	t_cut = t_cut / t_cut[0]
	f_cut = f_cut / f_cut[0]

    # fit
    try:
        popt_t, _ = curve_fit(exp_decay, time_lag, t_cut, p0=[1, 100])
        tau_d = np.abs(popt_t[1])
    except:
        tau_d = np.nan

    try:
        #popt_f, _ = curve_fit(gaussian, freq_lag, f_cut, p0=[1, 0.1])
		popt_f, _ = curve_fit(lorentzian, freq_lag, f_cut, p0=[1, np.max(freq_lag)/10])
        delta_nu_d = np.abs(popt_f[1])
    except:
        delta_nu_d = np.nan

    print("Scintillation timescale (s):", tau_d)
    print("Scintillation bandwidth (MHz):", delta_nu_d)

    return tau_d, delta_nu_d

def secondary_spectrum(self):
    """
    Calculates and plots the secondary spectrum
    """
    # Orientation correction
    self = np.rot90(self.reshape(a.get_nsubint(), a.get_nchan()))
    print ('Orientation is corrected...')

    # Defining conjugate frequency and time, be careful for right units
    conjT = np.fft.fftshift(np.fft.fftfreq(nsub, d= tobs/nsub ))  ### NEED PROOF-READING ###
    conjF = np.fft.fftshift(np.fft.fftfreq(nchan, d= np.abs(a.get_bandwidth())/nchan)) ### NEED PROOF-READING ###
    print ('Computing conjugate time and frequency is done...')

    # Definition of secondary spectrum
    scd = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft2(self - np.mean(self))))**2)
    print ('Power Spectrum calculation is done...')

    plt.xlabel('conjugate time\n(fringe frequecy in Hz)', fontsize = 8)
    plt.ylabel('conjugate frequency\n(time delay in micro-seconds)', fontsize = 8)
    plt.title(fits + "\nSecondary spectrum with nsub=" + str(nsub) + ' and nchan=' + str(nchan), fontsize = 12)
    plt.set_cmap('jet')
    
    # Dynamic Range
    #low  = np.min(scd)
    #low  = 30
    low  = np.median(scd)
    high = np.max(scd)
    
    #scd[:,nsub/2] = 0
	scd[:, nsub//2] = 0

    # Plotting
    plt.imshow(scd[:int(scd.shape[0]/2), : int(scd.shape[1])], vmin = low, vmax = high, extent=(min(conjT), max(conjT), 0, max(conjF)), aspect = 'auto')
    cbar1 = plt.colorbar()
    plt.savefig(fits+"_Secondary_spectrum.png")
    np.save(fits+'_secondary_spectrum.npy', scd[:int(scd.shape[0]/2), :int(scd.shape[1])] )
    #plt.show()

def noise_cal_single_peak(left_edge,right_edge):
    """
    Noise calibration of single peak pulsar data
    """
    a.remove_baseline()
    a.centre_max_bin()
    self = []
    w0 = (nbin/2) - left_edge
    w1 = (nbin/2) + right_edge

    xon    = np.linspace(w0, w1,     num = w1-w0+1, dtype=int)
    xtot   = np.linspace(0,  nbin-1, num = nbin,    dtype=int)
    xoff1  = np.linspace(0,  w0,     num = w0+1,    dtype=int)
    xoff2  = np.linspace(w1, nbin-1, num = nbin-w1, dtype=int)

    for isub in range(a.get_nsubint()):
        i = a[isub]
        for ichan in reversed(range(a.get_nchan())):
            j = ichan

            ytot     = i.get_Profile(0,j).get_amps()[xtot]
            yoff1    = i.get_Profile(0,j).get_amps()[xoff1]
            yoff2    = i.get_Profile(0,j).get_amps()[xoff2]

            mean1    = np.nanmean(yoff1)
            std1     = np.nanstd(yoff1)
            sq_rms1  = std1**2 + mean1**2

            mean2    = np.nanmean(yoff2)
            std2     = np.nanstd(yoff2)
            sq_rms2  = std2**2 + mean2**2

            sq_rms   = ((w0+1)*sq_rms1 + (nbin-w1)*sq_rms2)/(w0+1+nbin-w1)
            mean     = ((w0+1)*mean1 + (nbin-w1)*mean2)/(w0+1+nbin-w1)
            std      = np.sqrt(sq_rms - mean**2)

            amp = i.get_Profile(0,j).get_amps()[xon]

            if (std != 0):
                if rms == 0:
                    cal_amp = (amp - mean)/std
                else:
                    cal_amp = (amp - mean)*(rms/std)

                flux = np.trapz(cal_amp,xon)/(w1-w0+1)
            else:
                flux = 0
            if (flux < 0):
                flux = 0

            self = np.append(self , flux)
    print ('Noise calibration is done...')
    return self

def noise_cal_two_peak(left_main,right_main,left_inter,right_inter):
    """
    Noise calibration if profile has interpulse also 
    """
    self = []
    for isub in range(a.get_nsubint()):
        i = a[isub]
        for j in reversed(range(nchan)): 
    
            xtot   = np.linspace( 0,  nbin-1,    num = nbin,       dtype=int)
            ytot   = i.get_Profile(0,j).get_amps()[xtot]
            
            b0 = i.get_Profile(0,j).find_max_bin()
            if (b0>nbin/4):
                ytot = np.roll(ytot,nbin/4-b0)
            if (b0<nbin/4):
                ytot = np.roll(ytot,nbin/4)

            w0 = left_main
            w1 = right_main
            w2 = left_inter
            w3 = right_inter

            xoff1  = np.linspace( 0,      w0,    num = w0+1,       dtype=int)
            xon1   = np.linspace( w0,     w1,    num = w1-w0+1,    dtype=int)
            xoff2  = np.linspace( w1,     w2,    num = w2-w1+1,    dtype=int)
            xon2   = np.linspace( w2,     w3,    num = w3-w2+1,    dtype=int)
            xoff3  = np.linspace( w3, nbin-1,    num = nbin-w3,    dtype=int)

            # Amplitudes of different regions of profiles
            yoff1 = ytot[xoff1]
            yon1  = ytot[xon1]
            yoff2 = ytot[xoff2]
            yon2  = ytot[xon2]
            yoff3 = ytot[xoff3]
                    
            # off-pulse 1
            mean1   = np.nanmean(yoff1)
            std1    = np.nanstd(yoff1)
            sq_rms1 = std1**2 + mean1**2

            # off-pulse 2
            mean2   = np.nanmean(yoff2)
            std2    = np.nanstd(yoff2)
            sq_rms2 = std2**2 + mean2**2

            # off-pulse 3
            mean3    = np.nanmean(yoff3)
            std3     = np.nanstd(yoff3)
            sq_rms3  = std3**2 + mean3**2

            # combined standard deviation and mean of all off-pulse regions
            sq_rms   = ((w0+1)*sq_rms1 + (w2-w1+1)*sq_rms2 + (nbin-w3)*sq_rms3)/(w0+nbin-w3+w2-w1+2)
            mean     = ((w0+1)*mean1 + (w2-w1+1)*mean2 + (nbin-w3)*mean3)/(w0+nbin-w3+w2-w1+2)
            std      = np.sqrt(sq_rms - mean**2)

            amp1 = i.get_Profile(0,j).get_amps()[xon1]
            amp2 = i.get_Profile(0,j).get_amps()[xon2]

            if (std != 0):
                
                if rms == 0:
                    cal_amp1 = (amp1 - mean)/std
                    cal_amp2 = (amp2 - mean)/std
                else:
                    cal_amp1 = (amp1 - mean)*(rms/std)
                    cal_amp2 = (amp2 - mean)*(rms/std)
                
                flux = (np.trapz(cal_amp1,xon1) + np.trapz(cal_amp2,xon2))/(w1-w0+1+w3-w2+1)
            else:
                flux = 0
            if (flux < 0):
                flux = 0
            self = np.append(self , flux)
    print ('Noise calibration is done...')
    return self

def integrated_profile():
    """
    Gives On-pulse window for single peak pulsars
    """
    b.tscrunch()
    b.fscrunch()
    b.remove_baseline()
    b.centre_max_bin()
    prof  = b[0].get_Profile(0,0).get_amps()
    prof  = prof - min(prof) # Brings the profile minimum to zero
    c = np.where(prof < max(prof)/N)
    prof[c] = 0 # Any count that is less than 1/Nth of the max
    d = []
    d = np.where(prof > max(prof)/N)
    f = [ (nbin/2) - (d[0][0] - 1) , (d[0][-1] + 1) - nbin/2 ]
    return f

def interpulse_profile():
    """
    Gives On-pulse window for double peak pulsars
    """
    b.tscrunch()
    b.fscrunch()
    b.remove_baseline()
    prof   = b[0].get_Profile(0,0).get_amps()
    b0 = b[0].get_Profile(0,0).find_max_bin()
    
    # Phase Aligning
    if (b0>nbin/4):
        prof = np.roll(prof,nbin/4-b0)
    if (b0<nbin/4):
        prof = np.roll(prof,nbin/4)
    
    prof  = prof - min(prof) # Brings the profile minimum to zero
    c = np.where(prof < max(prof)/N)
    prof[c] = 0 # Any count that is less than 1/Nth of the max
    l =[]
    for i in range(len(prof)):
        if (0 < i < len(prof)-1):
            #print i
            if (prof[i-1] > 0) and (prof[i+1] == 0):
                l = np.append(l , int(i))
            elif (prof[i-1] == 0) and (prof[i+1] > 0):
                l = np.append(l , int(i))
    return l

def scatter_tau_spectrum():

    a.tscrunch()

    taus = []
    freqs = np.linspace(freq_lo, freq_hi, nchan)

    for j in range(nchan):
        prof = a[0].get_Profile(0, j).get_amps()

        prof = prof - np.min(prof)
        peak = np.argmax(prof)

        x = np.arange(peak, nbin)
		dt = tobs / (nsub * nbin)
		x = (np.arange(peak, nbin) - peak) * dt
        y = prof[peak:]
		y = prof[peak:]
		y = y / np.max(y)

		mask = y > 0.05
		x = x[mask]
		y = y[mask]

        if len(y) < 5:
            taus.append(np.nan)
            continue

        try:
            popt, _ = curve_fit(exp_decay, x, y, p0=[np.max(y), 10])
            taus.append(np.abs(popt[1]))
        except:
            taus.append(np.nan)

    taus = np.array(taus)

    # fit power-law tau ~ nu^-alpha
    good = ~np.isnan(taus)
    if np.sum(good) > 5:
        coeff = np.polyfit(np.log(freqs[good]), np.log(taus[good]), 1)
        alpha = -coeff[0]
        print("Scattering index alpha ~", alpha)

    np.save(fits+"_tau_spectrum.npy", taus)

    plt.figure()
    plt.plot(freqs, taus)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Tau (bins)")
    plt.title("Scattering timescale vs frequency")
    plt.savefig(fits+"_tau_spectrum.png")

def dspec():
    
    print ('Computing Dynamic Spectrum... Please wait')

    if args.npeak == 1:
        print ("Pulsar has single-pulse profile")
        x  = integrated_profile()
        print ('ON-pulse is calculated...')
        #print x
        dynamic_spectrum(noise_cal_single_peak(x[0],x[1]))
        
    elif args.npeak == 2:
        print ("Pulsar has double-pulse profile")
        print ('ON-pulse is calculated...')
        x = interpulse_profile()
        #print x
        dynamic_spectrum(noise_cal_two_peak(x[0] , x[3] , x[-4] , x[-1]))
    
    else:
        print ("Error: please specify valid number of pulses")

def acf():

    print ('Computing Autocorrelation Spectrum... Please wait')

    if args.npeak == 1:
        print ("Pulsar has single-pulse profile")
        x  = integrated_profile()
        print ('ON-pulse is calculated...')
        #print x
        #autocorrelation_spectrum(noise_cal_single_peak(x[0],x[1]))
        acf_data, tlag, flag = autocorrelation_spectrum(noise_cal_single_peak(x[0],x[1]), return_data=True)
        scint_params(acf_data, tlag, flag)

    elif args.npeak == 2:
        print ("Pulsar has double-pulse profile")
        x = interpulse_profile()
        print ('ON-pulse is calculated...')
        #print x
        #autocorrelation_spectrum(noise_cal_two_peak(x[0] , x[3] , x[-4] , x[-1]))
        acf_data, tlag, flag = autocorrelation_spectrum(noise_cal_two_peak(x[0] , x[3] , x[-4] , x[-1]), return_data=True)
        scint_params(acf_data, tlag, flag)

    else:
        print ("Error: please specify valid number of pulses")

def sspec():
    
    print ('Computing Secondary Spectrum... Please wait')

    if args.npeak == 1:
        print ("Pulsar has single-pulse profile")
        x  = integrated_profile()
        print ('ON-pulse is calculated...')
        #print x
        secondary_spectrum(noise_cal_single_peak(x[0],x[1]))
    
    elif args.npeak == 2:
        print ("Pulsar has double-pulse profile")
        x = interpulse_profile()
        print ('ON-pulse is calculated...')
        #print x
        secondary_spectrum(noise_cal_two_peak(x[0] , x[3] , x[-4] , x[-1]))
    
    else:
        print ("Error: please specify valid number of pulses")

def fspec():
    """
    plots calibrated flux vs frequency channels
    """
    a.tscrunch()
    freq = np.linspace(0,nchan-1,num = nchan,dtype=int)
    self = []

    if args.npeak == 1:
        print ("Pulsar has single-pulse profile")
        x  = integrated_profile()
        print ('ON-pulse is calculated...')
        self = noise_cal_single_peak(x[0],x[1])
        # Flux density estimation:
        Sobs = np.trapz(self,freq)/nchan
        print ('Sobs (in mJy) ='),Sobs

    elif args.npeak == 2:
        print ("Pulsar has double-pulse profile")
        x  = interpulse_profile()
        print ('ON-pulse is calculated...')
        self = (noise_cal_two_peak(x[0] , x[3] , x[-4] , x[-1]))
        # Flux density estimation:
        Sobs = np.trapz(self,freq)/nchan
        print ('Sobs (in mJy) ='),Sobs
    
    else:
        print ("Error: please specify valid number of pulses")
    
    # Plotting
    plt.title(fits+' Flux vs Frequency channel plot')
    plt.xlabel('Frequency channels; nchan= ' + str(nchan))
    plt.ylabel('Flux (in mJy)')
    plt.plot(self)
    plt.savefig(fits+"_fspec.png")
    #plt.show()

def intpf():
    """
    plots integrated profile
    """
    b.tscrunch()
    b.fscrunch()
    b.bscrunch_to_nbin(a.get_data().shape[3]) 
    nbin = b.get_data().shape[3]
    prof = b[0].get_Profile(0,0).get_amps()
    b0 = b[0].get_Profile(0,0).find_max_bin()
    
    # Phase Allignment:
    if (b0>nbin/4):
        prof = np.roll(prof, nbin/4-b0)
    if (b0<nbin/4):
        prof = np.roll(prof, nbin/4)

    prof  = prof - min(prof) #Brings the profile minimum to zero 
    
    # Plotting:
    plt.title(fits +'integrated Pulse profile')
    plt.axhline(max(prof), color = "yellow", label = 'maximum')
    plt.axhline(max(prof)/N, color = "green", label = '1/'+str(N)+'th of max')
    plt.axhline(max(prof)/10.0, color = "red", label = 'Tenth of max')
    plt.axhline(max(prof)/2.0, color = "black", label = 'Half of max')
    plt.legend(loc = 'best')
    plt.plot(prof)
    plt.xlabel('Phase Bins')
    plt.ylabel('Flux in arbitrary units')
    plt.savefig(fits+"_integrated_profile.png")
    #plt.show()

# Actions of arguments

if args.dspec:    
    dspec()

elif args.sspec:
    sspec()

elif args.acf:
    acf()
elif args.fspec:
    fspec()

elif args.intpf:
    intpf()

elif args.tau:
    scatter_tau_spectrum()

psrchive.Archive_load(fits).unload(fits)
psrchive.Archive_load(fits).unload(fits)
print ('GoodBye Pulsar!')
########################################## END ######################################################################################################################################
