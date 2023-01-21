# Code Author : Piyush M

# Importing Libraries
import psrchive
import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy import save

parser = argparse.ArgumentParser(prog='pulspec', #name of the program file should be finalised
                            usage='''python pulspec.py [FITS] [OPTION]...\nPlease use -h command for help
                            ''',
                            description =  '''
----------------------------------------------------------------------
   This Python scrtpt is under-development and aspires to be a
   pulsar scintillometry data analysis pipeline for GMRT and ORT data
   under the guidance of Prof. Bhal Chandra Joshi.
   For any comments, suggestions, and reporting bugs, please contact:
   Piyush Marmat (pmarmat@ph.iitr.ac.in)

Notes: 1. The nsub given in input is rounded to nearest possible
          subints (intrinsic to PSRCHIVE),
       2. Nnumber of frequency channels should be a multiple of 2
          For major uses, calibration of profiles are not neccessary,
       3. ALWAYS DO scrunch polarisation channels to ONE, if NOT the case
       4. Keep number of phase bins maximum
       5. Bounds on color plot of secondary spectrum are hardcoded
          and can be optimised
       6. if DM assigned is zero or negative, then that command is
          ignored and default is used
       7. linearisation of secondary spectrum is subject to further
          development and will be implemented in another script
       8. Thus all methods for extracting curvature parameter(s)
          will be in added in another script
       9. All spectra generation requires number of peaks in the
          intergrated pulse profile (current version handles one or
          two gaussian peaks)

-Piyush (1 Feb 2022)
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
                           type = int)

args = parser.parse_args()

# FITS file as input
fits = args.fits
print (fits)

# Loading the archives
a = psrchive.Archive_load(fits) # This is used to generate spectra
b = psrchive.Archive_load(fits) # This is used to get onpulse windows and integrated profile

# Setting the given onpulse-width or keep default
if (args.onw > 0):
    N = args.onw
else:
    N = 10

# Assigning the DM value
if args.DM:
    if (args.DM > 0.0):
        psrchive.Archive.set_dispersion_measure(a, args.DM)
    else:
        print ("Please give valid DM value")

# De-dispersing the archive
a.dedisperse()

# Data-shape
print ('Default data-shape (Nsub, Npol, Nchan, Nbin):') , (a.get_data().shape)

# Observation time
tobs = a.integration_length()
print ("Observation time: {} minutes or {} seconds".format(tobs/60.0, tobs))

# High and low edges of Bandwidth
freq_lo = a.get_centre_frequency() - np.abs(a.get_bandwidth())/2.0
freq_hi = a.get_centre_frequency() + np.abs(a.get_bandwidth())/2.0

# DM value used
print ("DM value used: {} parsec per cubic cm".format(a.get_dispersion_measure()))

# Bandwidth
print ("Bandwidth: {} MHz; ({} MHz - {} MHz)".format(np.abs(a.get_bandwidth()), freq_lo, freq_hi))
print ('Centre frequency: {} MHz'.format(a.get_centre_frequency()))

# Scrunching in time
if args.Nsub:
    a.tscrunch_to_nsub(args.Nsub)
    print ("Nsub after scrunching = {}".format(a.get_data().shape[0]))
    
nsub  = a.get_data().shape[0]

# Scrunching in frequency
if args.Nchan:
    a.fscrunch_to_nchan(args.Nchan)
    print ("Nchan after scrunching = {}".format(a.get_data().shape[2]))

nchan = a.get_data().shape[2]

# Scrunching in bins
if args.Nbin:
    a.bscrunch_to_nbin(args.Nbin)
    print ("Nbin after scrunching = {}".format(a.get_data().shape[3]))

nbin  = a.get_data().shape[3]

# Scrunching in polarization
if args.Npol:
    a.pscrunch_to_npol(args.Npol)
    print ("Npol after scrunching = {}".format(a.get_data().shape[1]))
    
npol  = a.get_data().shape[1]

# Telescope name
tscope = str(a.get_telescope())
print ("Telescope used: {}".format(tscope))

#Calculation of RMS noise fluctuations for calibration
rms = 0
    
if (tscope == str('or')) or (tscope == str('ort')) or (tscope == str('OR')) or (tscope == str('ORT')):
    print ('(Ooty Radio Telecope)')
    # Gain is 3 K/Jy, Tsys = 150 K
    rms = 50/np.sqrt((np.abs(a.get_bandwidth())/nchan)*(tobs/nbin)) # in mJy; This is radiometer equation
    
elif (tscope == str('GMRT')) or (tscope == str('gmrt')):
    print ('(Giant Meterwave Radio Telescope)')
    # Gain is 0.38 K/Jy
    if ((args.mode) and (args.Nant)):
        mode = args.mode
        print ('mode of observation is read...')
        Nant = args.Nant
        print ('number of antennae is read...')
        
        if mode == 'PA':
            # PA mode radiometer equation
            print ('mode of observation is Phased-Array')
            if (args.Tsys):
                Tsys = args.Tsys
                print ('using Tsys...')
                rms = ((Tsys)/0.38)*(1/np.sqrt((Nant**2)*2*(np.abs(a.get_bandwidth())/nchan)*(tobs/nbin))) # in mJy; This is radiometer equation
                
            elif (args.Tsky):
                Tsky = args.Tsky
                print ('using Tsky')
                rms = ((Tsky+66)/0.38)*(1/np.sqrt((Nant**2)*2*(np.abs(a.get_bandwidth())/nchan)*(tobs/nbin))) # in mJy; This is radiometer equation

        elif mode == 'IA':
            # IA mode radiometer equation
            print ('mode of observation is Incoherent-Array')
            if (args.Tsys):
                Tsys = args.Tsys
                print ('using Tsys...')
                rms = ((Tsys)/0.38)*(1/np.sqrt(Nant*2*(np.abs(a.get_bandwidth())/nchan)*(tobs/nbin))) # in mJy; This is radiometer equation
                
            elif (args.Tsky):
                Tsky = args.Tsky
                print ('using Tsky')
                rms = ((Tsky+66)/0.38)*(1/np.sqrt(Nant*2*(np.abs(a.get_bandwidth())/nchan)*(tobs/nbin))) # in mJy; This is radiometer equation 
        else:
            print ("Error : please give valid mode of observation")
        
    else:
        print ("Not enough calibration parameters, continuing without calibration")
            
else:
    print ("Computing without calibration")

if rms == 0:
    print ("RMS flux variation is not computed")
else:
    print ("RMS flux variation is {} mJy".format(rms))

# Functions:

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

def autocorrelation_spectrum(self):
    """
    Plots the Autocorrelation  spectrum
    """
    # Orientation correction
    self = np.rot90(self.reshape(a.get_nsubint(), a.get_nchan()))

    # Definition of autocorrelation spectrum
    self = 10*np.log10(np.abs(np.fft.fftshift(np.fft.ifft(np.fft.fft2(self)*np.conjugate((np.fft.fft2(self)))))))

    # Plotting
    plt.xlabel('Time (in min) with nsub=' + str(nsub))
    plt.ylabel('Frequency (in MHz) with nchan=' + str(nchan))
    plt.title(fits + "\nAutocorrelation spectrum")
    plt.set_cmap('plasma')
    plt.imshow(self, extent=(0,tobs/60.0,freq_lo,freq_hi), aspect = 'auto')
    cbar2 = plt.colorbar()
    plt.savefig(fits+"_Autocorrelation_spectrum.png")
    #plt.show()

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
    
    scd[:,nsub/2] = 0

    # Plotting
    plt.imshow(scd[:int(scd.shape[0]/2), : int(scd.shape[1])], vmin = low, vmax = high, extent=(min(conjT), max(conjT), 0, max(conjF)), aspect = 'auto')
    cbar1 = plt.colorbar()
    plt.savefig(fits+"_Secondary_spectrum.png")
    save(fits+'_secondary_spectrum.npy', scd[:int(scd.shape[0]/2), :int(scd.shape[1])] )
    #plt.show()

def noise_cal_single_peak(left_edge,right_edge):
    """
    Noise calibration of single peak pulsar data
    """
    a.remove_baseline
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
    b.remove_baseline
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
    b.remove_baseline
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
        autocorrelation_spectrum(noise_cal_single_peak(x[0],x[1]))

    elif args.npeak == 2:
        print ("Pulsar has double-pulse profile")
        x = interpulse_profile()
        print ('ON-pulse is calculated...')
        #print x
        autocorrelation_spectrum(noise_cal_two_peak(x[0] , x[3] , x[-4] , x[-1]))

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


psrchive.Archive_load(fits).unload(fits)
psrchive.Archive_load(fits).unload(fits)
print ('GoodBye Pulsar!')
########################################## END ######################################################################################################################################
