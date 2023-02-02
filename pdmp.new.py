import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# Help message and CLI
parser = argparse.ArgumentParser(prog='pdmp.py',
                            usage='''\033[96m python pdmp.py [PATH]... \033[00m\nPlease use -h command for help
                            ''',
                            description =  ''' 
-----------------------------------------------------------------------------------------------
\033[93mPDMP DM time-series, plots with error bars \033[00m

This simple program run the PDMP program of the PSRCHIVE package (Hotan et al. 2004)
on a directory containing PSRFITS data archives to produce the DM time series and a plot
of pulsar DM in pc/cm^3 vs MJD with error bars.

Dependencies: PSRCHIVE, numPy, MatplotLib and all standard Python libraries.
NOTE: before running this script, the shell environment must be allowing pdmp to run.
\033[91mPiyush 3-7-2022 \033[00m
-----------------------------------------------------------------------------------------------
                            '''
                            ,formatter_class=argparse.RawDescriptionHelpFormatter,
                            add_help=True)
#Define all command line arguments
parser.add_argument('fitsdir',
                            help = 'Path to the directory containing the PSRFITS files',
                            type = str)
args = parser.parse_args()

# Initialise inputs from command line
if args.fitsdir:
    fitsdir = args.fitsdir
else:
    print("\033[91m [ERR] \033[00m : Please give valid path to the directory having PSRFITS files.")
    exit()
print('[INFO] : Path to PSRFITS files is {}'.format(fitsdir))
# if this path doesnt exist then exit

# Making a list of the PSRFITS files in the given directory
cmd = 'ls {}/*.fits >> fitslist'.format(fitsdir)
print("[CMD] : {}".format(cmd))
os.system(cmd)

if EnvironmentError:
    os.system('rm fitslist')
    exit()

# Loads the fitlist as numpy array of minimiu dimension being 1 so that the code runs for a single file also
filelist = np.loadtxt('fitslist', dtype=str, ndmin=1)

# Running PDMP on each PSRFITS file and placing output in pdmp.log
for i in range(len(filelist)):
    cmd = 'pdmp '+filelist[i]+' >> pdmp.log'
    os.system(cmd)

# Taking "Best" DM out from pdmp.log for each file and making a "best" DM timeseries as pdmp.dm
cmd = 'grep "Best DM" pdmp.log > pdmp.bdm'
print("[CMD] : {}".format(cmd))
os.system(cmd)

# Taking MJDs from the first column of pdmp.per and putting it in mjd.dm file
cmd = "awk '{print $1}' pdmp.per >> mjd.dm"
print("[CMD] : {}".format(cmd))
os.system(cmd)

# Taking DM and errors from the fourth and tenth columns of pdmp.bdm and putting it in pdmp.dm
cmd = "awk '{print $4, $10}' pdmp.bdm > pdmp.dm"
print("[CMD] : {}".format(cmd))
os.system(cmd)

# Taking DM and errors from the pdmp.dm and add them as new columns in plot.dm, pdmp_timeseries.dm contains MJD|DM|error
cmd = """awk '{getline to_add < "pdmp.dm"; print $0,to_add}' mjd.dm >> pdmp_timeseries.dm"""
print("[CMD] : {}".format(cmd))
os.system(cmd)

# Removing useless files
cmd = "rm fitslist pdmp.log mjd.dm pdmp.dm pdmp.bdm pdmp.per pdmp.posn"
print("[CMD] : {}".format(cmd))
os.system(cmd)

# Plotting the DM time series with error bars
mjd,dm,err = np.loadtxt("pdmp_timeseries.dm", comments='#', usecols=(0,1,2), unpack=True)
plt.errorbar(mjd, dm, yerr=err, fmt="o", color = 'hotpink')
plt.title('PDMP DM time series')
plt.xlabel('MJD')
plt.ylabel(r"DM ($pc\,cm^{-3}$)")
plt.tight_layout()
plt.savefig('pdmp_DM_timeseries.png')
