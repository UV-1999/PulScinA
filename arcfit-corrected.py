# A python code to fit the secondary spectrum with a parabolic arc
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(prog='Final', usage='''python Final.py [DATA] [OPTION]...\nPlease use -h command for help''',
description =
'''
----------------------------------------------------------------------

code i made to estimate the arc curvature... i wanna do uncertainities
but i dont know how to do it exactly...cry cry...

----------------------------------------------------------------------
'''
,formatter_class=argparse.RawDescriptionHelpFormatter, add_help=True)


parser.add_argument('-PSR',
                    help = "J Name of Pulsar")
parser.add_argument('-data',
                    help = " numpy file of the secondary spectrum plot matrix")
parser.add_argument('-bw',
                    help = "Bandwidth in MHz")
parser.add_argument('-obt',
                    help = "Observation time in seconds")

args = parser.parse_args()
PSR = args.PSR
bw = args.bw
dur_sec = args.obt 
data = args.data

data = np.load(data)

def tester():
    #for i in range(0,600):
    #    data[:,i] = 0

    #for i in range(800,1440):
    #    data[:,i] = 0

    bw = 200 # Bandwidth in MHz
    dur_sec = 28798.4189440003 # observation span in seconds
    conjT = np.fft.fftshift(np.fft.fftfreq( data.shape[1]  , d= (float(dur_sec)/float(data.shape[1]))))  # Conjugate time units : Hertz
    conjF = np.fft.fftshift(np.fft.fftfreq( data.shape[0]*2, d= (float(bw)/float(data.shape[0]*2)))) # Conjugate frequency units : micro-seconds
    # The ratio of conjF and conjT is (mili seconds)^2

    a = 0.05101171011710117

    py =  max(conjF)/(data.shape[0])
    px = (max(conjT) - min(conjT))/data.shape[1]
    # py/(px**2) ### the conversion factor

    print ('Curvature paramter:',10e-6*(a*py)/(px**2), 'second cubed')
    # curvature parameter is in 10^-6 second cubed. 
        
    x = np.linspace( min(conjT) , max(conjT) , 1000, dtype = float) 

    plt.imshow(data, aspect = 'auto' , extent = (min(conjT), max(conjT), 0, max(conjF)) )
    plt.set_cmap('jet')
    plt.title('scintillaion arc, curvature parameter fit')
    plt.xlabel('conjugate time in Hz')
    plt.ylabel('conjugate frequency in microssecond')
    plt.plot(x ,((a*py)/(px**2))*x**2, color = 'black', alpha = 0.5, label = 'curvature='+str(10e-6*(a*py)/(px**2))+ '$s^3$')
    plt.ylim([0, 1])
    plt.legend(loc = 'best')
    #plt.savefig('fit.png')
    plt.show()

def fitter(index_fdop_min, index_fdop_max,eta):
# Calculates power along a parabola of curvature = eta
    power = 0
    x = np.linspace(index_fdop_min,  index_fdop_max, abs(index_fdop_max - index_fdop_min +1), dtype = int)
    y = abs((data.shape[0] - 1  - eta*((x-(data.shape[1]/2))**2)).astype(int))
    for i in x:
        if  y[i-index_fdop_min] < 512:
            power = power + data[y[i-index_fdop_min],i]
    return(power)

def trial(min_curv, max_curv, N):
# Takes a range of trial curvature, returns curvature that maximises power
    # range of trial curvatures
    trial = np.linspace(min_curv, max_curv, N, dtype = float)
    print ('Trial curvature range is ['+str(min_curv)+','+str(max_curv)+'] with '+str(N)+' steps')  
    pop = []
    for t in trial:
        pop = np.append(pop, fitter(600,800,t))
    a = np.argmax(pop)
    print ('Curvature parameter: ' + str(trial[a]))

    plt.plot(trial, pop)
    plt.xlabel('trial curvature values')
    plt.ylabel('absolute power')
    plt.axvline(x = trial[a], color = 'red', label = 'maxima')
    plt.legend(loc ='best')
    plt.savefig('run.png')

trial(0.05,0.06,10000)

def npy2txt():
    # The following code is for generating spectrum as numerical data text file
    conjT = np.fft.fftshift(np.fft.fftfreq( data.shape[1]  , d= (float(dur_sec)/float(data.shape[1]))))  # Conjugate time units : Hertz
    conjF = np.fft.fftshift(np.fft.fftfreq( data.shape[0]*2, d= (float(bw)/float(data.shape[0]*2)))) # Conjugate frequency units : micro-seconds
    # The ratio of conjF and conjT is (mili seconds)^2

    # Spectrum as numerical data text file
    file1 = open(str(PSR)+'.txt', 'w')
    for j in range(len(conjT)):
            for i in range(len(conjF)):
                if conjF[-i] > 0:
                    l = str(conjT[j])+ str(' ') + str( conjF[-i]) + str(' ') + str( data[i,j])
                    file1.writelines(l + str('\n'))
    file1.close()

def scatterplot_arcs(a):
    # Plotting the spectrum as a scatter-plot with the fitted parabola
    # Only needed when a is determined
    # a is arc-curvature in units: second cubed
    x, y, z = np.loadtxt(str(PSR)+'.txt', comments='#', usecols=(0,1,2), unpack=True)
    plt.scatter(x,y, s = 1, c = z)
    plt.plot(x,a*(1e5)*x*x, linewidth=1)
    #plt.ylim([0, 1])
    #plt.xlim([-0.02,0.02])
    plt.xlabel(r'fdop in Hz')
    plt.ylabel(r'$\tau$ in $\mu$s')
    plt.title('Secondary spectrum PSR ' + str(PSR) + '\ncurvature = '+str(a)+r'$s^{3}$')
    plt.set_cmap('jet')
    plt.savefig(str(PSR)+'.png')

# for 0837: 2.114129145597985
# for 2048: 0.27309
#scatterplot_arcs()