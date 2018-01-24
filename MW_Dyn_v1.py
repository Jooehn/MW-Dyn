from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii
from astropy.table import Table
from random import *

import cProfile


############## Initialiser ##################

gc_sun_dist = 8.20

v_rot = 232.8

v_sun = coord.CartesianDifferential((11.1,v_rot,7.25), unit=u.km/u.s)

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }


try:
    data_raw
except NameError:
    data_raw = ascii.read('Distances_PJM2017.csv', format='fast_csv')
    pass

data = Table(data_raw, copy=True)

RA = data['RAdeg']*u.degree
DEC = data['DEdeg']*u.degree
dist = data['distance']*u.pc
pm_RA = data['pmRA_TGAS']*u.mas/u.yr
pm_DEC = data['pmDE_TGAS']*u.mas/u.yr
rad_vel = data['HRV']*u.km/u.s

e_dist = data['edistance']*u.pc
e_pm_RA = data['pmRA_error_TGAS']*u.mas/u.yr
e_pm_DEC = data['pmDE_error_TGAS']*u.mas/u.yr
e_rad_vel = data['eHRV']*u.km/u.s

"""Removing data for given thresholds of relative distances and errors"""

#row_ind=[]
#
#for i in range(len(data['distance'])):
#    if data['edistance'][i]/data['distance'][i] >= 0.25:
#        row_ind.append(i)
#        
#    if data['distance'][i] >= 700:
#        row_ind.append(i)
#
#
#data.remove_rows(row_ind)

############## Bootstrapper ##################

my_data_order=['dist', 'pm_RA', 'pm_DEC', 'rad_vel']

my_sample = np.array([dist, pm_RA, pm_DEC, rad_vel])

e_my_sample = np.array([e_dist, e_pm_RA, e_pm_DEC, e_rad_vel])

class Bootstrap:
    
    
    """The bootstrap class which holds a sample and can perform statistical tests through bootstrapping. 
    
    Takes the args:
        
        sample: The data sample at hand. Needs to be converted from Tablet to a set of arrays
        e_sample: Errors in the sample quantities if any. Required to use Bootstrap.bootstrap_err. Should be same shape as sample
        data_order: Just a conventient way to see the order of quantities in sample. Not needed to initialise
        
    Functions:
        
        bootstrap_err: implements bootstrapping for uncertainties. Returns a resampling of the original sample
        bootstrap_rand: creates a random resample of angular velocities from the original sample
        bootstrap_mean: computes the mean of N resamples from bootstrap_err or bootstrap_mean. If the intention is to compute
            the average value from bootstrap_rand, the average amount of counts in each bin is returned. 
            Thus the number of bins need to be given in N_bins
        plot_sample: plots the original sample in a histogram
        plot_resample: plots N resamples computed using a given method within a given v_phi limit
        plot_mean: plots the computed mean that is stored in self.mean_sample
        save_mean_data: saves the current data stored in self.mean_sample to an ascii file
    """
    
    def __init__(self,sample,e_sample=None,data_order=None):
        
        
        self.data_order = data_order
        
        self.sample = sample
        
        self.e_sample = e_sample
        
        self.sample_tp = self.sample.transpose()
        
        self.resample = np.zeros(shape(self.sample))
        
        self.resample_tp = self.resample.transpose()
        
        self.mean_sample = None
        
        self.st_dev = None
        
        icrs=coord.ICRS(ra = RA,dec = DEC,distance=self.sample[0]*u.pc,
                        pm_ra_cosdec=self.sample[1]*u.mas/u.yr,
                        pm_dec=self.sample[2]*u.mas/u.yr,
                        radial_velocity=self.sample[3]*u.km/u.s)

        self.gc = icrs.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        self.gc.set_representation_cls(coord.CylindricalRepresentation)
        
        self.gc_res = None
        
        self.bin_heights = None
        
        self.re_bin_heights = None
        
        self.bin_vals = None

    def bootstrap_err(self):
        
        if any(self.e_sample) == None:
            raise Exception('Uncertainties are needed to perform this action')
        
        for i in range(len(self.sample)):  
            
            for j in range(len(self.sample[i])):
                
                err = self.e_sample[i][j]
                
                n_deci = len(str(err).split('.')[1]) 
                
                rand_err = round(uniform(-err,err),n_deci)
                
                self.resample[i][j] = self.sample[i][j]+rand_err
                       
        return self.resample
    
    def bootstrap_rand(self):
        
        for i in range(len(self.sample_tp)):
            
            self.resample_tp[i]=choice(self.sample_tp)
            
        self.resample = self.resample_tp.transpose()
            
        icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=self.resample[0]*u.pc,
                            pm_ra_cosdec=self.resample[1]*u.mas/u.yr,
                            pm_dec=self.resample[2]*u.mas/u.yr,
                            radial_velocity=self.resample[3]*u.km/u.s)

        self.gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
    
        return self.gc_res.d_phi.value
        
    def bootstrap_mean(self, N, N_bins=None):
        
        if N_bins != None:
            
            s = np.zeros([N_bins])

            
            for i in range(N):
                
                self.bin_heights, self.bin_vals = np.histogram(self.gc.d_phi.value, bins=N_bins)
                
                self.re_bin_heights, bin_vals = np.histogram(self.bootstrap_rand(), bins=self.bin_vals)
                
                s += self.re_bin_heights

        else:
            
            s = np.zeros(shape(self.sample))
            
            for i in range(N):
                
                s+= self.bootstrap_err()
            
        self.mean_sample = s/N
            
        return self.mean_sample
    
    
#    def get_st_dev(self, N_bins, N, method):
#        
#        
#        s = np.zeros(N_bins)
#        
#        self.bin_heights, self.bin_vals = np.histogram(self.gc.d_phi.value, bins=N_bins)
#        
#        if method == 'error':
#            
#            for i in range(N):
#                
#                
#        
#                icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=self.resample[0]*u.pc,
#                        pm_ra_cosdec=self.resample[1]*u.mas/u.yr,
#                        pm_dec=self.resample[2]*u.mas/u.yr,
#                        radial_velocity=self.resample[3]*u.km/u.s)
#        
#                self.gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
#                self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
#                
#                self.re_bin_heights, bin_vals = np.histogram(self.gc_res.d_phi, bins=self.bin_vals)
#         
#                s += (self.bin_heights-self.re_bin_heights)**2
#            
#        else:
#            
#            for i in range(N):
#    
#                self.re_bin_heights, bin_vals = np.histogram(self.bootstrap_rand(), bins=self.bin_vals)
#              
#                s += (self.bin_heights-self.re_bin_heights)**2
#    
#        self.st_dev = sqrt(s/N)
#        
#        return self.st_dev
    
    def plot_sample(self,lim,N_bins=None):
        
        if N_bins == None:
            N_bins = 'auto'
        
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi$')
        
        plt.hist(self.gc.d_phi, bins=N_bins, log=True, range=(-lim,lim),histtype='step',label='Sample')
        
        plt.legend()
        return 
    
    def plot_resamples(self, N, method, lim, N_bins=None ):
        
        
        if N_bins == None:
            N_bins = 'auto'
        
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi$')
        
        plt.hist(self.gc.d_phi.value, bins=N_bins, log=True, range=(-lim,lim),histtype='step',label='Original sample')

        for i in range(N):

            if method == 'random':
                
                self.bootstrap_rand()
                d_phi = self.gc_res.d_phi.value
                
            else:
                
                self.bootstrap_err()
                
                icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=self.resample[0]*u.pc,
                pm_ra_cosdec=self.resample[1]*u.mas/u.yr,
                pm_dec=self.resample[2]*u.mas/u.yr,
                radial_velocity=self.resample[3]*u.km/u.s)
        
                self.gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
                self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
                
                d_phi = self.gc_res.d_phi
    
    
        
            plt.hist(d_phi, bins=N_bins, log=True, range=(-lim,lim),histtype='step')
    
        plt.legend()
        
    
    def plot_mean(self, lim, N_bins=None):
        
        if any(self.mean_sample) == None:
            raise Exception('You need to compute a mean using your method of choice before plotting')
        
        if N_bins == None:
            N_bins = len(self.mean_sample)
            
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$')
        plt.ylabel('$\mathrm{Number\ of\ stars}$')
        plt.xlabel('$v_\phi$')
        
        if self.mean_sample.shape == (shape(self.sample)):

            icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=self.mean_sample[0]*u.pc,
                    pm_ra_cosdec=self.mean_sample[1]*u.mas/u.yr,
                    pm_dec=self.mean_sample[2]*u.mas/u.yr,
                    radial_velocity=self.mean_sample[3]*u.km/u.s)
    
            self.gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
            self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
            
            d_phi = self.gc_res.d_phi
            
            self.plot_sample(lim,N_bins)
            plt.hist(d_phi, bins=N_bins, log=True, range=(-lim,lim),histtype='step',label='Mean of resample using {} bins'.format(N_bins))
            
        else:
            d_phi = self.mean_sample
            
            plt.bar(self.bin_vals[:-1], self.bin_heights,color='blue', log=True,label='Sample')
            plt.bar(self.bin_vals[:-1], self.mean_sample,color='orange', log=True,label='Mean of resample using {} bins'.format(N_bins))#, range=(-lim,lim),histtype='step',label='Mean of resample')
            plt.xlim(-lim,lim)
            
           
        plt.legend()
        return
        
    def save_mean_data(self,filename):
        
        return ascii.write(self.mean_sample.transpose(), filename,names=self.data_order)
        

############## Tests ##################

#cProfile.run('smp.bootstrap_err(e_my_sample)')
        
smp = Bootstrap(my_sample,e_my_sample,my_data_order)

"""1. Making a histogram with 100 bins of original sample and 10 resamples using bootstrap_rand"""

#smp.plot_resamples(10, 'random', 14, N_bins=50)

"""2. Making a histogram with 100 bins of original sample and 5 resamples using bootstrap_err. Takes ~30 s!"""

#smp.plot_resamples(5, 'error', 14, N_bins=100)

"""3. Computing the mean for 10 iterations of bootstrap_rand using 100 bins and plots the result."""

#smp.bootstrap_mean(10)
#
#smp.plot_mean(10)

"""4. Computing the mean for 5 iterations of bootstrap_err and plots the result"""

#smp.bootstrap_mean(5)
#
#smp.plot_mean(14,100)