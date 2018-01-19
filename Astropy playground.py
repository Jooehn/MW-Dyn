from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii
from astropy.table import Table
#from astropy.stats import histogram as hist
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

"""Obtaining data in GC coords after initiating with ICRS frame"""





############## Bootstrapper ##################

my_data_order=['dist', 'pm_RA', 'pm_DEC', 'rad_vel']

my_sample = np.array([dist, pm_RA, pm_DEC, rad_vel])

e_my_sample = np.array([e_dist, e_pm_RA, e_pm_DEC, e_rad_vel])

class Bootstrap:
    
    
    """The bootstrap class which holds a sample and can perform statistical tests through bootstrapping"""
    
    def __init__(self,sample,data_order):
        
        self.data_order = data_order
        
        self.sample = sample
        
        self.resample = np.zeros([len(sample),len(sample[0])])
        
        self.mean_sample = None

    def bootstrap_err(self, e_sample):
        
        """Function that implements bootstrapping for uncertainties. Takes the arguments:
        e_sample: the errors for the quantities in the sample"""
        
        
        
        for i in range(len(self.sample)):  
            
            for j in range(len(self.sample[i])):
                
                err = e_sample[i][j]
                
                n_deci = len(str(err).split('.')[1]) 
                
                rand_err = round(uniform(-err,err),n_deci)
                     
                
                self.resample[i][j] = self.sample[i][j]+rand_err
                       
  
        return self.resample
    
    #def bootstrap_rand(data):
        
    def bootstrap_mean(self, e_sample,method, N):
        
        
        s = np.zeros([len(self.sample),len(self.sample[0])])

        
        if method is 'error':
            
            for i in range(N):
                
                s = s + self.bootstrap_err(e_sample)
                
        if method is 'random':
            
            for i in range(N):
                
                s = s + self.bootstrap_rand(e_sample)
                
        self.mean_sample = s/N
        
        return self.mean_sample
    
    def plot_sample(self,N_bins,Range):
        
        icrs=coord.ICRS(ra = RA,dec = DEC,distance=self.sample[0]*u.pc,
                pm_ra_cosdec=self.sample[1]*u.mas/u.yr,
                pm_dec=self.sample[2]*u.mas/u.yr,
                radial_velocity=self.sample[3]*u.km/u.s)

        gc = icrs.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        gc.set_representation_cls(coord.CylindricalRepresentation)
        
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi$')

        plt.hist(gc.d_phi, bins=N_bins, log=True, range=(-Range,Range),histtype='step',label='Sample')
        plt.legend()
        return 
    
    def plot_mean(self, N_bins,Range):
        
        icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=self.mean_sample[0]*u.pc,
                pm_ra_cosdec=self.mean_sample[1]*u.mas/u.yr,
                pm_dec=self.mean_sample[2]*u.mas/u.yr,
                radial_velocity=self.mean_sample[3]*u.km/u.s)

        gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        gc_res.set_representation_cls(coord.CylindricalRepresentation)
        
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi$')
        
        self.plot_sample(N_bins,Range)
        plt.hist(gc_res.d_phi, bins=N_bins, log=True, range=(-Range,Range),histtype='step',label='Mean of resample')
        plt.legend()
        plt.show()
        return
        
    def save_mean_data(self,filename):
        
        return ascii.write(self.mean_sample.transpose(), filename,names=self.data_order)
        

############## Datasaver ##################  



"""The plotter"""


#icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=smp.mean_sample[0]*u.pc,
#                pm_ra_cosdec=smp.mean_sample[1]*u.mas/u.yr,
#                pm_dec=smp.mean_sample[2]*u.mas/u.yr,
#                radial_velocity=smp.mean_sample[3]*u.km/u.s)
#
#gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
#gc_res.set_representation_cls(coord.CylindricalRepresentation)
#
#
#plt.figure()
#plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
#plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
#plt.xlabel('$v_\phi$')
#plt.hist(gc.d_phi, bins=100, log=True, range=(-4,4),histtype='step',label='Sample')
#plt.hist(gc_res.d_phi, bins=100, log=True, range=(-4,4),histtype='step',label='Mean of 100 resamples')
#plt.legend(loc=4)
#
##for i in range(N):
##    resample = bootstrap_err(my_sample,e_my_sample)
##    plt.hist(resample.d_phi, bins=100, log=True, range=(-12,12),histtype='step')
##plt.ylim(0,600)
#
#plt.show()

#plt.savefig('100_bins.png')