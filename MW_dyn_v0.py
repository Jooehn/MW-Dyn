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

############## Bootstrapper ##################

my_data_order=['dist', 'pm_RA', 'pm_DEC', 'rad_vel']

my_sample = np.array([dist, pm_RA, pm_DEC, rad_vel])

e_my_sample = np.array([e_dist, e_pm_RA, e_pm_DEC, e_rad_vel])

class Bootstrap:
    
    
    """The bootstrap class which holds a sample and can perform statistical tests through bootstrapping"""
    
    def __init__(self,sample,data_order):
        
        
        self.data_order = data_order
        
        self.sample = sample
        
        self.sample_tp = self.sample.transpose()
        
        self.resample = np.zeros(shape(self.sample))
        
        self.resample_tp = self.resample.transpose()
        
        self.mean_sample = None
        
        icrs=coord.ICRS(ra = RA,dec = DEC,distance=self.sample[0]*u.pc,
                        pm_ra_cosdec=self.sample[1]*u.mas/u.yr,
                        pm_dec=self.sample[2]*u.mas/u.yr,
                        radial_velocity=self.sample[3]*u.km/u.s)

        self.gc = icrs.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        self.gc.set_representation_cls(coord.CylindricalRepresentation)
        
        self.gc_res = None

    def bootstrap_err(self, e_sample):
        
        """Function that implements bootstrapping for uncertainties. Returns a resampling of the original sample."""
        
        for i in range(len(self.sample)):  
            
            for j in range(len(self.sample[i])):
                
                err = e_sample[i][j]
                
                n_deci = len(str(err).split('.')[1]) 
                
                rand_err = round(uniform(-err,err),n_deci)
                
                self.resample[i][j] = self.sample[i][j]+rand_err
                       
        return self.resample
    
    def bootstrap_rand(self):
        
        """Function that creates a random resample of angular velocities from the original sample."""
        
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
        
    def bootstrap_mean(self, N, e_sample=None):
        
        
#        s = np.zeros([len(self.sample),len(self.sample[0])])

        
        if e_sample is None:
            
            s = np.zeros([len(self.sample_tp)])
            
            for i in range(N):
                
                s += self.bootstrap_rand()
                
        else:
            
            s = np.zeros(shape(self.sample))
            
            for i in range(N):
                
                s += self.bootstrap_err(e_sample)
                
        self.mean_sample = s/N
        
        return self.mean_sample
    
    def plot_sample(self,N_bins,lim):
        

        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi$')

        plt.hist(self.gc.d_phi, bins=N_bins, log=True, range=(-lim,lim),histtype='step',label='Sample')
        plt.legend()
        return 
    
    def plot_mean(self, N_bins,lim):
        
        if self.mean_sample.shape == (shape(self.sample)):
            
        
            icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=self.mean_sample[0]*u.pc,
                    pm_ra_cosdec=self.mean_sample[1]*u.mas/u.yr,
                    pm_dec=self.mean_sample[2]*u.mas/u.yr,
                    radial_velocity=self.mean_sample[3]*u.km/u.s)
    
            self.gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
            self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
            
            d_phi = self.gc_res.d_phi
            
        else:
            d_phi = self.mean_sample
        
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi$')
        
        self.plot_sample(N_bins,lim)
        plt.hist(d_phi, bins=N_bins, log=True, range=(-lim,lim),histtype='step',label='Mean of resample')
        plt.legend()
        return
        
    def save_mean_data(self,filename):
        
        return ascii.write(self.mean_sample.transpose(), filename,names=self.data_order)
        


############## Datasaver ##################  

#cProfile.run('smp.bootstrap_err(e_my_sample)')

"""The plotter"""

icrs=coord.ICRS(ra = RA,dec = DEC,distance=smp.sample[0]*u.pc,
        pm_ra_cosdec=smp.sample[1]*u.mas/u.yr,
        pm_dec=smp.sample[2]*u.mas/u.yr,
        radial_velocity=smp.sample[3]*u.km/u.s)

gc = icrs.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
gc.set_representation_cls(coord.CylindricalRepresentation)

N = 10

lim = 14

#plt.figure()
#
#for i in range(N):
#    
#    smp.bootstrap_rand()
#    
#    plt.hist(smp.gc_res.d_phi, bins=100, log=True, range=(-lim,lim),histtype='step')
#    
#
##icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=smp.mean_sample[0]*u.pc,
##            pm_ra_cosdec=smp.mean_sample[1]*u.mas/u.yr,
##            pm_dec=smp.mean_sample[2]*u.mas/u.yr,
##            radial_velocity=smp.mean_sample[3]*u.km/u.s)
##
##plt.hist(smp.gc_res.d_phi, bins=100, log=True, range=(-lim,lim),histtype='step')
#
#plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
#plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
#plt.xlabel('$v_\phi$')
#plt.hist(gc.d_phi, bins=100, log=True, range=(-lim,lim),histtype='step',label='Sample')
#
#plt.legend(loc=1)
#
#plt.show()

#plt.savefig('100_bins.png')
