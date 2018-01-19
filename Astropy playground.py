from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii
from astropy.table import Table
from astropy.stats import histogram as hist
from random import *

import cProfile
import re


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

icrs=coord.ICRS(ra = RA,dec = DEC,distance=dist,
                pm_ra_cosdec=pm_RA,
                pm_dec=pm_DEC,
                radial_velocity=rad_vel)

#e_icrs = coord.ICRS(ra=RA, dec= e_DEC, distance = e_dist,
#                    pm_ra_cosdec=e_pm_RA, pm_dec=e_pm_DEC,
#                    radial_velocity = e_rad_vel)

gc = icrs.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
gc.set_representation_cls(coord.CylindricalRepresentation)


############## Bootstrapper ##################



my_sample = np.array([dist, pm_RA, pm_DEC, rad_vel])

e_my_sample = np.array([e_dist, e_pm_RA, e_pm_DEC, e_rad_vel])



def bootstrap_err(sample, e_sample):
    
    """Function that implements bootstrapping for uncertainties. Takes the arguments:
    sample: the sample at hand containing all the data
    e_sample: the errors for the quantities in the sample"""
    
    
    for i in range(len(sample)):  
        
        for j in range(len(sample[i])):
            
            err = e_sample[i][j]
            
            
            rand_err = round(uniform(-err,err),len(str(err).split('.')[1]))
                 
            
            sample[i][j] = sample[i][j]+rand_err
                   
#    icrs_res=coord.ICRS(ra = RA,dec = DEC,distance=sample[0]*u.pc,
#        pm_ra_cosdec=sample[1]*u.mas/u.yr,
#        pm_dec=sample[2]*u.mas/u.yr,
#        radial_velocity=sample[3]*u.km/u.s)
#    
#    gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
#    gc_res.set_representation_cls(coord.CylindricalRepresentation)
#    
    return sample

#cProfile.run('bootstrap_err(my_sample,e_my_sample)')

#try:
#    resample
#except NameError:  
#
#    
#    resample = bootstrap_err(data, para, epara)
#    
#    icrs_res=coord.ICRS(ra = resample['RAdeg']*u.degree,dec = resample['DEdeg']*u.degree,distance=resample['distance']*u.pc,
#                    pm_ra_cosdec=resample['pmRA_TGAS']*u.mas/u.yr,
#                    pm_dec=resample['pmDE_TGAS']*u.mas/u.yr,
#                    radial_velocity=resample['HRV']*u.km/u.s)
#    
#    gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = 8.20*u.kpc, galcen_v_sun=v_sun))
#    gc_res.set_representation_cls(coord.CylindricalRepresentation)
#    pass



"""The plotter"""

N=5

plt.figure()
plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
plt.xlabel('$v_\phi$')
plt.hist(gc.d_phi, bins=100, log=True, range=(-12,12),histtype='step')

for i in range(N):
    resample = bootstrap_err(my_sample,e_my_sample)
    plt.hist(resample.d_phi, bins=100, log=True, range=(-12,12),histtype='step')
plt.ylim(0,600)

plt.show()

#plt.savefig('100_bins.jpg')


#data2 = Table(data, copy=True)
#
#err = data2['edistance'][0]
#
#rand_err = round(uniform(-err,err),len(str(err).split('.')[1]))
#
#data2['distance'][0] = data['distance'][0]+rand_err
#
#print(data2['distance'][0])
        
    

#def bootstrap_rand(data):
        
def bootstrap_mean(sample, e_sample, N):
    
    
    
    for i in range(N):
        
        bootstrap_err(sample,e_sample)
        
        

    