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
cProfile.run('bootstrap_err(data, para, epara)')

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


icrs=coord.ICRS(ra = data['RAdeg']*u.degree,dec = data['DEdeg']*u.degree,distance=data['distance']*u.pc,
                pm_ra_cosdec=data['pmRA_TGAS']*u.mas/u.yr,
                pm_dec=data['pmDE_TGAS']*u.mas/u.yr,
                radial_velocity=data['HRV']*u.km/u.s)



gc = icrs.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
gc.set_representation_cls(coord.CylindricalRepresentation)


############## Bootstrapper ##################



def bootstrap_err(table, par, e_par):
    
    """Function that implements bootstrapping for uncertainties. Takes the arguments:
    table: the table at hand containing all the data
    par: a list of strings that describe all the quantities that should be resampled
    e_par: same type of list as par but for the error in said quantities"""
    
    boot = Table(table, copy = True)
    
    for i in range(len(par)):
        

        
        for j in range(len(boot[par[i]])):
            
            err = boot[e_par[i]][j]
            
            
            rand_err = round(uniform(-err,err),len(str(err).split('.')[1]))
                 
            
            boot[par[i]][j] = boot[par[i]][j]+rand_err
                   
    icrs_res=coord.ICRS(ra = boot['RAdeg']*u.degree,dec = boot['DEdeg']*u.degree,distance=boot['distance']*u.pc,
        pm_ra_cosdec=boot['pmRA_TGAS']*u.mas/u.yr,
        pm_dec=boot['pmDE_TGAS']*u.mas/u.yr,
        radial_velocity=boot['HRV']*u.km/u.s)
    
    gc_res = icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
    gc_res.set_representation_cls(coord.CylindricalRepresentation)
    
    return gc_res


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

#N=5
#
#resample = bootstrap_err(data, para, epara)
#
#plt.figure()
#plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
#plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
#plt.xlabel('$v_\phi$')
#plt.hist(gc.d_phi, bins=100, log=True, range=(-12,12),histtype='step')
#
#for i in range(N):
#    resample = bootstrap_err(data, para, epara)
#    plt.hist(resample.d_phi, bins=100, log=True, range=(-12,12),histtype='step')
#plt.ylim(0,600)

#plt.show()

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
        
#def bootstrap_mean(data, N):
    