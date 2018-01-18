from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii
from astropy.table import Table
from astropy.stats import histogram as hist


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

data = Table(data_raw, copy=False)

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

ra = coord.Angle(data['RAdeg'], unit=u.degree)

dec = coord.Angle(data['DEdeg'], unit=u.degree)

icrs=coord.ICRS(ra = ra,dec = dec,distance=data['distance']*u.pc,
                pm_ra_cosdec=data['pmRA_TGAS']*u.mas/u.yr,
                pm_dec=data['pmDE_TGAS']*u.mas/u.yr,
                radial_velocity=data['HRV']*u.km/u.s)

v_sun = coord.CartesianDifferential((11.1,232.8,7.25), unit=u.km/u.s)

gc = icrs.transform_to(coord.Galactocentric(galcen_distance = 8.20*u.kpc, galcen_v_sun=v_sun))
gc.set_representation_cls(coord.CylindricalRepresentation)

plt.figure()
plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
plt.hist(gc.d_phi, bins='auto', log=True)
#plt.ylim(0,600)
plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
plt.xlabel('$v_\phi$')
plt.show()



"""Removing rows for a certain threshold of edistance/distance. Very slow."""

#for i in range(len(data['distance'])):
#    
#    if data['edistance'][i]/data['distance'][i] >= 0.25:
#        data.remove_rows(i)


"""Plotting the distribution of stars with galactic coordinates"""

#glon = coord.Angle(data['Glon'], unit=u.degree)
#
#glon = glon.wrap_at(180*u.degree)
#
#glat = coord.Angle(data['Glat'], unit=u.degree)


#fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111, projection="mollweide")
#ax.scatter(glon.radian, glat.radian,s=0.1)
#ax.grid(True)