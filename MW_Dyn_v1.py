from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
import random
from astropy.io import ascii
from astropy.table import Table

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

"""flag_list should contain the flags that are to be removed from the data if raised"""

flag_list = ['flag_any']

try:
    bad_rows
except NameError:
    bad_rows=[]
    
    for i in range(len(flag_list)):
        
        for j in range(len(data[flag_list[i]])):
            
            if data['flag_any'][j]==1:
                bad_rows.append(j)
            
data.remove_rows(bad_rows)

RA = data['RAdeg']*u.degree
DEC = data['DEdeg']*u.degree
dist = data['distance']*u.pc
pm_RA = data['pmRA_TGAS']*u.mas/u.yr
pm_DEC = data['pmDE_TGAS']*u.mas/u.yr
rad_vel = data['HRV']*u.km/u.s

e_RA = np.zeros(len(data))*u.degree
e_DEC = np.zeros(len(data))*u.degree
e_dist = data['edistance']*u.pc
e_pm_RA = data['pmRA_error_TGAS']*u.mas/u.yr
e_pm_DEC = data['pmDE_error_TGAS']*u.mas/u.yr
e_rad_vel = data['eHRV']*u.km/u.s


############## Bootstrapper ##################

my_data_order=['RA', 'DEC', 'dist', 'pm_RA', 'pm_DEC', 'rad_vel']

my_sample = np.array([RA, DEC, dist, pm_RA, pm_DEC, rad_vel])

e_my_sample = np.array([e_RA, e_DEC, e_dist, e_pm_RA, e_pm_DEC, e_rad_vel])

class Bootstrap:
    
    
    """The bootstrap class which holds a sample and can perform statistical tests through bootstrapping. 
    
    Takes the args:
        
        sample: The data sample at hand. Needs to be converted from Tablet to a set of arrays
        e_sample: Errors in the sample quantities if any. Required to use Bootstrap.bootstrap_err. Should be same shape as sample
        data_order: Just a conventient way to see the order of quantities in sample. Not needed to initialise
        
    Functions:
        
        bootstrap_err: implements bootstrapping for uncertainties. Returns a resampling of the original sample
        bootstrap_rand: creates a random resample of angular velocities from the original sample
        bootstrap_mean: computes the mean of N resamples from bootstrap_err or bootstrap_mean.
        get_st_dev: computes the standard deviation for N resamples in every bin for either the 'error' or 'rand' method
        model_vel: creates a pseudosample from a velocity model for each coordinate in self.sample, 
                then adds a random uncertainty within given range for each coordinate.
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
        
        self.icrs=coord.ICRS(ra = self.sample[0]*u.degree,dec = self.sample[1]*u.degree,
                        distance=self.sample[2]*u.pc,
                        pm_ra_cosdec=self.sample[3]*u.mas/u.yr,
                        pm_dec=self.sample[4]*u.mas/u.yr,
                        radial_velocity=self.sample[5]*u.km/u.s)


        self.gc = self.icrs.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        self.gc.set_representation_cls(coord.CylindricalRepresentation)

        self.icrs_res=None
        
        self.v_phi = (self.gc.d_phi*self.gc.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        self.v_phis = None
        
        self.gc_res = None
        
        self.res_v_phi = None
        
        self.bin_heights = None
        
        self.re_bin_heights = None
        
        self.re_bin_vals = None
        
        self.bin_vals = None

    def bootstrap_err(self):
        
        if any(self.e_sample) == None:
            raise Exception('Uncertainties are needed to perform this action')
            
        err = self.e_sample*np.random.uniform(low=-1,high=1,size=(self.e_sample.shape))
        
#        for i in range(2,len(self.e_sample)):  
#            
#            for j in range(len(self.sample_tp)):
#                
#                err = self.e_sample[i][j]
#                
#                n_deci = len(str(err).split('.')[1]) 
#                
#                rand_err = round(random.uniform(-err,err),n_deci)
#                
#                self.resample[i][j] = self.sample[i][j]+rand_err
        
        self.resample = self.sample + err
                
        self.icrs_res=coord.ICRS(ra = self.sample[0]*u.degree,dec = self.sample[1]*u.degree,
                            distance=self.resample[2]*u.pc,
                            pm_ra_cosdec=self.resample[3]*u.mas/u.yr,
                            pm_dec=self.resample[4]*u.mas/u.yr,
                            radial_velocity=self.resample[5]*u.km/u.s)

        self.gc_res = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (self.gc_res.d_phi*self.gc.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
                       
        return self.res_v_phi
    
    def bootstrap_rand(self):
        
        for i in range(len(self.sample_tp)):
            
            self.resample_tp[i]=random.choice(self.sample_tp)
            
        self.resample = self.resample_tp.transpose()
            
        self.icrs_res=coord.ICRS(ra = self.resample[0]*u.degree,dec = self.resample[1]*u.degree,
                            distance=self.resample[2]*u.pc,
                            pm_ra_cosdec=self.resample[3]*u.mas/u.yr,
                            pm_dec=self.resample[4]*u.mas/u.yr,
                            radial_velocity=self.resample[5]*u.km/u.s)

        self.gc_res = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (self.gc_res.d_phi*self.gc_res.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        return self.res_v_phi
        
    def bootstrap_mean(self, N, N_bins, method):
        
        s = np.zeros([N_bins])
        
        if method == 'rand':
            
            func = bootstrap.rand().value
            
        else:
            
            func = bootstrap.err().value
            
        for i in range(N):
            
            self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=N_bins)
            
            self.re_bin_heights, bin_vals = np.histogram(func, bins=self.bin_vals)
            
            s += self.re_bin_heights
            
        self.mean_sample = s/N
            
        return self.mean_sample
    
    
    def get_st_dev(self, N_bins, N, method):#, dip=False, dip_lim=None):
        
        self.v_phis = np.zeros([N,N_bins])
        
        s = np.zeros(N_bins)
        
        var = np.zeros(N_bins)
        
        self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=N_bins)
        
        if method in ('error','err'):
            
            func = self.bootstrap_err
        
        elif method in ('random','rand'):
            
            func = self.bootstrap_rand
        
        else:
            
            func = self.model_vel
            
            self.bin_vals = 100
            
        for i in range(N):
            
            if i == N/2:
                print('Resampling 50 % done')
            
            self.re_bin_heights, self.re_bin_vals = np.histogram(func(), bins=self.bin_vals)
            
            self.v_phis[i] = self.re_bin_heights

            s+=self.re_bin_heights
        
        print('Resampling done')
            
        s = s/N
            
        for i in range(N_bins):
            
            if i == N_bins/2:
                
                print('50 % done computing sigma')
            
            for j in range(N):
                
                var[i] += (self.v_phis[j][i] - s[i])**2
            
        self.mean_sample = s
    
        st_dev = sqrt(var/N)
        
        self.st_dev = st_dev
        
        return self.st_dev
    
    def model_vel(self):#, dip=False, dip_lim=None):
    
        dip=False
        dip_lim=0
    
        wthin = 0.75
        wthick=0.2
        whalo = 1.-wthin-wthick
        
        thin0 = np.array([0,215,0])
        thick0 = np.array([0,180,0])
        halo0 = np.array([0,0,0])
        
        thin_disp  = np.array([30,20,17])
        thick_disp  = np.array([80,60,55])
        halo_disp = np.array([160,100,100])
        
        which = np.random.random_sample(len(self.gc))
        
        vel_tot=np.zeros([len(self.gc),3])
    
        for j in range(len(self.gc)):
    
            if which[j] < wthin :
                velocity = thin0 + np.random.randn(3)*thin_disp
            elif which[j] < wthin+wthick :
                velocity = thick0 + np.random.randn(3)*thick_disp
            else :
                velocity = halo0 + np.random.randn(3)*halo_disp
                
            if dip==True and abs(velocity[1])<=dip_lim:
                velocity[1]=0
                
            vel_tot[j] = velocity
        
        vel_tot = vel_tot.transpose()
        vel_tot[1]=vel_tot[1]/self.gc.rho.to(u.km).value
        
        cyl_diff = coord.CylindricalDifferential(d_rho=vel_tot[0]*u.km/u.s,d_phi=vel_tot[1]/u.s,d_z=vel_tot[2]*u.km/u.s)
        
        self.gc_res = coord.Galactocentric(representation = coord.CylindricalRepresentation, rho=self.gc.rho,
                                           phi=self.gc.phi,z=self.gc.z,d_rho=cyl_diff.d_rho.to((u.mas*u.pc)/(u.yr*u.rad),equivalencies=u.dimensionless_angles()),
                                           d_phi=cyl_diff.d_phi.to(u.mas/u.yr,equivalencies=u.dimensionless_angles()),
                                           d_z=cyl_diff.d_z.to((u.mas*u.pc)/(u.yr*u.rad),equivalencies=u.dimensionless_angles()),
                                           galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun, differential_cls=coord.CylindricalDifferential)
        
        self.icrs_res = self.gc_res.transform_to(coord.ICRS)
        self.icrs_res.set_representation_cls(coord.SphericalRepresentation,s=coord.SphericalCosLatDifferential)
        
        err = self.e_sample*np.random.uniform(low=-1,high=1,size=(self.e_sample.shape))
        
        self.resample = array([self.icrs_res.ra,self.icrs_res.dec,self.icrs_res.distance,self.icrs_res.pm_ra_cosdec,self.icrs_res.pm_dec,self.icrs_res.radial_velocity]) + err
                
        self.icrs_res = coord.ICRS(ra = self.resample[0]*u.degree,dec = self.resample[1]*u.degree,
                            distance=self.resample[2]*u.pc,
                            pm_ra_cosdec=self.resample[3]*u.mas/u.yr,
                            pm_dec=self.resample[4]*u.mas/u.yr,
                            radial_velocity=self.resample[5]*u.km/u.s)
        
        self.gc_res = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist*u.kpc, galcen_v_sun=v_sun))
        self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (self.gc_res.d_phi*self.gc.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        return self.res_v_phi
    
    
    def plot_sample(self,lim,N_bins=None):
        
        if N_bins == None:
            N_bins = 'auto'
        
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi\ /\ \mathrm{km\ s}^{-1}$', fontdict=font)
        
        plt.hist(self.v_phi, bins=N_bins, log=True, range=(-lim,lim),histtype='step',label='Sample')
        
        plt.legend()
        return 
    
    def plot_resamples(self, N, method, lim, N_bins=None ):
        
        
        if N_bins == None:
            N_bins = 'auto'
        
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi\ /\ \mathrm{km\ s}^{-1}$', fontdict=font)
        
        plt.hist(self.v_phi.value, bins=N_bins, log=True, range=(-lim,lim),histtype='step',label='Original sample')

        if method in ('random','rand'):
                
            func = self.bootstrap_rand
        elif method in ('model','mod'):
            
            func = self.model_vel
			
        else:
                
            func = self.bootstrap_err

        for i in range(N):
    
            func()
        
            plt.hist(self.res_v_phi, bins=N_bins, log=True, range=(-lim,lim),histtype='step')
    
        plt.legend()
        
        return

    def plot_mean(self, lim, err=False, N_bins=None, ymax=None,ymin=None, model=False):        

        if any(self.mean_sample) == None:
            raise Exception('You need to compute a mean using your method of choice before plotting')
            
        if any(self.bin_vals)==None:
            self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=N_bins)

        if ymax == None:
            ymax = 100000
        
        if ymin == None:
            ymin = 1
            
        if N_bins == None:
            N_bins = len(self.mean_sample)            
    
        if err is True:
            err = self.st_dev
            
        plt.figure()
        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$')
        plt.ylabel('$\mathrm{Number\ of\ stars}$')
        plt.xlabel('$v_\phi\ /\ \mathrm{km\ s}^{-1}$', fontdict=font)
        
        if model == True:
            
            plt.bar(self.re_bin_vals[:-1], self.mean_sample,width=np.diff(self.re_bin_vals),color='none', log=True,label='Mean of modeled $v_\phi$ using {} bins'.format(N_bins),edgecolor='red')
            
            plt.errorbar(self.re_bin_vals[:-1], self.mean_sample, yerr=err, fmt='none',ecolor='black', elinewidth=min(np.diff(self.re_bin_vals))/6 )

            plt.xlim(-lim,lim)
            plt.ylim(ymin,ymax)
            plt.legend()
            
            return
           
        plt.legend()
        
        plt.bar(self.bin_vals[:-1], self.bin_heights, width=np.diff(self.bin_vals),color='none',edgecolor='blue', log=True,label='Sample')
        plt.bar(self.bin_vals[:-1], self.mean_sample, width=np.diff(self.bin_vals),color='none', log=True,label='Mean of resample using {} bins'.format(N_bins),edgecolor='orange')

        plt.errorbar(self.bin_vals[:-1], self.mean_sample, yerr = err, fmt = 'none', ecolor = 'black', elinewidth=min(np.diff(self.bin_vals))/6)


        return
        
    def save_mean_data(self,filename):
        
        return ascii.write([self.mean_sample,self.st_dev], filename+'.txt',names=['v_phi','sigma'])
    
############ Velocity dispersion model ############
        
#smp.model_vel()
#v_phi = smp.gc_res.d_phi*smp.gc_res.rho.to(u.kpc)
#
#v_phi = v_phi.to(u.km/u.s,equivalencies=u.dimensionless_angles())
#
#plt.figure()       
#plt.hist(v_phi.value, log=True, bins=100, range=(-400,400))
#plt.ylim(1,100000)

############## Tests ##################

#wthin = 0.75
#wthick=0.2
#whalo = 1.-wthin-wthick
#
#thin0 = np.array([0,215,0])
#thick0 = np.array([0,180,0])
#halo0 = np.array([0,0,0])
#
#
#thin_disp  = np.array([30,20,17])
#thick_disp  = np.array([80,60,55])
#halo_disp = np.array([160,100,100])
#
#which = np.random.rand()
#if which < wthin :
#    velocity = thin0 + np.random.randn(3)*thin_disp
#elif which < wthin+wthick :
#    velocity = thick0 + np.random.randn(3)*thick_disp
#else :
#    velocity = halo0 + np.random.randn(3)*halo_disp
#        

#################### Tests ########################

#cProfile.run('smp.bootstrap_err(e_my_sample)')

        
smp = Bootstrap(my_sample,e_my_sample,my_data_order)

#cProfile.run('smp.get_st_dev(100,10,str(rand))')

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