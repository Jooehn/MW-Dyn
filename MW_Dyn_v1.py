from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
import random
from astropy.io import ascii
from astropy.table import Table
from astropy.constants import G

import cProfile


############## Initialiser ##################

gc_sun_dist = 8.20*u.kpc

gp_z_sun = 15*u.pc

v_rot = 232.8

v_sun = coord.CartesianDifferential((11.1,12.24+v_rot,7.25), unit=u.km/u.s)

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

flag_list = ['flag_dup']

try:
    bad_rows
except NameError:
    bad_rows=[]
    
    for i in flag_list:
        
        for j in range(len(data[i])):
            
            if data[i][j]==1 and j not in bad_rows:
                bad_rows.append(j)
            
data.remove_rows(bad_rows)

RA = data['RAdeg']*u.degree
DEC = data['DEdeg']*u.degree
dist = data['distance']*u.pc
pm_RA = data['pmRA_TGAS']*u.mas/u.yr
pm_DEC = data['pmDE_TGAS']*u.mas/u.yr
rad_vel = data['HRV']*u.km/u.s
mass = data['mass']*u.Msun
met = data['Met_N_K']*u.dex

e_RA = np.zeros(len(data))*u.degree
e_DEC = np.zeros(len(data))*u.degree
e_dist = data['edistance']*u.pc
e_pm_RA = data['pmRA_error_TGAS']*u.mas/u.yr
e_pm_DEC = data['pmDE_error_TGAS']*u.mas/u.yr
e_rad_vel = data['eHRV']*u.km/u.s
e_mass = data['e_mass']*u.Msun
e_met = data['eMet_K']*u.dex


############## Bootstrapper ##################

my_data_order=['RA', 'DEC', 'dist', 'pm_RA', 'pm_DEC', 'rad_vel','mass','metallicity']

my_sample = np.array([RA, DEC, dist, pm_RA, pm_DEC, rad_vel, mass, met])

e_my_sample = np.array([e_RA, e_DEC, e_dist, e_pm_RA, e_pm_DEC, e_rad_vel, e_mass, e_met])

class MW_dyn:
    
    
    """The MW_dyn class which holds a sample of Milky Way stars and can perform statistical tests. 
    
    Takes the args:
        
        sample: The data sample at hand. Needs to be converted from Tablet to a set of arrays
        e_sample: Errors in the sample quantities if any. Required to use Bootstrap.bootstrap_err. Should be same shape as sample
        data_order: Just a conventient way to see the order of quantities in sample. Not needed to initialise
        
    Functions:
        
        bootstrap_err: implements bootstrapping for uncertainties. Returns a resampling of the original sample
        bootstrap_rand: creates a random resample of angular velocities from the original sample
        bootstrap_mean: computes the mean of N resamples from bootstrap_err or bootstrap_mean.
        get_st_dev: computes the standard deviation for N resamples in every bin for either the 'error', 'rand' or 'model' method
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


        self.gc = self.icrs.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
        self.gc.set_representation_cls(coord.CylindricalRepresentation)
        
        self.mass = self.sample[6]*u.Msun
        
        self.ang_mom = np.zeros(len(self.sample))
        
        self.energy = np.zeros(len(self.sample))
        
        self.met = self.sample[7]*u.dex

        self.halo = None

        self.icrs_res = None
        
        self.v_phi = (self.gc.d_phi*self.gc.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        self.v_rho = self.gc.d_rho.to(u.km/u.s)
        
        self.v_z = self.gc.d_z.to(u.km/u.s)
        
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
            
        err = self.e_sample*np.random.randn(self.e_sample.shape[0],self.e_sample.shape[1])
        
        self.resample = self.sample + err
                
        self.icrs_res=coord.ICRS(ra = self.resample[0]*u.degree,dec = self.resample[1]*u.degree,
                            distance=self.resample[2]*u.pc,
                            pm_ra_cosdec=self.resample[3]*u.mas/u.yr,
                            pm_dec=self.resample[4]*u.mas/u.yr,
                            radial_velocity=self.resample[5]*u.km/u.s)

        self.gc_res = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
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

        self.gc_res = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
        self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (self.gc_res.d_phi*self.gc_res.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        return self.res_v_phi
        
    def bootstrap_mean(self, N, N_bins, method):
        
        s = np.zeros([N_bins])
        
        if method == 'rand':
            
            func = self.bootstrap_rand
            
        else:
            
            func = self.bootstrap_err
            
        for i in range(N):
            
            self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=N_bins)
            
            self.re_bin_heights, bin_vals = np.histogram(func(), bins=self.bin_vals)
            
            s += self.re_bin_heights
            
        self.mean_sample = s/N
            
        return self.mean_sample
    
    
    def get_st_dev(self, binwidth, N, method):
        
        self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=np.arange(min(self.v_phi.value),max(self.v_phi.value)+binwidth,binwidth))
        
        N_bins = len(self.bin_vals[:-1])
        
        self.v_phis = np.zeros([N,N_bins])
        
        s = np.zeros(N_bins)
        
        var = np.zeros(N_bins)
        
        if method in ('error','err'):
            
            func = self.bootstrap_err
        
        elif method in ('random','rand'):
            
            func = self.bootstrap_rand
        
        elif method in ('model','mod'):
            
            func = self.model_vel
            
#            dummy, self.bin_vals = np.histogram(func(), bins=np.arange(min(self.v_phi.value),max(self.v_phi.value)+binwidth,binwidth))
            
#            N_bins = len(self.bin_vals[:-1])
#            
#            self.v_phis = np.zeros([N,N_bins])
#        
#            s = np.zeros(N_bins)
#        
#            var = np.zeros(N_bins)
            
        else:
            raise Exception('Not a valid method')
            
        for i in range(N):
            
            if i == N/4:
                print('Resampling 25 % done')
            
            if i == N/2:
                print('Resampling 50 % done')
            
            if i == (3*N)/4:
                print('Resampling 75 % done')
            
            self.re_bin_heights, self.re_bin_vals = np.histogram(func(), bins=self.bin_vals)
            
            self.v_phis[i] = self.re_bin_heights

            s+=self.re_bin_heights
            
        s = s/N
            
        for i in range(N_bins):
            
            for j in range(N):
                
                var[i] += (self.v_phis[j][i] - s[i])**2
            
        self.mean_sample = s
    
        st_dev = sqrt(var/N)
        
        self.st_dev = st_dev
        
        return self.st_dev
    
    def model_vel(self):
    
        dip=True
        dip_lim=15
        wthin = 0.75
        wthick=0.2
        whalo = 1.-wthin-wthick
        
        thin0 = np.array([0,-215,0])
        thick0 = np.array([0,-180,0])
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
                if velocity[1]<=0:
                    velocity[1]-=dip_lim
                if velocity[1]>=0:
                    velocity[1]+=dip_lim               
                
            vel_tot[j] = velocity
        
        vel_tot = vel_tot.transpose()
        vel_tot[1]=vel_tot[1]/self.gc.rho.to(u.km).value
        
        cyl_diff = coord.CylindricalDifferential(d_rho=vel_tot[0]*u.km/u.s,d_phi=vel_tot[1]/u.s,d_z=vel_tot[2]*u.km/u.s)
        
        self.gc_res = coord.Galactocentric(representation = coord.CylindricalRepresentation, rho=self.gc.rho,
                                           phi=self.gc.phi,z=self.gc.z,d_rho=cyl_diff.d_rho.to((u.mas*u.pc)/(u.yr*u.rad),equivalencies=u.dimensionless_angles()),
                                           d_phi=cyl_diff.d_phi.to(u.mas/u.yr,equivalencies=u.dimensionless_angles()),
                                           d_z=cyl_diff.d_z.to((u.mas*u.pc)/(u.yr*u.rad),equivalencies=u.dimensionless_angles()),
                                           galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun, differential_cls=coord.CylindricalDifferential)
        
        self.icrs_res = self.gc_res.transform_to(coord.ICRS)
        self.icrs_res.set_representation_cls(coord.SphericalRepresentation,s=coord.SphericalCosLatDifferential)
        
        err = self.e_sample*np.random.randn(self.e_sample.shape[0],self.e_sample.shape[1])
        
        self.resample = array([self.icrs_res.ra,self.icrs_res.dec,self.icrs_res.distance,self.icrs_res.pm_ra_cosdec,self.icrs_res.pm_dec,self.icrs_res.radial_velocity]) + err[:-2]
                
        self.icrs_res = coord.ICRS(ra = self.resample[0]*u.degree,dec = self.resample[1]*u.degree,
                            distance=self.resample[2]*u.pc,
                            pm_ra_cosdec=self.resample[3]*u.mas/u.yr,
                            pm_dec=self.resample[4]*u.mas/u.yr,
                            radial_velocity=self.resample[5]*u.km/u.s)
        
        self.gc_res = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
        self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (self.gc_res.d_phi*self.gc.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        return self.res_v_phi
    
    def get_ang_mom(self, halo=False):
        
        if halo==True:
            
            self.ang_mom = self.gc_res.rho.to(u.kpc)*self.res_v_phi
            
            return self.ang_mom
        
        self.ang_mom = self.gc.rho.to(u.kpc)*self.v_phi
        
        return self.ang_mom
        
    def get_energy(self,halo=False):
        
        """To do: Introduce cuts of metallicity and distance from GC to obtain set of halo stars"""
        
        v_halo = 173.2*(u.km/u.s)
        d_halo = 12*u.kpc
        
        a_d = 6.5*u.kpc
        b_d = 0.26*u.kpc
        M_disc = 6.3e10*u.Msun
        
        M_bulge = 2.1e10*u.Msun
        c_b = 0.7*u.kpc
        
        if halo==True:
            
            rho = self.gc_res.rho.to(u.kpc)
            
            z = self.gc_res.z.to(u.kpc)
            
            v_phi = (self.gc_res.d_phi*self.gc_res.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
            
            v_rho = self.gc_res.d_rho.to(u.km/u.s)
            
            v_z = self.gc_res.d_z.to(u.km/u.s)
            
        else:
            
            rho = self.gc.rho.to(u.kpc)
            
            z = self.gc.z.to(u.kpc)
            
            v_phi = self.v_phi
            
            v_rho = self.v_rho
            
            v_z = self.v_z
            
        r = np.sqrt(rho**2+z**2)
        
        E_halo = v_halo**2*np.log(1+rho**2/d_halo**2+z**2/d_halo**2)
        
        E_disc = - (G*M_disc)/np.sqrt(rho**2+(a_d+np.sqrt(z**2+b_d**2))**2)
    
        E_bulge = - (G*M_bulge)/(r+c_b)
        
        E_kin = (v_rho**2+v_phi**2+v_z**2)/2
    
        self.energy = E_halo+E_disc+E_bulge+E_kin
        
        return self.energy
    
    def get_halo(self):
  
        halo_dist = 100
          
        for i in range(len(self.sample_tp)):
        
            if self.met[i].value<=-1.5 and smp.icrs.distance[i].value>=halo_dist:
                try:
                    halo
                except NameError:
                    halo = self.sample_tp[i]
                    pass
                halo = np.vstack((halo,self.sample_tp[i]))  
       
        self.halo = halo.transpose()
        
        self.icrs_res=coord.ICRS(ra = self.halo[0]*u.degree,dec = self.halo[1]*u.degree,
                            distance = self.halo[2]*u.pc,
                            pm_ra_cosdec = self.halo[3]*u.mas/u.yr,
                            pm_dec = self.halo[4]*u.mas/u.yr,
                            radial_velocity = self.halo[5]*u.km/u.s)

        self.gc_res = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
        self.gc_res.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (self.gc_res.d_phi*self.gc_res.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        return
    
    def plot_sample(self,lim,binwidth):

        plt.figure()
#        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$')
        plt.xlabel('$v_\phi\ \ [\mathrm{km\ s}^{-1}$]')
        
        self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=np.arange(min(self.v_phi.value),max(self.v_phi.value)+binwidth,binwidth))
        plt.bar(self.bin_vals[:-1], self.bin_heights, width=np.diff(self.bin_vals),color='none',edgecolor='blue', log=True,label='$TGAS\ & \ RAVE\ data$')
        
        plt.legend()
        return 
    
    def plot_resamples(self, N, method, lim, binwidth ):
        
        
        if binwidth == None:
            binwidth = 15
        
        plt.figure()
#        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font)
        plt.xlabel('$v_\phi\ \ [\mathrm{km\ s}^{-1}$]', fontdict=font)

        if method in ('random','rand'):
                
            func = self.bootstrap_rand
            
            plt.hist(self.v_phi.value, bins=np.arange(min(self.res_v_phi.value),max(self.res_v_phi.value)+binwidth,binwidth), log=True, range=(-lim,lim),histtype='step',label='$TGAS\ & \ RAVE\ data$')
            plt.legend()
            
        elif method in ('model','mod'):
            
            func = self.model_vel
			
        else:
                
            func = self.bootstrap_err
            plt.hist(self.v_phi.value, bins=np.arange(min(self.res_v_phi.value),max(self.res_v_phi.value)+binwidth,binwidth), log=True, range=(-lim,lim),histtype='step',label='$TGAS\ & \ RAVE\ data$')
            plt.legend()

        for i in range(N):
    
            func()
        
            plt.hist(self.res_v_phi, bins=np.arange(min(self.res_v_phi.value),max(self.res_v_phi.value)+binwidth,binwidth), log=True, range=(-lim,lim),histtype='step')

        return

    def plot_mean(self, lim, binwidth=None, err=False, ymax=None,ymin=None, model=False):        

        if any(self.mean_sample) == None:
            raise Exception('You need to compute a mean using your method of choice before plotting')

        if any(self.bin_vals) == None:
            self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=np.arange(min(self.v_phi.value),max(self.v_phi.value)+binwidth,binwidth))

        if ymax == None:
            ymax = 100000
        
        if ymin == None:
            ymin = 1
    
        if err is True:
            err = self.st_dev
            
        plt.figure()
#        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$')
        plt.ylabel('$\mathrm{Number\ of\ stars}$')
        plt.xlabel('$v_\phi\ \ [\mathrm{km\ s}^{-1}$]', fontdict=font)
        plt.xlim(-lim,lim)
        plt.ylim(ymin,ymax)
        
        if model == True:
            
            plt.bar(self.bin_vals[:-1], self.mean_sample,width=np.diff(self.bin_vals),color='none', log=True,label='Mean of modeled $v_\phi$',edgecolor='red')
            
            plt.errorbar(self.bin_vals[:-1], self.mean_sample, yerr=err, fmt='none',ecolor='black', elinewidth=min(np.diff(self.bin_vals))/6 )


            plt.legend()
            
            return
        
        ####Change labels for the legends to include binwidth#####
        
        plt.bar(self.bin_vals[:-1], self.bin_heights, width=np.diff(self.bin_vals),color='none',edgecolor='blue', log=True,label='TGAS & RAVE data')
        plt.bar(self.bin_vals[:-1], self.mean_sample, width=np.diff(self.bin_vals),color='none', log=True,label='Mean of bootstrap samples',edgecolor='orange')

        plt.errorbar(self.bin_vals[:-1], self.mean_sample, yerr = err, fmt = 'none', ecolor = 'black', elinewidth=min(np.diff(self.bin_vals))/6)

        plt.legend()

        return
    
    def plot_E_L(self, halo=False):
        
        if halo==True:
            
            self.get_halo()
            
            ang_mom = self.get_ang_mom(halo=True)
            
            energy = self.get_energy(halo=True)*10**(-5)
            
        else:
            
            ang_mom = self.get_ang_mom()
            
            energy = self.get_energy()*10**(-5)
        
        plt.figure()
        plt.xlabel('$L_z$ [km s$^{-1}$ kpc]')
        plt.ylabel('Energy [$10^5$ km$^2$ s$^{-2}$]')
        plt.scatter(ang_mom, energy, s=2, c='none', edgecolors='blue')
        
    def save_mean_data(self,filename):
        
        return ascii.write([self.mean_sample,self.st_dev], filename+'.txt',names=['v_phi','sigma'])


#################### Tests ########################

#cProfile.run('smp.bootstrap_err(e_my_sample)')

        
smp = MW_dyn(my_sample,e_my_sample,my_data_order)

#cProfile.run('smp.get_st_dev(100,10,str(rand))')

"""1. Making a histogram with 100 bins of original sample and 10 resamples using bootstrap_rand"""

#smp.plot_resamples(10, 'random', 14, N_bins=50)

"""2. Making a histogram with 100 bins of original sample and 5 resamples using bootstrap_err. Takes ~30 s!"""

#smp.plot_resamples(5, 'error', 14, N_bins=100)

#halo_dist = 100
#  
#for i in range(len(smp.sample_tp)):
#
#    try:
#        if float(Fe[i])<=-1.5:# and smp.icrs.distance[i].value>=halo_dist:
#            try:
#                halo
#            except NameError:
#                halo = smp.sample_tp[i]
#                pass
#            halo = np.vstack((halo,smp.sample_tp[i]))  
#    except ValueError:
#            continue
#
#halo = halo.transpose()