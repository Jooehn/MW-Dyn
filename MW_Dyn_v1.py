from scipy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
import random
from astropy.io import ascii
from astropy.table import Table
from astropy.constants import G
from scipy.stats import binned_statistic_2d as b_stat2d
from matplotlib.ticker import AutoMinorLocator

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


def load_data(file,flagset):
    
    data_raw = ascii.read(file, format='fast_csv')
    
    data = Table(data_raw, copy=True)
    
    bad_rows=[]
    
    if flagset == 'flag_dup':
        
        flag_list=['flag_dup']
        
    elif flagset == 'flag_any':
        
        flag_list=['flag_any']

    elif flagset == 'flag_any-lowlogg':
        
        flag_list=['flag_dup','flag_N','flag_outlier','flag_pole']
        
    elif flagset == 'None':
        
        flag_list = []
        
    elif flagset == 'custom':
        
        flag_list=[str(input('Tell me which flags to remove: '))]

    else:
        raise Exception('Not a valid flagset')
    
    for i in flag_list:
        
        for j in range(len(data[i])):
            
            if data[i][j]==1 and j not in bad_rows:
                bad_rows.append(j)
                
    data.remove_rows(bad_rows)
    
    return data


flagset = 'flag_dup'

try:
    data
except NameError:
    
    data = load_data('Distances_PJM2017.csv',flagset)

RA = data['RAdeg']*u.degree
DEC = data['DEdeg']*u.degree
dist = data['distance']*u.pc
pm_RA = data['pmRA_TGAS']*u.mas/u.yr
pm_DEC = data['pmDE_TGAS']*u.mas/u.yr
rad_vel = data['HRV']*u.km/u.s
met = data['Met_N_K']*u.dex

e_RA = np.zeros(len(data))*u.degree
e_DEC = np.zeros(len(data))*u.degree
e_dist = data['edistance']*u.pc
e_pm_RA = data['pmRA_error_TGAS']*u.mas/u.yr
e_pm_DEC = data['pmDE_error_TGAS']*u.mas/u.yr
e_rad_vel = data['eHRV']*u.km/u.s
e_met = data['eMet_K']*u.dex

#e_dist = np.zeros(len(data))*u.pc
#e_pm_RA = np.zeros(len(data))*u.mas/u.yr
#e_pm_DEC = np.zeros(len(data))*u.mas/u.yr
#e_rad_vel = np.zeros(len(data))*u.km/u.s
#e_met = np.zeros(len(data))*u.dex

my_data_order=['RA', 'DEC', 'dist', 'pm_RA', 'pm_DEC', 'rad_vel','metallicity']

my_sample = np.array([RA, DEC, dist, pm_RA, pm_DEC, rad_vel, met])

e_my_sample = np.array([e_RA, e_DEC, e_dist, e_pm_RA, e_pm_DEC, e_rad_vel, e_met])

class MW_dyn:
    
    
    """The MW_dyn class which holds a sample of Milky Way stars and can perform statistical tests. 
    
    Takes the args:
        
        sample: The data sample at hand. Needs to be converted from Tablet to a set of arrays
        e_sample: Errors in the sample quantities if any. Required to use Bootstrap.error_sampling. Should be same shape as sample
        data_order: Just a conventient way to see the order of quantities in sample. Not needed to initialise
        
    Functions:
        
        error_sampling: implements bootstrapping for uncertainties. Returns a resampling of the original sample
        bootstrap: creates a random resample of angular velocities from the original sample
        get_st_dev: computes the standard deviation for N resamples in every bin for either the 'error', 'rand' or 'model' method
        model_vel: creates a pseudosample from a velocity model for each coordinate in self.sample, 
                then adds a random uncertainty within given range for each coordinate.
        plot_sample: plots the original sample in a histogram
        plot_resample: plots N resamples computed using a given method within a given v_phi limit
        plot_mean: plots the computed mean that is stored in self.mean_sample
        save_mean_data: saves the current data stored in self.mean_sample to an ascii file
    """
    
    def __init__(self,sample,flagset,e_sample=None,data_order=None):
        
        self.flagset = flagset
        
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
       
        
        self.ang_mom = np.zeros(len(self.sample))
        
        self.energy = np.zeros(len(self.sample))
        
        self.met = self.sample[6]*u.dex
        
        self.re_met = None

        self.halo = None
        
        self.e_halo = None
        
        self.halo_gc = None
        
        self.halo_gc_res = None
        
        self.disc = None
        
        self.e_disc = None

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
        
        self.dip_limit = None
        
        self.method = None
        
        self.scale_dist = None
        
    def error_sampling(self,halo=False,scale_dist=None):
        
        if any(self.e_sample) == None:
            raise Exception('Uncertainties are needed to perform this action')
            
        if halo == True:
            
            if scale_dist==None:
                
                self.scale_dist = 1
                
            else:
            
                self.scale_dist = scale_dist
            
            sample = self.halo
            
            error_data = self.e_halo
                
        else:
    
            self.scale_dist = 1
            
            sample = self.sample

            error_data = self.e_sample
            
        err = error_data*np.random.randn(error_data.shape[0],error_data.shape[1])
        
        self.resample = sample + err
                
        self.icrs_res=coord.ICRS(ra = self.resample[0]*u.degree,dec = self.resample[1]*u.degree,
                            distance=self.scale_dist*self.resample[2]*u.pc,
                            pm_ra_cosdec=self.resample[3]*u.mas/u.yr,
                            pm_dec=self.resample[4]*u.mas/u.yr,
                            radial_velocity=self.resample[5]*u.km/u.s)

        
        new_frame = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
        new_frame.set_representation_cls(coord.CylindricalRepresentation)
        
        if halo == True:
            
            self.halo_gc_res = new_frame
            
        else:
            
            self.gc_res = new_frame
            
        self.res_v_phi = (new_frame.d_phi*new_frame.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())

                       
        return self.res_v_phi
    
    def bootstrap(self):
        
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
    
    def get_st_dev(self, binwidth, N, method, dip = 0):
        
        self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=np.arange(min(self.v_phi.value),max(self.v_phi.value)+binwidth,binwidth))
        
        N_bins = len(self.bin_vals[:-1])
        
        self.v_phis = np.zeros([N,N_bins])
        
        s = np.zeros(N_bins)
        
        var = np.zeros(N_bins)
        
        self.method = method
        
        self.dip_limit = dip
        
        if method in ('error','err'):
            
            func = self.error_sampling
        
        elif method in ('bootstrap','boot'):
            
            func = self.bootstrap
        
        elif method in ('model','mod'):
            
            func = self.model_vel
            
        else:
            raise Exception('Not a valid method')
        
            
        for i in range(N):
            
            if i == N/4:
                print('Resampling 25 % done')
            
            if i == N/2:
                print('Resampling 50 % done')
            
            if i == (3*N)/4:
                print('Resampling 75 % done')
                
            if method in ('model'):
                
                self.re_bin_heights, self.re_bin_vals = np.histogram(func(str(dip)), bins=self.bin_vals)
                
            else:
                
                self.re_bin_heights, self.re_bin_vals = np.histogram(func(), bins=self.bin_vals)
            
            self.v_phis[i] = self.re_bin_heights

            s+=self.re_bin_heights
            
        print('Resampling 100 % done')
            
        s = s/N

        for i in range(N_bins):
            
            for j in range(N):
                
                var[i] += (self.v_phis[j][i] - s[i])**2
            
        self.mean_sample = s
    
        st_dev = sqrt(var/N)
        
        self.st_dev = st_dev
        
        return self.st_dev
    
    def model_vel(self,dip_lim='', halo=False,scale_dist=None):
            
        try:
            if len(dip_lim)==0:
    
                dip_lim = raw_input('What dip limit do you want? ')
            dip_lim = int(dip_lim)
        except TypeError:
            pass
            
        wthin = 0.91
        wthick= 0.08
        whalo = 1.-wthin-wthick
        
        thin0 = np.array([0,-215,0])
        thick0 = np.array([0,-180,0])
        halo0 = np.array([0,0,0])
        
        thin_disp  = np.array([30,20,17])
        thick_disp  = np.array([80,60,55])
        halo_disp = np.array([160,100,100])
        
        which = np.random.random_sample(len(self.gc))
        
        if halo == True:
            
            if scale_dist==None:
                
                self.scale_dist = 1
                
            else:
            
                self.scale_dist = scale_dist
            
            frame = self.halo_gc
            
            metallicity = self.halo[-1]
            
            vel_tot = np.zeros([len(self.halo_gc),3])
            
            error_data = self.e_halo
            
            for j in range(len(smp.halo_gc)):
                
                velocity = halo0 + np.random.randn(3)*halo_disp
                vel_tot[j] += velocity
            
#            vel_tot = halo0 + np.random.randn(vel_tot.shape[0],vel_tot.shape[1])*halo_disp
                
        else:
    
            self.scale_dist = 1
            
            frame = self.gc
            
            metallicity = self.met
            
            vel_tot=np.zeros([len(self.gc),3])
            
            error_data = self.e_sample
            
            for j in range(len(self.gc)):
        
                if which[j] < wthin :
                    velocity = thin0 + np.random.randn(3)*thin_disp
                elif which[j] < wthin+wthick :
                    velocity = thick0 + np.random.randn(3)*thick_disp
                else :
                    velocity = halo0 + np.random.randn(3)*halo_disp
    
                if dip_lim not in (0,None) and abs(velocity[1])<=dip_lim:
                    if velocity[1]<=0:
                        velocity[1]-=dip_lim
                    if velocity[1]>=0:
                        velocity[1]+=dip_lim
                 
                vel_tot[j] = velocity
        
        vel_tot = vel_tot.transpose()
        vel_tot[1]=vel_tot[1]/frame.rho.to(u.km).value
        
        cyl_diff = coord.CylindricalDifferential(d_rho=vel_tot[0]*u.km/u.s,d_phi=vel_tot[1]/u.s,d_z=vel_tot[2]*u.km/u.s)
        
        new_frame = coord.Galactocentric(representation = coord.CylindricalRepresentation, rho=frame.rho,
                                           phi=frame.phi,z=frame.z,d_rho=cyl_diff.d_rho.to((u.mas*u.pc)/(u.yr*u.rad),equivalencies=u.dimensionless_angles()),
                                           d_phi=cyl_diff.d_phi.to(u.mas/u.yr,equivalencies=u.dimensionless_angles()),
                                           d_z=cyl_diff.d_z.to((u.mas*u.pc)/(u.yr*u.rad),equivalencies=u.dimensionless_angles()),
                                           galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun, differential_type=coord.CylindricalDifferential)
        
        self.icrs_res = new_frame.transform_to(coord.ICRS)
        self.icrs_res.set_representation_cls(coord.SphericalRepresentation,s=coord.SphericalCosLatDifferential)
        
        err = error_data*np.random.randn(error_data.shape[0],error_data.shape[1])
            
        self.resample = array([self.icrs_res.ra,self.icrs_res.dec,self.scale_dist*self.icrs_res.distance,self.icrs_res.pm_ra_cosdec,self.icrs_res.pm_dec,self.icrs_res.radial_velocity,metallicity]) + err

        self.re_met = self.resample[6]*u.dex
            
        self.icrs_res = coord.ICRS(ra = self.resample[0]*u.degree,dec = self.resample[1]*u.degree,
                            distance=self.resample[2]*u.pc,
                            pm_ra_cosdec=self.resample[3]*u.mas/u.yr,
                            pm_dec=self.resample[4]*u.mas/u.yr,
                            radial_velocity=self.resample[5]*u.km/u.s)
        
        new_frame = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
        new_frame.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (new_frame.d_phi*new_frame.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        if halo == True:
            
            self.halo_gc_res = new_frame
            
        else:
            
            self.gc_res = new_frame
        
#        return self.res_v_phi.value
            return vel_tot
    
    def get_ang_mom(self, halo=False):
        
        if halo==True:
            
            self.ang_mom = -self.halo_gc_res.rho.to(u.kpc)*self.res_v_phi
            
            return self.ang_mom
        
        self.ang_mom = -self.gc.rho.to(u.kpc)*self.v_phi
        
        return self.ang_mom
        
    def get_energy(self,halo=False):
        
        v_halo = 173.2*(u.km/u.s)
        d_halo = 12*u.kpc
        
        a_d = 6.5*u.kpc
        b_d = 0.26*u.kpc
        M_disc = 6.3e10*u.Msun
        
        M_bulge = 2.1e10*u.Msun
        c_b = 0.7*u.kpc
        
        if halo==True:
            
            rho = self.halo_gc_res.rho.to(u.kpc)
            
            z = self.halo_gc_res.z.to(u.kpc)
            
            v_phi = (self.halo_gc_res.d_phi*self.halo_gc_res.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
            
            v_rho = self.halo_gc_res.d_rho.to(u.km/u.s)
            
            v_z = self.halo_gc_res.d_z.to(u.km/u.s)
            
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
    
        self.energy = (E_halo+E_disc+E_bulge+E_kin-1.7e5*(u.km**2/u.s**2))*1e-5
        
        return self.energy
    
    def get_halo(self):
            
        input_data = self.sample_tp
        
        e_input_data = self.e_sample.transpose()
        
        met = self.met
        
        dist = self.icrs.distance
  
        halo_dist = 100
         
        for i in range(len(input_data)):
        
            if met[i].value<=-1.5 and dist[i].value>=halo_dist:
                try:
                    halo
                    e_halo
                except NameError:
                    halo = input_data[i]
                    e_halo = e_input_data[i]
                    pass
                halo = np.vstack((halo,input_data[i]))  
                e_halo = np.vstack((e_halo,e_input_data[i]))
       
        self.halo = halo.transpose()
        
        self.e_halo = e_halo.transpose()
        
        self.icrs_res=coord.ICRS(ra = self.halo[0]*u.degree,dec = self.halo[1]*u.degree,
                            distance = self.halo[2]*u.pc,
                            pm_ra_cosdec = self.halo[3]*u.mas/u.yr,
                            pm_dec = self.halo[4]*u.mas/u.yr,
                            radial_velocity = self.halo[5]*u.km/u.s)

        self.halo_gc = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
        self.halo_gc.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (self.halo_gc.d_phi*self.halo_gc.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        return
    
    def get_halo_cut(self,N,E_cut,model=False,scale_dist=None):
        
        l_cut_av = 0
        
        l_cut = np.zeros(N)
        
        var = 0
        
        N_halo = 0
        
        self.get_halo()

        for k in range(N):
            
            left = 0
            
            if model == True:
            
                self.model_vel(dip_lim = None, halo=True,scale_dist=scale_dist)
                
            else:
                
                self.error_sampling(halo=True)
            
            ang_mom = self.get_ang_mom(halo=True).value
            
            energy = self.get_energy(halo=True).value
            
            low_E = np.where(energy<E_cut)
                    
            energy = np.delete(energy,low_E)
                    
            ang_mom = np.delete(ang_mom,low_E)
            
            left = len(np.where(ang_mom < 0)[0])
                    
            l_cut[k] = left / len(ang_mom)
            
            l_cut_av += left / len(ang_mom)
            
            N_halo += len(ang_mom)
          
        l_cut_av = l_cut_av / N  
        
        N_halo = N_halo / N
            
        for i in range(N):
        
            var += (l_cut[i] - l_cut_av)**2
    
        st_dev = np.round(sqrt(var/N),decimals=3)
            
        neg_L = np.round(l_cut_av,decimals=3)
        
        pos_L = 1-neg_L
        
        return neg_L,pos_L,st_dev,N_halo
    
    def get_disc(self,model=False,scale_dist=None):
        
        frame = self.gc
        
        input_data = self.sample_tp
        
        e_input_data = self.e_sample.transpose()
        
        if model == True:
            
            self.model_vel(dip_lim=None,scale_dist=scale_dist)
            
            input_data = self.resample.T
            
            frame = self.gc_res
        
        z = frame.z.to(u.kpc).value
        rho = frame.rho.to(u.kpc).value
        
        z_cond1 = np.where(z<-2)
        z_cond2 = np.where(2<z)
        z_cond = np.concatenate([z_cond1[0],z_cond2[0]])
        
        rho_cond1 = np.where(rho<5)
        rho_cond2 = np.where(10<rho)
        rho_cond = np.concatenate([rho_cond1[0],rho_cond2[0]])
        
        not_disc = np.union1d(z_cond,rho_cond)
        
        disc = np.delete(input_data,not_disc,0)

        e_disc = np.delete(e_input_data,not_disc,0)    
        
        self.disc = disc.transpose()
        
        self.e_disc = e_disc.transpose()
        
        self.icrs_res=coord.ICRS(ra = self.disc[0]*u.degree,dec = self.disc[1]*u.degree,
                            distance = self.disc[2]*u.pc,
                            pm_ra_cosdec = self.disc[3]*u.mas/u.yr,
                            pm_dec = self.disc[4]*u.mas/u.yr,
                            radial_velocity = self.disc[5]*u.km/u.s)

        self.disc_gc = self.icrs_res.transform_to(coord.Galactocentric(galcen_distance = gc_sun_dist, galcen_v_sun=v_sun, z_sun=gp_z_sun))
        self.disc_gc.set_representation_cls(coord.CylindricalRepresentation)
        
        self.res_v_phi = (self.disc_gc.d_phi*self.disc_gc.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
        
        return
                    
    
    def plot_sample(self,lim,binwidth):

        plt.figure()
#        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontsize = 'xx-large')
        plt.xlabel('$v_\phi\ \ [\mathrm{km\ s}^{-1}$]', fontsize = 'xx-large')
        
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.xlim(-lim,lim)
        plt.ylim(1,100000)
        self.bin_heights, self.bin_vals = np.histogram(self.v_phi, bins=np.arange(min(self.v_phi.value),max(self.v_phi.value)+binwidth,binwidth))
        plt.bar(self.bin_vals[:-1], self.bin_heights, width=np.diff(self.bin_vals),color='none',edgecolor='blue', log=True,label='$TGAS\ & \ RAVE\ data$')
        plt.tight_layout()
        plt.legend(fontsize='x-large')
        return 

    
    def plot_vel_field(self,comp,model=False,scale_dist=None):
        
        if model == True:
            N=100
        else:
            N=1

        
        binwidth = 0.2
        z_bins = np.arange(-2,2+binwidth,binwidth)
        rho_bins = np.arange(5,10+binwidth,binwidth)
        
        for i in range(N):
        
            self.get_disc(model)
                
            z = self.disc_gc.z.to(u.kpc)
            rho = self.disc_gc.rho.to(u.kpc)
            
            if comp == 'z':
                v = self.disc_gc.d_z.to(u.km/u.s).value
                v_string = '$v_z$'
                vmin = -20
                vmax = 20
            elif comp == 'rho':
                v = self.disc_gc.d_rho.to(u.km/u.s).value
                v_string = r'$v_\rho$'
                vmin = -20
                vmax = 20
            elif comp == 'phi':
                v = (self.disc_gc.d_phi*self.disc_gc.rho.to(u.kpc)).to(u.km/u.s,equivalencies =u.dimensionless_angles())
                v_string = r'$v_\phi$'
                vmin=-150
                vmax=230
    
            b_vel = b_stat2d(rho,z,v,statistic='median',bins=[rho_bins,z_bins])
            
            try:
                v_med
            except NameError:
                v_med = np.zeros(b_vel.statistic.shape)
                counts = np.zeros(b_vel.statistic.shape)
                var = b_vel.statistic
                pass
            
            v_med += b_vel.statistic
            
            b_vel_c = b_stat2d(rho,z,v,statistic='count',bins=[rho_bins,z_bins])
            
            counts += b_vel_c.statistic
            
            if i == 1:
                continue
                           
            var = np.dstack((var,b_vel.statistic))
       
        v_med = v_med / N
        counts = counts / N
        
        st_dev = np.nanstd(var,axis=2,ddof=1)
        
        st_dev[counts < 50] = np.nan
        v_med[counts < 50] = np.nan
        
        zc = (b_vel.y_edge[:-1] + b_vel.y_edge[1:]) / 2
        rhoc = (b_vel.x_edge[:-1] + b_vel.x_edge[1:]) / 2
        
        extent = [b_vel.x_edge[0], b_vel.x_edge[-1],b_vel.y_edge[0],b_vel.y_edge[-1]]
        
#        plt.figure()
#        ax = plt.axes()
#        if model == True:
#            plt.title('Mean modelled median '+'{}'.format(v_string)+' for '+'{}'.format(self.flagset),size='xx-large')
#        else:     
#            plt.title('Median '+'{}'.format(v_string)+' for '+'{}'.format(self.flagset),size='xx-large')
#        plt.text(2,2,s='{}'.format(self.flagset))
#        plt.xlabel(r'$\rho\ \mathrm{[kpc]}$',size='xx-large')
#        plt.ylabel('$z\ \mathrm{[kpc]}$',size='xx-large')
#        plt.xticks(size='x-large')
#        plt.yticks(size='x-large')
##        plt.axis([rhoc.min()-0.5,rhoc.max()+0.5,zc.min()-0.5,zc.max()+0.5])
#        plt.axis([8,9.5,-1.5,0])
#
#        ax.minorticks_on()   
##        ax.set_yticks(np.arange(-2,2+1,1))
##        ax.set_xticks(np.arange(5,10+1,1))   
#        ax.set_yticks(np.arange(-1.5,0.5,0.5))
#        ax.set_xticks(np.arange(8,10,0.5)) 
#        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
#        ax.grid(True,which='major',linestyle=':',lw=1.2)
#        ax.grid(True,which='minor',linestyle=':',lw=1.2)
#        
##        plt.contourf(rhoc,zc,v_med.T,vmin=vmin,vmax=vmax,cmap = plt.cm.get_cmap('jet'))
##        
#        plt.imshow(v_med.T,origin='lower',interpolation='bilinear',vmin=vmin,vmax=vmax,cmap = plt.cm.get_cmap('jet'),extent=extent)
#        cb = plt.colorbar(orientation='vertical', extend='both')
#        cb.set_label('km s$^{-1}$',size='x-large')
#        plt.tight_layout()
        
        return v_med, st_dev, zc, rhoc, extent, v_string
    
    def plot_resamples(self, N, method, lim, binwidth ):
        
        
        if binwidth == None:
            binwidth = 15
        
        plt.figure()
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
#        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$',fontdict=font)
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontdict=font, fontsize = 'xx-large')
        plt.xlabel('$v_\phi\ \ [\mathrm{km\ s}^{-1}$]', fontdict=font, fontsize='xx-large')

        if method in ('bootstrap','boot'):
                
            func = self.bootstrap
            
            plt.hist(self.v_phi.value, bins=np.arange(min(self.res_v_phi.value),max(self.res_v_phi.value)+binwidth,binwidth), log=True, range=(-lim,lim),histtype='step',label='$TGAS\ & \ RAVE\ data$')
            plt.legend()
            
        elif method in ('model','mod'):
            
            func = self.model_vel

			
        else:
                
            func = self.error_sampling
            plt.hist(self.v_phi.value, bins=np.arange(min(self.res_v_phi.value),max(self.res_v_phi.value)+binwidth,binwidth), log=True, range=(-lim,lim),histtype='step',label='$TGAS\ & \ RAVE\ data$')
            plt.legend(fontsize='x-large')

        for i in range(N):
    
            func()
        
            plt.hist(self.res_v_phi, bins=np.arange(min(self.res_v_phi.value),max(self.res_v_phi.value)+binwidth,binwidth), log=True, range=(-lim,lim),histtype='step')

        return

    def plot_mean(self, lim, binwidth=None, err=False, ymax=None,ymin=None):        

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
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
#        plt.title('$\mathrm{Histogram\ of\ stars\ with\ a\ given\ angular\ velocity\ }v_\phi$')
        plt.ylabel('$\mathrm{Number\ of\ stars}$', fontsize='xx-large')
        plt.xlabel('$v_\phi\ \ [\mathrm{km\ s}^{-1}$]', fontdict=font, fontsize='xx-large')
        plt.xlim(-lim,lim)
        plt.ylim(ymin,ymax)
        
        if self.method in ('model'):
            
            plt.title('$\lambda = {}'.format(self.dip_limit)+'\ \mathrm{km\ s}^{-1}$',fontsize='x-large')
#            plt.bar(self.bin_vals[:-1], self.bin_heights, width=np.diff(self.bin_vals),color='grey',edgecolor='grey', log=True,label='TGAS & RAVE data',lw=1.6)
            plt.bar(self.bin_vals[:-1], self.mean_sample,width=np.diff(self.bin_vals),color='none', log=True,label='Mean of $v_\phi$ from model',edgecolor='red')
            
            plt.errorbar(self.bin_vals[:-1], self.mean_sample, yerr=err, fmt='none',ecolor='black', elinewidth=min(np.diff(self.bin_vals))/5 )
            
            plt.tight_layout()
            plt.legend(fontsize='x-large')
            
            return
        
        ####Change labels for the legends to include binwidth#####
        
        plt.bar(self.bin_vals[:-1], self.bin_heights, width=np.diff(self.bin_vals),color='grey',edgecolor='grey', log=True,label='TGAS & RAVE data',lw=1.6)
        plt.bar(self.bin_vals[:-1], self.mean_sample, width=np.diff(self.bin_vals),color='none', log=True,label='Mean of {} sampling'.format(self.method),edgecolor='b',lw=1.6)

        plt.errorbar(self.bin_vals[:-1], self.mean_sample, yerr = err, fmt = 'none', ecolor = 'black', elinewidth=min(np.diff(self.bin_vals))/5)
        plt.tight_layout()
        plt.legend(fontsize='x-large')

        return
    
    def plot_E_L(self, halo=False,model=False,scale_dist=None):
        
        plt.figure()
        
        if halo==True:
            
            self.get_halo()
            
            self.halo_gc_res = self.halo_gc
            
            if model==True:
                    
                self.model_vel(dip_lim = None, halo=True,scale_dist=scale_dist)
                
                if self.scale_dist == 1:
                    plt.title(r'$\mathrm{Model\ halo\ with\ default}\ \rho$',fontsize='x-large')
                    
                else:
                    
                    plt.title('$\mathrm{Model\ halo\ with\ }'+ '{}'.format(self.scale_dist)+r'\rho$',fontsize='x-large')
                
            ang_mom = self.get_ang_mom(halo)
            
            energy = self.get_energy(halo)
            
        else:
            
            ang_mom = self.get_ang_mom()
            
            energy = self.get_energy()
        

        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.xlabel('$L_z$ [km s$^{-1}$ kpc]',fontsize='xx-large')
        plt.ylabel('Energy [$10^5$ km$^2$ s$^{-2}$]',fontsize='xx-large')
#        plt.title(r'$\mathrm{Default\ sample}$',size='x-large')
        plt.xlim(-4500,4500)
        plt.ylim(-2.1,0.2)
        plt.scatter(ang_mom, energy, s=2, c='none', edgecolors='blue', label='{} halo subset'.format(self.flagset))
        plt.legend(fontsize='x-large')
        plt.tight_layout()
        
    def save_mean_data(self,filename):
        
        return ascii.write([self.mean_sample,self.st_dev], filename+'.txt',names=['v_phi','sigma'])

smp = MW_dyn(my_sample,flagset,e_my_sample,my_data_order)