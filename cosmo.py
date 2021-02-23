# importing relevant modules
import numpy as np
from scipy import integrate
import camb
import tools


#################################################################################################################################


# defining relevant constants
c = 3e8  # speed of light in m/s
G = 4.3*10**(-9)  # gravitational constant in Mpc*M_sun^-1*(km/s)^2


#################################################################################################################################


class CosmoCalc:
    # ...
    def __init__(self, h = 0.674, omega_r = 0, ombh2=0.0224, omch2=0.120, omega_k = 0):
        self.h = h
        self.omega_r = omega_r
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.omega_k = omega_k
        
       
    def hubble_parameter(self, z):
        H0 = 100*self.h
        omega_m = self.ombh2/self.h**2 + self.omch2/self.h**2
        omega_lambda = 1-self.omega_r-omega_m-self.omega_k
        H = H0 * np.sqrt(self.omega_r*(1+z)**4 + omega_m*(1+z)**3 + self.omega_k*(1+z)**2 + omega_lambda)
        return H  # in km/s/Mpc 
    

    def hubble_distance(self, z):
        H = self.hubble_parameter(z)
        DH = (c*1e-3)/H
        return DH  # in Mpc  
   
    
    def comoving_distance(self, z1, z2):
        def inv_H(z):
            H = self.hubble_parameter(z)
            return 1/H
        chi, _ = (c*1e-3) * np.array(integrate.quad(inv_H, z1, z2))  
        return chi  # in Mpc 


    def comoving_angular_diameter_distance(self, z1, z2): 
        DH = self.hubble_distance(0)  
        chi = self.comoving_distance(z1, z2)  
        if self.omega_k > 0: 
            fk = DH/np.sqrt(self.omega_k) * np.sinh(np.sqrt(self.omega_k)*chi/DH)  
        elif self.omega_k == 0: 
            fk = chi  
        else: 
            fk = DH/np.sqrt(abs(self.omega_k)) * np.sin(np.sqrt(abs(self.omega_k))*chi/DH)  
        return fk  # in Mpc 
    
         
    def angular_diameter_distance(self, z1, z2): 
        fk = self.comoving_angular_diameter_distance(z1, z2)  
        Dang = np.array(fk) / (1+z2) 
        return Dang  # in Mpc   

    
    def critical_density(self, z):
        H = self.hubble_parameter(z)  
        rho_c = (3*H**2)/(8*np.pi*G)  
        return rho_c  # in M_sun/Mpc^3
    
    
    def cmb_power_spectrum(self, tau = 0.054, TCMB = 2.72548, As = 2.10e-9, ns=0.965):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=100*self.h, ombh2=self.ombh2, omch2=self.omch2, omk=self.omega_k, tau = tau, TCMB = TCMB)
        pars.InitPower.set_params(As = As, ns=ns)
        pars.set_for_lmax(9949)
        results = camb.get_results(pars)
        powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
        dl = powers['lensed_scalar'][:,0]
        l = np.arange(dl.shape[0])
        cl = dl * 2 * np.pi / (l*(l+1))
        return l, cl