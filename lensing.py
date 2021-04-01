# importing relevant modules
import numpy as np
import scipy as sp
from cosmo import CosmoCalc
import tools


#################################################################################################################################


# defining relevant constants
c = 3e8  # speed of light in m/s
G = 4.3*10**(-9)  # gravitational constant in Mpc*M_sun^-1*(km/s)^2


#################################################################################################################################


def lensing_distances(zl, zs):
    
    Dl = CosmoCalc().angular_diameter_distance(0, zl)  
    Ds = CosmoCalc().angular_diameter_distance(0, zs)  
    Dls = CosmoCalc().angular_diameter_distance(zl, zs)
    
    return Dl, Ds, Dls # in Mp  
      
    
def critical_surface_mass_density(zl, zs):
    
    Dl, Ds, Dls = lensing_distances(zl, zs)  
    sigma_c = (((c*1e-3)**2)/(4*np.pi*G))*((Ds)/(Dl*Dls))  
    
    return sigma_c # in M_sun/Mpc^2       
  
    
def deflection_from_convergence(mapparams, kappa_map):
    
    # defining underlying grid in harmonic space
    grid, _ = tools.make_grid(mapparams, Fourier = True)
    lX, lY = grid
    l2d = np.hypot(lX, lY)
    
    # computing deflection angle from convergence map
    kappa_map_fft = np.fft.fft2(kappa_map)
    alphaX_fft =  1j * lX * 2. *  kappa_map_fft / l2d**2
    alphaY_fft =  1j * lY * 2. *  kappa_map_fft / l2d**2
    alphaX_fft[np.isnan(alphaX_fft)] = 0
    alphaY_fft[np.isnan(alphaY_fft)] = 0
    alphaX = np.degrees(np.fft.ifft2(alphaX_fft).real)*60
    alphaY = np.degrees(np.fft.ifft2(alphaY_fft).real)*60
    alpha_vec = [alphaX, alphaY]
    
    return alpha_vec # in arcmin    
    
    
def lens_map(map_params, unlensed_map, alpha_vec):   
    
    # creating undeflected field
    grid, _ = tools.make_grid(map_params)  
    betaX, betaY = grid
   
    # cromputing deflected field    
    alphaX, alphaY = alpha_vec
    thetaX = betaX + alphaX
    thetaY = betaY + alphaY
    
    # computing lensed map through interpolation
    interpolate = sp.interpolate.RectBivariateSpline(betaY[:,0], betaX[0,:], unlensed_map, kx = 5, ky = 5)
    lensed_map  = interpolate.ev(thetaY.flatten(), thetaX.flatten()).reshape([len(betaY), len(betaX)]) 
    
    return lensed_map


#################################################################################################################################    

class NFW:    
    
    def __init__(self, M_200, c_200, z_l, z_s):
        
        self.M_200 = M_200
        self.c_200 = c_200
        self.z_l = z_l
        self.z_s = z_s
            
            
    def convergence_profile(self, theta):
        
        # computing cosmological parameters
        Dl, _, _ = lensing_distances(self.z_l, self.z_s)  
        rho_c = CosmoCalc().critical_density(self.z_l)    
    
    
        # computing nfw parameters 
        r_200 = ((3*self.M_200)/(4*np.pi*200*rho_c))**(1/3)   
        r_s = r_200/self.c_200  
        rho_s = (200 / 3) * rho_c * (self.c_200 ** 3 / (np.log(1 + self.c_200) - (self.c_200 / (1 + self.c_200))))  
    
    
        # computing x
        theta_rad = np.radians(theta/60)
        theta_s = r_s/Dl
        x = theta_rad/theta_s  
   

        # computing kappa_s
        sigma_c = critical_surface_mass_density(self.z_l, self.z_s)  
        kappa_s = rho_s*r_s/sigma_c
        
        
        # computing f
        x1 = np.where(x > 1)
        x2 = np.where(x == 1)
        x3 = np.where(x < 1)
        f = np.zeros(len(x))
        f[x1] = (1/np.sqrt(x[x1]**2-1))*np.arctan(np.sqrt(x[x1]**2-1))
        f[x2] = 1
        f[x3] = (1/np.sqrt(1-x[x3]**2))*np.arctanh(np.sqrt(1-x[x3]**2))
       
        
        # computing kappa
        kappa_profile = (2*kappa_s*(1-f)/(x**2-1)) 
        
        return kappa_profile
    
    
    def convergence_map(self, map_params, centroid_shift = None):
        
        # getting kappa_profile
        nx, dx, _, _ = map_params
        theta = np.linspace(0, nx*dx/2, nx)
        kappa_profile = self.convergence_profile(theta)
        
        # creating map grid 
        grid, _ = tools.make_grid(map_params)      
        
        # adding Gaussian positional uncertainties
        gridX, gridY = grid 
        if centroid_shift is not None:
            x_shift, y_shift = centroid_shift
            gridX += x_shift
            gridY += y_shift 
        grid = gridX, gridY
        
        # computing convergence map from convergence profile
        kappa_map = tools.convert_to_2d(grid, theta, kappa_profile)
        
        return kappa_map