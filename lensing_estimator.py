# importing relevant modules
import numpy as np
from scipy import signal
import random
from tqdm import tqdm
import tools
from tqdm import tqdm
import lensing
import cosmo
import stats
import sims
import ilc


#################################################################################################################################


def get_aligned_cutout(map_params, image, image_noiseless = None, cutout_size_am = 10, cutout_size_for_grad_est_am = 6, l_cut = 2000, l = None, cl = None, cl_noise = None):  
     
    if image_noiseless is None:
        image_noiseless = np.copy(image)
        
    if cl_noise is None:
        l = np.arange(10000)
        cl = np.ones(max(l)+1)
        cl_noise = np.zeros(max(l)+1)
    
    _, dx, _, _ = map_params
    cutout = tools.central_cutout(map_params, image_noiseless, cutout_size_am)
    wiener_filter = tools.wiener_filter(l, cl, cl_noise)
    low_pass_filter = tools.low_pass_filter(l, l_cut)
    filtered_map = tools.convolve(image, l, wiener_filter * low_pass_filter, map_params) 
    filtered_cutout = tools.central_cutout(map_params, filtered_map, cutout_size_for_grad_est_am)
    _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
    angle, magnitude_weight = np.median(angle), np.median(magnitude) 
    cutout_aligned = tools.rotate(cutout, angle)
    cutout_aligned -= np.median(cutout_aligned)
    
    return cutout_aligned, magnitude_weight
    
    
def get_stack(cutouts, magnitude_weights = None, noise_weights = None):
    
    if magnitude_weights is None:
        magnitude_weights = np.ones(len(cutouts))
    if noise_weights is None:
        noise_weights = np.ones(len(cutouts))    
    weights = np.array(magnitude_weights)*np.array(noise_weights)
    weighted_cutouts = [cutouts[i]*weights[i] for i in range(len(cutouts))]
    stack = np.sum(weighted_cutouts, axis = 0)/sum(weights)
    
    return stack


def stack(map_params, maps, cutout_size_am = 10, cutout_size_for_grad_est_am = 6, l_cut = 2000, l = None, cl = None, cl_noise = None, use_magnitude_weights = True, noise_weights = None, correct_for_tsz = False):
     
    cutouts_arr = []
    magnitude_weights_arr = []
    for i in range(len(maps)):
        cutout, weight = get_aligned_cutout(map_params, maps[i], cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am =
                                            cutout_size_for_grad_est_am, l_cut = l_cut, l = l, cl = cl, cl_noise = cl_noise)
        cutouts_arr.append(cutout)
        magnitude_weights_arr.append(weight)
    if use_magnitude_weights is False:
        magnitude_weights_clus_arr = None
    stack = get_stack(cutouts_arr, magnitude_weights_arr, noise_weights) 
   
    if correct_for_tsz is True:
        cutouts_arr = []
        for i in range(len(maps)):
            cutout = tools.central_cutout(map_params, maps[i], cutout_size_am)
            cutout -= np.median(cutout)
            cutouts_arr.append(cutout)
        stack_tsz = get_stack(cutouts_arr, magnitude_weights_arr, noise_weights) 
        
        return stack, stack_tsz
    
    return stack
    
    
def lensing_dipole(map_params, stack_clus, stack_bg, stack_tsz = None, circular_disk_rad_am = None):
    
    if stack_tsz is None:
        stack_dipole = stack_clus-stack_bg
    else:
        stack_dipole = stack_clus-stack_bg-stack_tsz
    
    if circular_disk_rad_am is not None: 
        nx, dx, _, _ = map_params
        bins = np.arange((-40*dx)/2., (40*dx)/2., dx)+dx/2.
        
        ny_, nx_ = stack_clus.shape
        map_params_mod = [nx_, ny_, map_params[1]]
        pixels_to_mask = tools.get_pixels_to_mask(map_params_mod, circular_disk_rad_am)
      
        stack_dipole[pixels_to_mask] = None
        stack_dipole = np.ma.masked_invalid(stack_dipole)

        profile_lensing_dipole = np.ma.mean(stack_dipole, axis = 0)
        profile_lensing_dipole = np.ma.filled(profile_lensing_dipole, fill_value = 0.)
          
    else:
        nx, dx, _, _ = map_params
        bins = np.arange((-20*dx)/2., (20*dx)/2., dx)+dx/2.
        profile_lensing_dipole = np.mean(stack_dipole, axis = 0)            
    
    return bins, profile_lensing_dipole, stack_dipole
    
    
def covariance_and_correlation_matrix(nber_cov, nber_clus, map_params, l, cl, freq_arr = None, cluster_corr_cutouts = None, cl_extragal = None, bl = None, nl = None, opbeam = None, experiment = None, components = 'all', cutout_size_am = 10, l_cut  = 2000, cl_noise = None, use_magnitude_weights = True, noise_weights = None, correct_for_tsz = False):
 
    
    sims_for_covariance = []
    for i in tqdm(range(nber_cov)):
          
        if experiment is None:
            maps_clus = sims.cmb_mock_data(map_params, l, cl, cluster_corr_cutouts = cluster_corr_cutouts,
                                           cl_extragal = cl_extragal, bl = bl, nl = nl, nber_obs = nber_clus)
        else:
            maps_clus = sims.cmb_forecast_data(experiment = experiment, freq_arr = freq_arr, map_params = map_params, l = l, cl = cl, cluster_corr_cutouts = cluster_corr_cutouts, cl_residual = cl_noise, bl_arr = bl, opbeam = opbeam, nber_obs = nber_clus)
               
   
        if correct_for_tsz is False:
            stack_clus = stack(map_params, maps_clus, cutout_size_am = cutout_size_am, l_cut = l_cut, l = l, cl = cl, cl_noise = cl_noise, use_magnitude_weights = use_magnitude_weights, noise_weights = noise_weights,correct_for_tsz = correct_for_tsz) 
           
        else:
            stack_clus, stack_tsz = stack(map_params, maps_clus, cutout_size_am = cutout_size_am, l_cut = l_cut, l = l, cl = cl, cl_noise = cl_noise,use_magnitude_weights = use_magnitude_weights, noise_weights = noise_weights, correct_for_tsz = correct_for_tsz) 
            stack_clus -= stack_tsz


        profile_lensing_dipole = np.mean(stack_clus, axis = 0)   
        sims_for_covariance.append(profile_lensing_dipole) 
    
    
    covariance_matrix, correlation_matrix = stats.covariance_and_correlation_matrix(sims_for_covariance)
    
    return covariance_matrix, correlation_matrix


def model_profiles(nber_fit, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = None, cl_noise = None, cutout_size_am = 10, use_magnitude_weights = True, use_noise_weights = False, apply_noise = True, circular_disk_rad_am = None):
    
    nx, dx, ny, dy = map_params
    if cl_noise is None:
        cl_noise = np.zeros(max(l)+1)
    
    mass_int = np.copy(mass_int)*1e14
    
    cutouts_clus_arr = []
    magnitude_weights_clus_arr = []    
    for i in tqdm(range(nber_fit)):
        sim = sims.cmb_mock_data(map_params, l, cl)
        x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value), np.random.normal(loc=0.0, scale =
                                                                                                     centroid_shift_value) 
        centroid_shift = [x_shift, y_shift]
        for j in range(len(mass_int)):
            c200c = cosmo.concentration_parameter(mass_int[j], z, 0.674)
            kappa_map = lensing.NFW(mass_int[j], c200c, z, 1100).convergence_map(map_params, centroid_shift = centroid_shift)
            alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map)
            sim_lensed = lensing.lens_map(map_params, sim, alpha_vec)     
            sim_lensed_noise = np.copy(sim_lensed)
            total_noise_map = tools.make_gaussian_realization(map_params, l, cl_noise)
            sim_lensed_noise += total_noise_map
            if bl is not None:
                sim_lensed = tools.convolve(sim_lensed, l, np.sqrt(bl), map_params = map_params)
                sim_lensed_noise = tools.convolve(sim_lensed_noise, l, np.sqrt(bl), map_params = map_params)
            if apply_noise is False:
                sim_lensed_noise = np.copy(sim_lensed)
    
            cutout_aligned, magnitude_weight = get_aligned_cutout(map_params, sim_lensed_noise, image_noiseless = sim_lensed, cutout_size_am = cutout_size_am,  l = l, cl = cl, cl_noise = cl_noise)
            if use_magnitude_weights is False:
                magnitude_weight = 1
            cutouts_clus_arr.append(cutout_aligned*magnitude_weight)
            magnitude_weights_clus_arr.append(magnitude_weight)
    profile_models_arr = [] 
    stack_bg = np.sum(cutouts_clus_arr[0::len(mass_int)], axis = 0)/np.sum(magnitude_weights_clus_arr[0::len(mass_int)])
    for i in tqdm(range(len(mass_int))):
        stack_clus = np.sum(cutouts_clus_arr[i::len(mass_int)], axis = 0)/np.sum(magnitude_weights_clus_arr[i::len(mass_int)])
        stack_dipole = stack_clus-stack_bg
        
        
        if circular_disk_rad_am is not None: 
            ny_, nx_ = stack_clus.shape
            map_params_mod = [nx_, ny_, map_params[1]]
            pixels_to_mask = tools.get_pixels_to_mask(map_params_mod, circular_disk_rad_am)
      
            stack_dipole[pixels_to_mask] = None
            stack_dipole = np.ma.masked_invalid(stack_dipole)

            profile_lensing_dipole = np.ma.mean(stack_dipole, axis = 0)
            profile_lensing_dipole = np.ma.filled(profile_lensing_dipole, fill_value = 0.)
            
            profile_models_arr.append(profile_lensing_dipole)
     
        
        else:
            profile_model = np.mean(stack_dipole, axis = 0)   
            profile_models_arr.append(profile_model)
    
    return profile_models_arr