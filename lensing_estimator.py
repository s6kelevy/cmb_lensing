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


#################################################################################################################################


def get_aligned_cutout(map_params, image, l = None, cl = None, cl_noise = None):  
        
    if cl_noise is None:
        l = np.arange(10000)
        cl = np.ones(max(l)+1)
        cl_noise = np.zeros(max(l)+1)
    
    _, dx, _, _ = map_params
    cutout = tools.central_cutout(map_params, image, 10)
    wiener_filter = tools.wiener_filter(l, cl, cl_noise)
    filtered_map = tools.convolve(image, l, wiener_filter, map_params) 
    low_pass_filter = tools.low_pass_filter(l, 2000)
    filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
    filtered_cutout = tools.central_cutout(map_params, filtered_map, 6)
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
    stack -= np.median(stack)
    
    return stack


def lensing_dipole_profile(map_params, maps_clus, maps_rand,  l = None, cl = None, cl_noise = None, use_magnitude_weights = True, noise_weights = None, correct_for_tsz = False):
    
    stacks = []
    
    cutouts_arr = []
    magnitude_weights_clus_arr = []
    for i in tqdm(range(len(maps_clus))):
        cutout, weight = get_aligned_cutout(map_params, maps_clus[i], l = l, cl = cl, cl_noise = cl_noise)
        cutouts_arr.append(cutout)
        magnitude_weights_clus_arr.append(weight)
    if use_magnitude_weights is False:
        magnitude_weights_clus_arr = None
    stack_clus = get_stack(cutouts_arr, magnitude_weights_clus_arr, noise_weights) 
    stacks.append(stack_clus)
    
    cutouts_arr = []
    magnitude_weights_rand_arr = []
    for i in tqdm(range(len(maps_rand))):
        cutout, weight = get_aligned_cutout(map_params, maps_rand[i], l = l, cl = cl, cl_noise = cl_noise)
        cutouts_arr.append(cutout)
        magnitude_weights_rand_arr.append(weight)
    if use_magnitude_weights is False:
        magnitude_weights_rand_arr = None
    stack_bg = get_stack(cutouts_arr, magnitude_weights_rand_arr, noise_weights)
    stacks.append(stack_bg)
    
    stack_dipole = stack_clus-stack_bg
    stacks.append(stack_dipole)
    
    if correct_for_tsz is True:
        cutouts_arr = []
        for i in tqdm(range(len(maps_clus))):
            cutout = tools.central_cutout(map_params, maps_clus[i], 10)
            cutout -= np.median(cutout)
            cutouts_arr.append(cutout)
        stack_tsz = get_stack(cutouts_arr, magnitude_weights_clus_arr, noise_weights) 
        stacks.append(stack_tsz)
        stack_dipole_corrected = stack_dipole - stack_tsz
        stacks.append(stack_dipole_corrected)
        stack_dipole = np.copy(stack_dipole_corrected)
        
    _, dx, _, _ = map_params
    bins = np.arange((-40*dx)/2, (40*dx)/2, dx)
    profile_lensing_dipole = np.median(stack_dipole, axis = 0)            
    
    return bins, profile_lensing_dipole, stacks
    
    
def covariance_and_correlation_matrix(nber_cov, nber_clus, map_params, l, cl, freq_arr = None, cluster_corr_cutouts = None, cl_extragal = None, bl = None, nl = None, opbeam = None, components = 'all', cl_noise = None, use_magnitude_weights = True, noise_weights = None, correct_for_tsz = False):    
 
    if cl_noise is None:
        cl_noise = np.zeros(max(l)+1)
    
    sims_for_covariance = []
    for i in tqdm(range(nber_cov)):
        
        maps_clus = []
        for j in range(nber_clus):
            if freq_arr is None:
                map_clus = sims.cmb_mock_data(map_params, l, cl, cluster_corr_cutouts = cluster_corr_cutouts, cl_extragal = cl_extragal, bl = bl, nl = nl)
                maps_clus.append(map_clus)
            else:
                freq_maps_clus_dict = sims.cmb_mock_data_dict(freq_arr, map_params, l, cl, cluster_corr_cutouts_dict = cluster_corr_cutouts, cl_extragal_dict = cl_extragal, bl_dict = bl, nl_dict = nl)
                map_clus, _ = ilc.ilc_map(freq_maps_clus_dict, opbeam, map_params, experiment, components = components)
                maps_clus.append(map_clus)
   
        cutouts = []
        magnitude_weights_clus = []
        for i in tqdm(range(len(maps_clus))):
            cutout, weight = get_aligned_cutout(map_params, maps_clus[i], l, cl, cl_noise)
            cutouts.append(cutout)
            magnitude_weights_clus.append(weight)
        if use_magnitude_weights is False:
            magnitude_weights_clus = None
        stack_clus = get_stack(cutouts, magnitude_weights_clus, noise_weights) 
       
        if correct_for_tsz is True:
            cutouts = []
            for i in tqdm(range(len(maps_clus))):
                cutout = tools.central_cutout(map_params, image, 10)
                cutout -= np.median(cutout)
                cutouts_rand.append(cutout)
            stack_tsz = get_stack(cutouts, magnitude_weights_clus, noise_weights) 
            stack_clus -= stack_tsz

        profile_lensing_dipole = np.median(stack_clus, axis = 0)    
        sims_for_covariance.append(profile_lensing_dipole) 
   
    covariance_matrix, correlation_matrix = stats.covariance_and_correlation_matrix(sims_for_covariance, int(nber_pixels))
    
    return covariance_matrix, correlation_matrix


def model_profiles(nber_clus_fit, nber_rand_fit, map_params, l, cl, mass_int, c, z, centroid_shift_value = 0, cl_extragal = None, bl = None, cl_noise = None, use_magnitude_weights = True, use_noise_weights = False, apply_noise = True):
    
    nx, dx, ny, dy = map_params
    if cl_noise is None:
        cl_noise = np.zeros(max(l)+1)
        
    cutouts_clus_arr = []
    magnitude_weights_clus_arr = []    
    for i in tqdm(range(nber_clus_fit)):
        sim = tools.make_gaussian_realization(map_params, l, cl) 
        x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value), np.random.normal(loc=0.0, scale = centroid_shift_value) 
        centroid_shift = [x_shift, y_shift]
        for j in range(len(mass_int)):
            kappa_map = lensing.NFW(mass_int[j], c, z, 1100).convergence_map(map_params, centroid_shift = centroid_shift)
            alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map)
            sim_lensed = lensing.lens_map(map_params, sim, alpha_vec)   
            sim_lensed_noise = np.copy(sim_lensed)
            total_noise_map = tools.make_gaussian_realization(mapparams, l, cl_noise)
            sim_lensed_noise += total_noise_map
            if bl is not None:
                sim_lensed = tools.convolve(sim_lensed, l, np.sqrt(bl), map_params = map_params)
                sim_lensed_noise = tools.convolve(sim_lensed_noise, l, np.sqrt(bl), map_params = map_params)
            if apply_noise is False:
                sim_lensed_noise = np.copy(sim_lensed)
            cutout = tools.central_cutout(map_params, sim_lensed, 10) 
            wiener_filter = tools.wiener_filter(l, cl, cl_noise)
            filtered_map = tools.convolve(sim_lensed_noise, l, wiener_filter, map_params) 
            low_pass_filter = tool.low_pass_filter(l, 2000)
            filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
            filtered_cutout = tools.central_cutout(map_params, filtered_map, 6)
            _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
            angle, magnitude_weight = np.median(angle), np.median(magnitude) 
            cutout_aligned = tools.rotate(cutout, angle)
            cutout_aligned -= np.median(cutout_aligned)
            if use_magnitude_weights is False:
                magnitude_weight = 1
            cutouts_clus_arr.append(cutout_aligned*magnitude_weight)
            magnitude_weights_clus_arr.append(magnitude_weight)
    
  
    cutouts_rand_arr = []
    magnitude_weights_rand_arr = []    
    for i in tqdm(range(nber_rand_fit)):
        sim = tools.make_gaussian_realization(map_params, l, cl) 
        sim_noise = np.copy(sim)
        total_noise_map = tools.make_gaussian_realization(mapparams, l, cl_noise)
        sim_noise += total_noise_map
        if bl is not None:
            sim = tools.convolve(sim, l, np.sqrt(bl), map_params = map_params)
            sim_noise = tools.convolve(sim_noise, l, np.sqrt(bl), map_params = map_params)
        if apply_noise is False:
            sim_noise = np.copy(sim)
        cutout = tools.central_cutout(map_params, sim, 10)
        wiener_filter = tools.wiener_filter(l, cl, cl_noise)
        filtered_map = tools.convolve(sim_noise, l, wiener_filter, map_params) 
        low_pass_filter = tool.low_pass_filter(l, 2000)
        filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
        filtered_cutout = tools.central_cutout(map_params, filtered_map, 6)
        _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
        angle, magnitude_weight = np.median(angle), np.median(magnitude) 
        cutout_aligned = tools.rotate(cutout, angle)
        cutout_aligned -= np.median(cutout_aligned)
        cutouts_rand_arr.append(cutout_aligned)
        magnitude_weights_rand_arr.append(magnitude_weight)
    if use_magnitude_weights is False:
        magnitude_weights_rand_arr = np.ones(nber_rand_fit)
    weighted_cutouts = [cutouts_rand_arr[i]*magnitude_weights_rand_arr [i] for i in range(nber_rand_fit)]
    stack_bg = np.sum(weighted_cutouts, axis = 0)/np.sum(magnitude_weights_rand_arr)
    stack_bg -= np.median(stack_bg)
    
    profile_models_arr = [] 
    for i in tqdm(range(len(mass_int))):
        stack_clus = np.sum(cutouts_clus_arr[i::len(mass_int)], axis = 0)/np.sum(magnitude_weights_clus_arr[i::len(mass_int)])
        stack_clus -= np.median(stack_clus)
        stack_dipole = stack_clus-stack_bg
        profile_model = np.median(stack_dipole, axis = 0)   
        profile_models_arr.append(profile_model)
    
    return profile_models_arr