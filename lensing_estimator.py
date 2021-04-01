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
    #cutout = tools.central_cutout(map_params, image, cutout_size_am)
    #wiener_filter = tools.wiener_filter(l, cl, cl_noise)
    #filtered_map = tools.convolve(image, l, wiener_filter, map_params) 
    #low_pass_filter = tools.low_pass_filter(l, l_cut)
    #filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
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
    
    
def lensing_dipole(map_params, stack_clus, stack_bg, stack_tsz = None):
    
    if stack_tsz is None:
        stack_dipole = stack_clus-stack_bg
    else:
        stack_dipole = stack_clus-stack_bg-stack_tsz
    
        
    nx, dx, _, _ = map_params
    bins = np.arange((-40*dx)/2, (40*dx)/2, dx)
    profile_lensing_dipole = np.mean(stack_dipole, axis = 0)            
    
    return bins, profile_lensing_dipole, stack_dipole
    
    
def covariance_and_correlation_matrix(nber_cov, nber_clus, map_params, l, cl, freq_arr = None, cluster_corr_cutouts = None, cl_extragal = None, bl = None, nl = None, opbeam = None, components = 'all', cutout_size_am = 10, l_cut  = 2000, cl_noise = None, use_magnitude_weights = True, noise_weights = None, correct_for_tsz = False):    
 
    
    sims_for_covariance = []
    for i in tqdm(range(nber_cov)):
          
        if freq_arr is None:
            maps_clus = sims.cmb_mock_data(map_params, l, cl, cluster_corr_cutouts = cluster_corr_cutouts, cl_extragal = cl_extragal, bl = bl, nl = nl, nber_obs = nber_clus)
        else:
            maps_clus = []
            for j in range(nber_clus):
                freq_maps_clus_dic = sims.cmb_mock_data_dic(freq_arr, map_params, l, cl, cluster_corr_cutouts_dic =
                                                            cluster_corr_cutouts, cl_extragal_dic = cl_extragal, bl_dic = bl,
                                                            nl_dic = nl)
                map_clus, _ = ilc.ilc_map(freq_maps_clus_dic, opbeam, map_params, experiment, components = components)
                maps_clus.append(map_clus)
   
        if correct_for_tsz is False:
            stack_clus = stack(map_params, maps_clus, cutout_size_am = cutout_size_am, l_cut = l_cut, l = l, cl = cl, cl_noise =
                               cl_noise, use_magnitude_weights = use_magnitude_weights, noise_weights = noise_weights, 
                               correct_for_tsz = correct_for_tsz) 
           
        else:
            stack_clus, stack_tsz = stack(map_params, maps_clus, cutout_size_am = cutout_size_am, l_cut = l_cut, l = l, cl = cl,
                                          cl_noise = cl_noise,use_magnitude_weights = use_magnitude_weights, noise_weights =
                                          noise_weights, correct_for_tsz = correct_for_tsz) 
            stack_clus -= stack_tsz


        profile_lensing_dipole = np.mean(stack_clus, axis = 0)   
        sims_for_covariance.append(profile_lensing_dipole) 
   
    covariance_matrix, correlation_matrix = stats.covariance_and_correlation_matrix(sims_for_covariance)
    
    return covariance_matrix, correlation_matrix


def model_profiles(nber_clus_fit, nber_rand_fit, map_params, l, cl, mass_int, z, centroid_shift_value = 0, cl_extragal = None, bl = None, cl_noise = None, use_magnitude_weights = True, use_noise_weights = False, apply_noise = True):
    
    nx, dx, ny, dy = map_params
    if cl_noise is None:
        cl_noise = np.zeros(max(l)+1)
    
    mass_int = np.copy(mass_int)*1e14
    
    cutouts_clus_arr = []
    magnitude_weights_clus_arr = []    
    for i in tqdm(range(nber_clus_fit)):
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
           # cutout = tools.central_cutout(map_params, sim_lensed, 10) 
           # wiener_filter = tools.wiener_filter(l, cl, cl_noise)
           # filtered_map = tools.convolve(sim_lensed_noise, l, wiener_filter, map_params) 
          #  low_pass_filter = tools.low_pass_filter(l, 2000)
          #  filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
          #  filtered_cutout = tools.central_cutout(map_params, filtered_map, 6)
          #  _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
          #  angle, magnitude_weight = np.median(angle), np.median(magnitude) 
          #  cutout_aligned = tools.rotate(cutout, angle)
          #  cutout_aligned -= np.median(cutout_aligned)
         #   if use_magnitude_weights is False:
        #        magnitude_weight = 1
         #   cutouts_clus_arr.append(cutout_aligned*magnitude_weight)
         #   magnitude_weights_clus_arr.append(magnitude_weight)
    
            cutout_aligned, magnitude_weight = get_aligned_cutout(map_params, sim_lensed_noise, image_noiseless = sim_lensed,
                                                                  l = l, cl = cl, cl_noise = cl_noise)
            if use_magnitude_weights is False:
                magnitude_weight = 1
            cutouts_clus_arr.append(cutout_aligned*magnitude_weight)
            magnitude_weights_clus_arr.append(magnitude_weight)
    
   # cutouts_rand_arr = []
   # magnitude_weights_rand_arr = []    
   # for i in tqdm(range(nber_rand_fit)):
   #     sim = sims.cmb_mock_data(map_params, l, cl) 
   #     sim_noise = np.copy(sim)
   #     total_noise_map = tools.make_gaussian_realization(map_params, l, cl_noise)
   #     sim_noise += total_noise_map
   #     if bl is not None:
   #         sim = tools.convolve(sim, l, np.sqrt(bl), map_params = map_params)
   #         sim_noise = tools.convolve(sim_noise, l, np.sqrt(bl), map_params = map_params)
   #     if apply_noise is False:
   #         sim_noise = np.copy(sim)
   #     cutout = tools.central_cutout(map_params, sim, 10)
   #     wiener_filter = tools.wiener_filter(l, cl, cl_noise)
   #     filtered_map = tools.convolve(sim_noise, l, wiener_filter, map_params) 
   #     low_pass_filter = tools.low_pass_filter(l, 2000)
   #     filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
   #     filtered_cutout = tools.central_cutout(map_params, filtered_map, 6)
   #     _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
   #     angle, magnitude_weight = np.median(angle), np.median(magnitude) 
   #     cutout_aligned = tools.rotate(cutout, angle)
   #     cutout_aligned -= np.median(cutout_aligned)
   #     cutouts_rand_arr.append(cutout_aligned)
   #     magnitude_weights_rand_arr.append(magnitude_weight)
   # if use_magnitude_weights is False:
   #     magnitude_weights_rand_arr = np.ones(nber_rand_fit)
   # weighted_cutouts = [cutouts_rand_arr[i]*magnitude_weights_rand_arr [i] for i in range(nber_rand_fit)]
   # stack_bg = np.sum(weighted_cutouts, axis = 0)/np.sum(magnitude_weights_rand_arr)
   
    profile_models_arr = [] 
    stack_bg = np.sum(cutouts_clus_arr[0::len(mass_int)], axis = 0)/np.sum(magnitude_weights_clus_arr[0::len(mass_int)])
    for i in tqdm(range(len(mass_int))):
        stack_clus = np.sum(cutouts_clus_arr[i::len(mass_int)], axis = 0)/np.sum(magnitude_weights_clus_arr[i::len(mass_int)])
        stack_dipole = stack_clus-stack_bg
        profile_model = np.mean(stack_dipole, axis = 0)   
        profile_models_arr.append(profile_model)
    
    return profile_models_arr