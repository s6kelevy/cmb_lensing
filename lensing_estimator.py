# importing relevant modules
import numpy as np
from scipy import signal
import random
import tools
from tqdm import tqdm
import lensing
import cosmo
import stats
import sims
import hlc



def get_aligned_cutout(map_params, image, l, cl, cl_noise):
    _, dx, _, _ = map_params
    cutout = tools.central_cutout(map_params, image, 10)
    wiener_filter = tools.wiener_filter(l, cl, cl_noise)
    filtered_map = tools.convolve(image, l, wiener_filter, map_params) 
    low_pass_filter = tools.low_pass_filter(l, 2000)
    filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
    filtered_cutout = tools.central_cutout(map_params, filtered_map, 6)
    _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
    angle, magnitude_weight = np.median(angle), np.median(magnitude) 
    rotated_cutout = tools.rotate(cutout, angle)
    cutout_aligned = rotated_cutout-np.median(rotated_cutout)
    return cutout_aligned, magnitude_weight
    
    
    
    
def get_random_cutout(mapparams, image):
    cutout = tools.central_cutout(mapparams, image, 10)
    cutout_rand = tools.rotate(cutout, random.randint(-180,180))
    cutout_rand = cutout_rand-np.mean(cutout_rand)
    return cutout_rand
    
    
    
      
def get_stack(cutouts, magnitude_weights = None, noise_weights = None):
    # stacking weighted cutouts to obtain weighted stacked signal
    if magnitude_weights is None:
        magnitude_weights = np.ones(len(cutouts))
    if noise_weights is None:
        noise_weights = np.ones(len(cutouts))    
    weights = np.array(magnitude_weights)*np.array(noise_weights)
    weighted_cutouts = [cutouts[i]*weights[i] for i in range(len(cutouts))]
    stack = np.sum(weighted_cutouts, axis = 0)/sum(weights)
    return stack



def get_dipole_profile(mapparams, maps_clus, maps_rand,  l, cl, cl_noise, use_magnitude_weights = True, noise_weights = None, correct_for_tsz = False):
    
    stacks = []
    cutouts_aligned = []
    magnitude_weights_clus = []
    for i in range(len(maps_clus)):
        cutout, weight = get_aligned_cutout(mapparams, maps_clus[i], l, cl, cl_noise)
        cutouts_aligned.append(cutout)
        magnitude_weights_clus.append(weight)
    if use_magnitude_weights is False:
        magnitude_weights_clus = None
    stack_clus = get_stack(cutouts_aligned, magnitude_weights_clus, noise_weights) 
    stacks.append(stack_clus)
    
        
    
    cutouts_aligned = []
    magnitude_weights = []
    for i in range(len(maps_rand)):
        cutout, weight = get_aligned_cutout(mapparams, maps_rand[i], l, cl, cl_noise)
        cutouts_aligned.append(cutout)
        magnitude_weights.append(weight)
    if use_magnitude_weights is False:
        magnitude_weights = None
    stack_bg = get_stack(cutouts_aligned, magnitude_weights, noise_weights)
    stacks.append(stack_bg)
    
    stack_dipole = stack_clus-stack_bg
    stacks.append(stack_dipole)
    
    
    if correct_for_tsz is True:
        cutouts_rand = []
        for i in range(len(maps_clus)):
            cutout = get_random_cutout(mapparams, maps_clus[i])
            cutouts_rand.append(cutout)
        stack_tsz = get_stack(cutouts_rand, magnitude_weights_clus, noise_weights) 
        stack_dipole_corrected = stack_dipole - stack_tsz
        stacks.append(stack_tsz)
        stacks.append(stack_dipole_corrected)
        stack_dipole = stack_dipole_corrected
    
    
    _, dx, _, _ = mapparams
    bins = np.arange((-40*dx)/2, (40*dx)/2, dx)
    dipole_profile = np.mean(stack_dipole, axis = 0)            
    
    
    return bins, dipole_profile, stacks
    
    
def covariance_matrix(nber_clus, nber_rand, nber_cov, map_params, l, cl, cluster_corr_cutouts = None, cl_extragal = None, bl = None, nl = None, cl_noise = 0, use_magnitude_weights = True, noise_weights = None, correct_for_tsz = False):    
    sims_for_covariance = []
    grid, _ = tools.make_grid(map_params)
    X,Y = grid
    maps_rand = [1.5*X]
    for i in tqdm(range(nber_cov)):
        maps_clus = sims.cmb_mock_data(nber_clus, map_params, l, cl, cluster_corr_cutouts = cluster_corr_cutouts, cl_extragal = cl_extragal, bl = bl, nl = nl)
        _, dipole_profile, _ = get_dipole_profile(map_params, maps_clus, maps_rand, l, cl, cl_noise, use_magnitude_weights, noise_weights, correct_for_tsz)
        sims_for_covariance.append(dipole_profile) 
    nx, dx, ny, dy = map_params
    nber_pixels = 10/dx
    cov = stats.covariance_matrix(sims_for_covariance, int(nber_pixels))
    return cov


def covariance_matrix2(nber_clus, nber_rand, nber_cov, freq_arr, map_params, l, cl, cluster_corr_cutouts_dict = None, cl_extragal_dict= None, bl_dict = None, nl_dict = None, cl_noise = 0, use_magnitude_weights = True, noise_weights = None, correct_for_tsz = False):    
    sims_for_covariance = []
    for i in tqdm(range(nber_cov)):
        maps_clus = []
        for i in range(nber_clus):
            cmb_maps_dict = sims.cmb_mock_data_dict(freq_arr, map_params, l, cl, cluster_corr_cutouts_dict = cluster_corr_cutouts_dict, cl_extragal_dict = cl_extragal_dict, bl_dict = bl_dict, nl_dict = nl_dict)
            hlc_map, _ = hlc.hlc_map(cmb_maps_dict, opbeam, map_params, components, experiment)
            maps_cluse.append(hlc_map)
    
        maps_rand = []
        for i in range(nber_rand):
            cmb_maps_dict = sims.cmb_mock_data_dict(freq_arr, map_params, l, cl, cl_extragal_dict = cl_extragal_dict, bl_dict = bl_dict, nl_dict = nl_dict)
            hlc_map, _ = hlc.hlc_map(cmb_maps_dict, opbeam, map_params, components, experiment)
            maps_rand.append(hlc_map) 
        _, dipole_profile, _ = get_dipole_profile(map_params, maps_clus, maps_rand, l, cl, cl_noise, use_magnitude_weights, noise_weights, correct_for_tsz)
        sims_for_covariance.append(dipole_profile) 
    nx, dx, ny, dy = map_params
    nber_pixels = 10/dx
    cov = stats.covariance_matrix(sims_for_covariance, int(nber_pixels))
    return cov


def fit_profiles(nber_clus_fit, nber_rand_fit, map_params, l, cl, mass_int, c, z, centroid_shift = None, cl_uncorr_fg = None, bl = None, nl = None, cl_noise = 0, use_magnitude_weights = True, use_noise_weights = False, apply_noise = True):
    _, dx, _, _ = mapparams
    cutouts = []
    magnitude_weights = []    
    for i in tqdm(range(nber_rand_fit)):
        sim = tools.make_gaussian_realization(map_params, l, cl) 
        sim_noise = np.copy(sim)
        if cl_uncorr_fg is not None:
            extragal_map = tools.make_gaussian_realization(map_params, l, cl_uncorr_fg)
            sim_noise += extragal_map
        if bl is not None:
            sim = tools.convolve(sim, l, np.sqrt(bl), map_params)
            sim_noise = tools.convolve(sim_noise, l, np.sqrt(bl), map_params)
        if nl is not None:
            noise_map = tools.make_gaussian_realization(mapparams, l, nl)
            sim_noise = sim_noise + noise_map
        cutout = tools.central_cutout(mapparams, sim, 10)
        if apply_noise is False:
            sim_noise = np.copy(sim)
        wiener_filter = tools.wiener_filter(l, cl, cl_noise)
        filtered_map = tools.convolve(sim_noise, l, wiener_filter, map_params) 
        low_pass_filter = tool.low_pass_filter(l, 2000)
        filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
        filtered_cutout = tools.central_cutout(mapparams, filtered_map, 6)
        _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
        angle, magnitude_weight = np.median(angle), np.median(magnitude) 
        rotated_cutout = tools.rotate(cutout, angle)
        rotated_cutout = rotated_cutout-np.median(rotated_cutout)
        cutouts.append(rotated_cutout)
        magnitude_weights.append(magnitude_weight)
    if use_magnitude_weights is False:
        magnitude_weights = np.ones(nber_rand_fit)
    weighted_cutouts = [cutouts[i]*magnitude_weights[i] for i in range(nber_rand_fit)]
    unlensed_stack = np.sum(weighted_cutouts, axis = 0)/np.sum(magnitude_weights)
    
    
    alpha_vecs = []
    for i in tqdm(range(len(mass_int))):
        kappa = lensing.NFW(mass_int[i], c, z, 1100).kappa_map(mapparams)
        alpha_vec = lensing.alpha_from_kappa(mapparams, kappa)
        alpha_vecs.append(alpha_vec)
    
    
    cutouts = []
    magnitude_weights = []    
    for i in tqdm(range(nber_clus_fit)):
        sim = tools.make_gaussian_realization(mapparams, l, cl) 
        for j in range(len(mass_int)):
            sim_lensed = lensing.lens_map(mapparams, sim, alpha_vecs[j], centroid_shift)   
            sim_noise = np.copy(sim_lensed)
            if cl_uncorr_fg is not None:
                extragal_map = tools.make_gaussian_realization(mapparams, l, cl_uncorr_fg)
                sim_noise += extragal_map
            if bl is not None:
                sim_lensed = tools.convolve(sim_lensed, l, np.sqrt(bl), map_params = map_params)
                sim_noise = tools.convolve(sim_noise, l, np.sqrt(bl), map_params = map_params)
            if nl is not None:
                noise_map = tools.make_gaussian_realization(mapparams, l, nl)
                sim_noise += noise_map
            cutout = tools.central_cutout(mapparams, sim_lensed, 10)
            wiener_filter = tools.wiener_filter(l, cl, cl_noise)
            filtered_map = tools.convolve(sim_noise, l, wiener_filter, map_params) 
            low_pass_filter = tool.low_pass_filter(l, 2000)
            filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
            filtered_cutout = tools.central_cutout(mapparams, filtered_map, 6)
            _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
            angle, magnitude_weight = np.median(angle), np.median(magnitude) 
            rotated_cutout = tools.rotate(cutout, angle)
            rotated_cutout = rotated_cutout-np.median(rotated_cutout)
            if use_magnitude_weights is False:
                magnitude_weight = 1
            cutouts.append(rotated_cutout*magnitude_weight)
            magnitude_weights.append(magnitude_weight)
    dipole_profile_models = [] 
    for i in tqdm(range(len(mass_int))):
        lensed_stack = np.sum(cutouts[i::len(mass_int)], axis = 0)/np.sum(magnitude_weights[i::len(mass_int)])
        dipole_stack = lensed_stack-unlensed_stack
        dipole_profile = np.mean(dipole_stack, axis = 0)   
        dipole_profile_models.append(dipole_profile)
    return dipole_profile_models






def fit_profiles2(nber_clus_fit, nber_rand_fit, mapparams, l, cl, mass_int, c, z, centroid_shift = None, cl_uncorr_fg = None, bl = None, cl_noise = None, use_magnitude_weights = True, use_noise_weights = False, apply_noise = True):
    _, dx, _, _ = mapparams
    cutouts = []
    magnitude_weights = []    
    for i in tqdm(range(nber_rand_fit)):
        sim = tools.make_gaussian_realization(mapparams, l, cl) 
        sim_noise = np.copy(sim)
        if cl_noise is not None:
            extragal_map = tools.make_gaussian_realization(mapparams, l, cl_noise)
            sim_noise += extragal_map
        if bl is not None:
            sim = tools.convolve(sim, l, np.sqrt(bl), map_params = map_params)
            sim_noise = tools.convolve(sim_noise, l, np.sqrt(bl), map_params = map_params)
        cutout = tools.central_cutout(mapparams, sim, 10)
        if apply_noise is False:
            sim_noise = np.copy(sim)
        wiener_filter = tools.wiener_filter(l, cl, cl_noise)
        filtered_map = tools.convolve(sim_noise, l, wiener_filter, map_params) 
        low_pass_filter = tool.low_pass_filter(l, 2000)
        filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
        _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
        angle, magnitude_weight = np.median(angle), np.median(magnitude) 
        rotated_cutout = tools.rotate(cutout, angle)
        rotated_cutout = rotated_cutout-np.median(rotated_cutout)
        cutouts.append(rotated_cutout)
        magnitude_weights.append(magnitude_weight)
    if use_magnitude_weights is False:
        magnitude_weights = np.ones(nber_rand_fit)
    weighted_cutouts = [cutouts[i]*magnitude_weights[i] for i in range(nber_rand_fit)]
    unlensed_stack = np.sum(weighted_cutouts, axis = 0)/np.sum(magnitude_weights)
    
    
    alpha_vecs = []
    for i in tqdm(range(len(mass_int))):
        kappa = lensing.NFW(mass_int[i], c, z, 1100).kappa_map(mapparams)
        alpha_vec = lensing.alpha_from_kappa(mapparams, kappa)
        alpha_vecs.append(alpha_vec)
    
    
    cutouts = []
    magnitude_weights = []    
    for i in tqdm(range(nber_clus_fit)):
        sim = tools.make_gaussian_realization(mapparams, l, cl) 
        for j in range(len(mass_int)):
            sim_lensed = lensing.lens_map(mapparams, sim, alpha_vecs[j], centroid_shift)   
            sim_noise = np.copy(sim_lensed)
            if cl_noise is not None:
                extragal_map = tools.make_gaussian_realization(mapparams, l, cl_noise)
                sim_noise += extragal_map
            if bl is not None:
                sim_lensed = tools.convolve(sim_lensed, l, np.sqrt(bl), map_params = map_params)
                sim_noise = tools.convolve(sim_noise, l, np.sqrt(bl), map_params = map_params)
            cutout = tools.central_cutout(mapparams, sim_lensed, 10)
            wiener_filter = tools.wiener_filter(l, cl, cl_noise)
            filtered_map = tools.convolve(sim_noise, l, wiener_filter, map_params) 
            low_pass_filter = tool.low_pass_filter(l, 2000)
            filtered_map = tools.convolve(filtered_map, l, low_pass_filter, map_params) 
            filtered_cutout = tools.central_cutout(mapparams, filtered_map, 6)
            _, _, magnitude, angle = tools.gradient(filtered_cutout, dx)
            angle, magnitude_weight = np.median(angle), np.median(magnitude) 
            rotated_cutout = tools.rotate(cutout, angle)
            rotated_cutout = rotated_cutout-np.median(rotated_cutout)
            if use_magnitude_weights is False:
                magnitude_weight = 1
            cutouts.append(rotated_cutout*magnitude_weight)
            magnitude_weights.append(magnitude_weight)
    dipole_profile_models = [] 
    for i in tqdm(range(len(mass_int))):
        lensed_stack = np.sum(cutouts[i::len(mass_int)], axis = 0)/np.sum(magnitude_weights[i::len(mass_int)])
        dipole_stack = lensed_stack-unlensed_stack
        dipole_profile = np.mean(dipole_stack, axis = 0)   
        dipole_profile_models.append(dipole_profile)
    return dipole_profile_models

