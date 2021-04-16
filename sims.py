# importing relevant modules
import numpy as np
import random
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18')
from colossus.halo import concentration, mass_defs
import cosmo
from cosmo import CosmoCalc
import lensing
import foregrounds as fg
import experiments as exp
import tools
import ilc


#################################################################################################################################


def cmb_mock_data(map_params, l, cl, cluster = None, centroid_shift_value = 0, nber_ch = 1, cluster_corr_cutouts = None, cl_extragal = None, bl = None, nl = None, nber_obs = 1):
    
    nx, dx, ny, dy = map_params
    
    sims = []
    for i in range(nber_obs):
        
        sim = tools.make_gaussian_realization(map_params, l, cl) 
        
        if cluster is not None:
            M, c, z = cluster
            x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value), np.random.normal(loc=0.0, scale =
                                                                                                         centroid_shift_value) 
            centroid_shift = [x_shift, y_shift]
            kappa_map = lensing.NFW(M, c, z, 1100).convergence_map(map_params, centroid_shift = centroid_shift)
            alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map)
            sim = lensing.lens_map(map_params, sim, alpha_vec) 
    
        sims_ch_arr = [np.copy(sim) for k in range(nber_ch)]
       
        if cluster_corr_cutouts is not None:
            if np.asarray(cluster_corr_cutouts).ndim == 3:
                cluster_corr_cutouts = [cluster_corr_cutouts]
            cluster_corr_cutout = cluster_corr_cutouts[0][0]
            nx_cutout, ny_cutout = cluster_corr_cutout.shape[0], cluster_corr_cutout.shape[1]
            s, e = int((nx-nx_cutout)/2), int((ny+ny_cutout)/2)
            rand_sel = random.randint(0, len(cluster_corr_cutouts[0])-1)
            rand_ang = random.randint(-180,180)
            for j in range(nber_ch):
                cluster_corr_cutout = tools.rotate(cluster_corr_cutouts[j][rand_sel], rand_ang)
                sims_ch_arr[j][s:e, s:e] = sims_ch_arr[j][s:e, s:e]+cluster_corr_cutout    
        
        if cl_extragal is not None:
            if isinstance(cl_extragal, list) is False:
                cl_extragal = [cl_extragal]
            extragal_maps = tools.make_gaussian_realization(map_params, l, cl_extragal)
            for j in range(nber_ch):
                sims_ch_arr[j] += extragal_maps[j]

        if bl is not None:
            if isinstance(bl, list) is False:
                bl = [bl]
            for j in range(nber_ch):    
                sims_ch_arr[j] = tools.convolve(sims_ch_arr[j], l, np.sqrt(bl[j]), map_params = map_params)

        if nl is not None:
            if isinstance(nl, list) is False:
                nl = [nl]
            for j in range(nber_ch):
                noise_map = tools.make_gaussian_realization(map_params, l, nl[j])
                sims_ch_arr[j] += noise_map
        
        sims.append(sims_ch_arr)
    
    if nber_ch == 1 and nber_obs == 1:  
        return sims[0][0]
    
    if nber_ch == 1:
        sims_one_freq = []
        for i in range(len(sims)):
            sims_one_freq.append(sims[i][0])
        return sims_one_freq
    
    if nber_obs == 1:
        return sims[0]
    
    sims_freq_sorted = []
    for i in range(nber_ch):
        maps_at_freq_i = []
        for j in range(nber_obs):
            maps_at_freq_i.append(sims[j][i])
        sims_freq_sorted.append(maps_at_freq_i)
    
    return sims_freq_sorted


def cmb_mock_data_dic(freq_arr, mapparams, l, cl, cluster = None, centroid_shift_value = 0, cluster_corr_cutouts_dic = None, cl_extragal_dic = None, bl_dic = None, nl_dic = None, nber_obs = 1):
    
    nber_ch = len(freq_arr)
    
    if cluster_corr_cutouts_dic is not None:
        cluster_corr_cutouts_arr = [cluster_corr_cutouts_dic[freq] for freq in sorted(cluster_corr_cutouts_dic.keys() )]  
    else:
        cluster_corr_cutouts_arr = None
    
    if cl_extragal_dic is not None:
        cl_extragal_arr = [cl_extragal_dic[freq] for freq in sorted(cl_extragal_dic.keys() )]
    else:
        cl_extragal_arr = None
    
    if bl_dic is not None:
        bl_arr = [bl_dic[freq] for freq in sorted(bl_dic.keys() )]
    else:
        bl_arr = None
    
    if nl_dic is not None:
        nl_arr = [nl_dic[freq] for freq in sorted(nl_dic.keys() )]
    else:
        nl_arr = None
   
    sims = cmb_mock_data(mapparams, l, cl, cluster = cluster, centroid_shift_value = centroid_shift_value, nber_ch = nber_ch, cluster_corr_cutouts = cluster_corr_cutouts_arr, cl_extragal = cl_extragal_arr, bl = bl_arr, nl = nl_arr, nber_obs = nber_obs)
    map_dic = {}
    for i, freq in enumerate(freq_arr):
        map_dic[freq] = sims[i]
   
    return map_dic


def cmb_test_data(map_params, l, cl, cluster = None, centroid_shift_value = 0, cluster_corr_cutouts = None, bl = None, nl = None, nber_obs = 1, estimator_validation = False, noise_comparison = False, clus_positions = False, foreground_bias = False):
    
    if estimator_validation is True:
        sims_clus_arr = []
        kappa_maps = [lensing.NFW(cluster[i][0], cluster[i][1], cluster[i][2], 1100).convergence_map(map_params) for i in range(len(cluster))]
        alpha_vecs = [lensing.deflection_from_convergence(map_params, kappa_maps[i]) for i in range(len(cluster))]
        for i in range(nber_obs):
            sim = tools.make_gaussian_realization(map_params, l, cl) 
            sims_lensed = [lensing.lens_map(map_params, sim, alpha_vecs[i]) for i in range(len(cluster))]
            if bl is not None:
                for j in range(len(sims_lensed)):
                    sims_lensed[j] = tools.convolve(sims_lensed[j], l, np.sqrt(bl), map_params = map_params)
            if nl is not None:
                noise_map =  tools.make_gaussian_realization(map_params, l, nl)
                for j in range(len(sims_lensed)):
                    sims_lensed[j] += noise_map
            sims_clus_arr.append(sims_lensed)
        sims_mass_sorted = []
        for i in range(len(cluster)):
            maps_at_mass_i = []
            for j in range(nber_obs):
                maps_at_mass_i.append(sims_clus_arr[j][i])
            sims_mass_sorted.append(maps_at_mass_i)
        return sims_mass_sorted
    
    
    if noise_comparison is True:
        sims_noise_arr = []
        kappa_map = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params) 
        alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map) 
        for i in range(nber_obs):
            sim = tools.make_gaussian_realization(map_params, l, cl) 
            sim_lensed = lensing.lens_map(map_params, sim, alpha_vec)
            if bl is not None:
                sim_lensed = tools.convolve(sim_lensed, l, np.sqrt(bl), map_params = map_params)
            sims_noise = [np.copy(sim_lensed) for i in range(len(nl))]
            noise_maps = [tools.make_gaussian_realization(map_params, l, nl[i]) for i in range(len(nl))]
            for i in range(len(sims_noise)):
                sims_noise[i] += noise_maps[i]
            sims_noise_arr.append(sims_noise)
        sims_noise_sorted = []
        for i in range(len(nl)):
            maps_at_noise_i = []
            for j in range(nber_obs):
                maps_at_noise_i.append(sims_noise_arr[j][i])
            sims_noise_sorted.append(maps_at_noise_i)
        return sims_noise_sorted
    
    
    if clus_positions is True:
        sims_clus_baseline, sims_clus_centroid_shift = [], []
        kappa_map_baseline = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params)
        alpha_vec_baseline = lensing.deflection_from_convergence(map_params, kappa_map_baseline)
        for i in range(nber_obs):    
            x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value), np.random.normal(loc=0.0, scale =
                                                                                                         centroid_shift_value) 
            centroid_shift = [x_shift, y_shift]
            kappa_map_centroid_shift = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params,
                                                                                                             centroid_shift)       
            alpha_vec_centroid_shift = lensing.deflection_from_convergence(map_params, kappa_map_centroid_shift)
            sim = tools.make_gaussian_realization(map_params, l, cl)
            sim_lensed_baseline = lensing.lens_map(map_params, sim, alpha_vec_baseline)
            sim_lensed_centroid_shift = lensing.lens_map(map_params, sim, alpha_vec_centroid_shift)
            if bl is not None:
                sim_lensed_baseline = tools.convolve(sim_lensed_baseline, l, np.sqrt(bl), map_params = map_params)
                sim_lensed_centroid_shift = tools.convolve(sim_lensed_centroid_shift, l, np.sqrt(bl), map_params = map_params)
            if nl is not None:
                noise_map =  tools.make_gaussian_realization(map_params, l, nl)
                sim_lensed_baseline += noise_map
                sim_lensed_centroid_shift += noise_map
            sims_clus_baseline.append(sim_lensed_baseline)
            sims_clus_centroid_shift.append(sim_lensed_centroid_shift)           
        return sims_clus_baseline, sims_clus_centroid_shift
      
        
    if foreground_bias is True:
        fname = '/Volumes/Extreme_SSD/codes/master_thesis/code/data/mdpl2_cutouts_for_tszksz_clus_detection_M1.7e+14to2.3e+14_z0.6to0.8_15320haloes_boxsize20.0am.npz'
        cutouts_dic = np.load(fname, allow_pickle = 1, encoding= 'latin1')['arr_0'].item()
        mass_z_key = list(cutouts_dic.keys())[0]
        cutouts = cutouts_dic[mass_z_key]
        scale_fac = fg.compton_y_to_delta_Tcmb(150, uK = True)
        tsz_cutouts, ksz_cutouts  = [], []
        for kcntr, keyname in enumerate( cutouts ):
            tsz_cutout = cutouts[keyname]['y']*scale_fac
            tsz_cutouts.append(tsz_cutout)
            ksz_cutout = cutouts[keyname]['ksz']*random.randrange(-1, 2, 2)
            ksz_cutouts.append(ksz_cutout)
        nx, dx, ny, dy = map_params
        s, e = int((nx-40)/2), int((ny+40)/2)

        sims_clus_baseline, sims_clus_tsz, sims_clus_ksz, sims_clus_tsz_ksz = [], [], [], []
        kappa_map = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params)
        alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map)
        for i in range(nber_obs):
            sim = tools.make_gaussian_realization(map_params, l, cl)
            sim_lensed = lensing.lens_map(map_params, sim, alpha_vec)
            sim_lensed_baseline, sim_lensed_tsz, sim_lensed_ksz, sim_lensed_tsz_ksz = np.copy(sim_lensed), np.copy(sim_lensed), np.copy(sim_lensed), np.copy(sim_lensed)
            tsz_cutout = tools.rotate(tsz_cutouts[random.randint(0, len(tsz_cutouts)-1)], random.randint(-180,180))
            ksz_cutout = tools.rotate(ksz_cutouts[random.randint(0, len(ksz_cutouts)-1)], random.randint(-180,180))
            tsz_ksz_cutout = tsz_cutout+ksz_cutout
            sim_lensed_tsz[s:e, s:e] = sim_lensed_tsz[s:e, s:e] + tsz_cutout
            sim_lensed_ksz[s:e, s:e] = sim_lensed_ksz[s:e, s:e] + ksz_cutout
            sim_lensed_tsz_ksz[s:e, s:e] = sim_lensed_tsz_ksz[s:e, s:e] + tsz_ksz_cutout
            if bl is not None:
                sim_lensed_baseline = tools.convolve(sim_lensed_baseline, l, np.sqrt(bl), map_params = map_params)
                sim_lensed_tsz = tools.convolve(sim_lensed_tsz, l, np.sqrt(bl), map_params = map_params)
                sim_lensed_ksz = tools.convolve(sim_lensed_ksz, l, np.sqrt(bl), map_params = map_params)
                sim_lensed_tsz_ksz = tools.convolve(sim_lensed_tsz_ksz, l, np.sqrt(bl), map_params = map_params)
            if nl is not None:     
                noise_map =  tools.make_gaussian_realization(map_params, l, nl)
                sim_lensed_baseline += noise_map
                sim_lensed_tsz += noise_map
                sim_lensed_ksz += noise_map
                sim_lensed_tsz_ksz += noise_map
            sims_clus_baseline.append(sim_lensed_baseline)
            sims_clus_tsz.append(sim_lensed_tsz)
            sims_clus_ksz.append(sim_lensed_ksz)
            sims_clus_tsz_ksz.append(sim_lensed_tsz_ksz)             
        return sims_clus_baseline, sims_clus_tsz, sims_clus_ksz, sims_clus_tsz_ksz
    
    
def cmb_forecast_data(experiment, freq_arr, map_params, l, cl, cluster = None, cluster_corr_cutouts = None, cl_residual = None, bl_arr = None, opbeam = None, nber_obs = 1):
   
    
    # obtaining weights     
    res_weights = ilc.residuals_and_weights(map_params = map_params, components = 'all', experiment = experiment, cov_from_sims = True) 
    l, cl_residual, res_ilc_dic, weights_arr = res_weights
   
    # rebeaming
    l, bl_dic = exp.beam_power_spectrum_dic(experiment = experiment, opbeam = opbeam)
    bl_rebeam_arr = exp.rebeam(bl_dic)

    # computing 2D versions 
    grid, _ = tools.make_grid(map_params, Fourier = True)
    weights_arr_2D = []
    for currW in weights_arr:
        l = np.arange(len(currW)) 
        currW_2D = tools.convert_to_2d(grid, l, currW)
        weights_arr_2D.append(currW_2D)
    weights_arr_2D = np.asarray( weights_arr_2D )
    
    rebeam_arr_2D = []
    for currB in bl_rebeam_arr:
        l = np.arange(len(currB))
        currB_2D = tools.convert_to_2d(grid, l, currB)
        rebeam_arr_2D.append(np.sqrt(currB_2D))
    rebeam_arr_2D = np.asarray( rebeam_arr_2D )
        
    # modify weights to include rebeam
    rebeamed_weights_arr = rebeam_arr_2D*weights_arr_2D

    nx, dx, ny, dy = map_params
    s, e = int((nx-40)/2), int((ny+40)/2)    
        
    if cluster is not None: 
        kappa_map = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params) 
        alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map)   
    if opbeam is not None:  
        l, bl_op = exp.beam_power_spectrum(beam_fwhm = opbeam)
    
    sims_arr = []
    
    for i in range(nber_obs):
        sim = tools.make_gaussian_realization(map_params, l, cl) 
        if cluster is not None: 
            sim = lensing.lens_map(map_params, sim, alpha_vec)
        if cl_residual is not None:
            residual_noise_map = tools.make_gaussian_realization(map_params, l, cl_residual)
            sim += residual_noise_map   
        if opbeam is not None:  
            sim = tools.convolve(sim, l, np.sqrt(bl_op), map_params = map_params)
        
        if cluster_corr_cutouts is not None:
            import sys
            print('Stop')
            sys.exit()
            tsz_ksz_cutout = tools.rotate(cluster_corr_cutouts[random.randint(0, len(cluster_corr_cutouts)-1)],
                                          random.randint(-180,180))
            tsz_ksz_arr = [np.copy(tsz_ksz_cutout) for k in range(len(freq_arr))]
            for v in range(len(freq_arr)):
                tsz_ksz_arr[v] *= fg.compton_y_to_delta_Tcmb(freq_arr[v], uK = True)
                tsz_ksz_arr[v] = tools.convolve(np.pad(tsz_ksz_arr[v], (100, 100), 'constant', constant_values=(0, 0)), l, np.sqrt(bl_arr[v]), map_params = map_params)
            # compute ilc map 
            weighted_maps_arr = []
            for mm in range(len(tsz_ksz_arr)):
                curr_map = tsz_ksz_arr[mm]
                rebeamed_weights_arr[mm][np.isnan(rebeamed_weights_arr[mm])]=0.
                rebeamed_weights_arr[mm][np.isinf(rebeamed_weights_arr[mm])]=0.
                map_weighted = np.fft.fft2(curr_map) * rebeamed_weights_arr[mm]
                weighted_maps_arr.append(map_weighted)
            ilc_map_fft = np.sum(weighted_maps_arr, axis = 0)
            ilc_map = np.fft.ifft2(ilc_map_fft).real
            sim[s:e, s:e] = sim[s:e, s:e] + ilc_map[s:e, s:e]
        sims_arr.append(sim)
    
    return sims_arr



def cmb_forecast_test_data(map_params, l, cl, cluster = None, cl_residuals = None, bl = None, nber_obs = 1):
        
    if cluster is not None: 
        kappa_map = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params) 
        alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map)   
   
    sims_arr_so, sims_arr_fyst, sims_arr_s4wide = [], [], []
    
    for i in range(nber_obs):
        sim = tools.make_gaussian_realization(map_params, l, cl) 
        if cluster is not None: 
            sim = lensing.lens_map(map_params, sim, alpha_vec)
        sim_so, sim_fyst, sim_s4wide = np.copy(sim), np.copy(sim), np.copy(sim)
        if cl_residuals is not None:
            residual_noise_map_so = tools.make_gaussian_realization(map_params, l, cl_residuals[0])
            sim_so += residual_noise_map_so   
            residual_noise_map_fyst = tools.make_gaussian_realization(map_params, l, cl_residuals[1])
            sim_fyst += residual_noise_map_fyst      
            residual_noise_map_s4wide = tools.make_gaussian_realization(map_params, l, cl_residuals[2])
            sim_s4wide += residual_noise_map_s4wide              
        if bl is not None:  
            sim_so = tools.convolve(sim_so, l, np.sqrt(bl), map_params = map_params)
            sim_fyst = tools.convolve(sim_fyst, l, np.sqrt(bl), map_params = map_params)
            sim_s4wide = tools.convolve(sim_s4wide, l, np.sqrt(bl), map_params = map_params)   
        sims_arr_so.append(sim_so)
        sims_arr_fyst.append(sim_fyst)
        sims_arr_s4wide.append(sim_s4wide)
    return sims_arr_so, sims_arr_fyst, sims_arr_s4wide