# importing relevant modules
import numpy as np
import random
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18')
from colossus.halo import concentration, mass_defs
from cosmo import CosmoCalc
import lensing
import foregrounds as fg
import experiments as exp
import tools


#################################################################################################################################


def cmb_mock_data(map_params, l, cl, cluster = None, centroid_shift_value = 0, nber_ch = 1, cluster_corr_cutouts = None, cl_extragal = None, bl = None, nl = None):
    
    nx, dx, ny, dy = map_params
    sim = tools.make_gaussian_realization(map_params, l, cl) 
   
    if cluster is not None:
        M, c, z = cluster
        x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value), np.random.normal(loc=0.0, scale = centroid_shift_value) 
        centroid_shift = [x_shift, y_shift]
        kappa = lensing.NFW(M, c, z, 1100).convergence_map(map_params, centroid_shift = centroid_shift)
        alpha_vec = lensing.deflection_from_convergence(map_params, kappa)
        sim = lensing.lens_map(map_params, sim, alpha_vec) 
    
    sims_ch_arr = [np.copy(sim) for k in range(nber_ch)]
    
    if cluster_corr_cutouts is not None:
        if np.asarray(cluster_corr_cutouts).ndim is 3:
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
    
    if nber_ch == 1:
         return sims_ch_arr[0]
        
    return sims_ch_arr 


def cmb_mock_data_dic(freq_arr, mapparams, l, cl, cluster = None, centroid_shift = None, cluster_corr_cutouts_dic = None, cl_extragal_dic = None, bl_dic = None, nl_dic = None):
    
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
    
    sims = cmb_mock_data(mapparams, l, cl, cluster = cluster, centroid_shift = centroid_shift, nber_ch = nber_ch, cluster_corr_cutouts = cluster_corr_cutouts_arr, ck_extragal = cl_extragal_arr, bl = bl_arr, nl = nl_arr)
    map_dic = {}
    for i, freq in enumerate(freq_arr):
        map_dic[freq] = sims[i]
        
    return map_dic


def cmb_test_data(nber_maps, validation_analyis = False, clus_position_analysis = False, extragal_bias_analysis = False):
    nx, dx, ny, dy = 240, 0.25, 240, 0.25
    map_params = [nx, dx, ny, dy]
    l, cl = CosmoCalc().cmb_power_spectrum()
    l, bl = exp.beam_power_spectrum(1.4)
    l, nl = exp.noise_power_spectrum(2.0)
    
    if validation_analyis is True:
        sims_2e14, sims_6e14, sims_10e14 = [], [], []
        kappa_map_2e14 = lensing.NFW(2e14, 3, 1, 1100).convergence_map(map_params)
        kappa_map_6e14 = lensing.NFW(6e14, 3, 1, 1100).convergence_map(map_params)
        kappa_map_10e14 = lensing.NFW(10e14, 3, 1, 1100).convergence_map(map_params)
        alpha_vec_2e14 = lensing.deflection_from_convergence(map_params, kappa_map_2e14)
        alpha_vec_6e14 = lensing.deflection_from_convergence(map_params, kappa_map_6e14)
        alpha_vec_10e14 = lensing.deflection_from_convergence(map_params, kappa_map_10e14)
        for i in range(nber_maps):
            sim = tools.make_gaussian_realization(map_params, l, cl) 
            sim_2e14 = lensing.lens_map(map_params, sim, alpha_vec_2e14)
            sim_6e14 = lensing.lens_map(map_params, sim, alpha_vec_6e14)
            sim_10e14 = lensing.lens_map(map_params, sim, alpha_vec_10e14)
            sim_2e14 = tools.convolve(sim_2e14, l, np.sqrt(bl), map_params = map_params)
            sim_6e14 = tools.convolve(sim_6e14, l, np.sqrt(bl), map_params = map_params)
            sim_10e14 = tools.convolve(sim_10e14, l, np.sqrt(bl), map_params = map_params)
            noise_map =  tools.make_gaussian_realization(map_params, l, nl)
            sim_2e14 += noise_map
            sim_6e14 += noise_map
            sim_10e14 += noise_map
            sims_2e14.append(sim_2e14)
            sims_6e14.append(sim_6e14)
            sims_10e14.append(sim_10e14)            
        return sims_2e14, sims_6e14, sims_10e14
    
    if clus_position_analysis is True:
        sims_baseline, sims_centorid_shift = [], []
        kappa_map_6e14_baseline = lensing.NFW(2e14, 3, 1, 1100).convergence_map(map_params)
        alpha_vec_6e14_baseline = lensing.deflection_from_convergence(map_params, kappa_map_6e14_baseline)
        for i in range(nber_maps):    
            x_shift, y_shift = np.random.normal(loc=0.0, scale = 0.5), np.random.normal(loc=0.0, scale = 0.5) 
            centroid_shift = [x_shift, y_shift]
            kappa_map_6e14_centroid_shift = lensing.NFW(6e14, 3, 1, 1100).convergence_map(map_params, centroid_shift)       
            alpha_vec_6e14_centroid_shift = lensing.deflection_from_convergence(map_params, kappa_map_6e14_centroid_shift)
            sim = tools.make_gaussian_realization(map_params, l, cl)
            sim_baseline = lensing.lens_map(map_params, sim, alpha_vec_6e14_baseline)
            sim_centroid_shift = lensing.lens_map(map_params, sim, alpha_vec_6e14_centroid_shift)
            sim_baseline = tools.convolve(sim_baseline, l, np.sqrt(bl), map_params = map_params)
            sim_centroid_shift = tools.convolve(sim_centroid_shift, l, np.sqrt(bl), map_params = map_params)
            noise_map =  tools.make_gaussian_realization(map_params, l, nl)
            sim_baseline += noise_map
            sim_centroid_shift += noise_map
            sims_baseline.append(sim_baseline)
            sims_centroid_shift.append(sim_centroid_shift)           
        return sims_baseline, sims_centroid_shift
      
    if extragal_bias_analysis is True:
        sims_baseline, sims_tsz, sims_ksz, sims_tsz_ksz = [], [], [], []
        c500 = concentration.concentration(2e14,'500c', 0.7)
        M200c, _, c200c = mass_defs.changeMassDefinition(2e14, c500, 0.7, '500c', '200c', profile='nfw')
        kappa_map_M200c = lensing.NFW(M200c, c200c, 0.7, 1100).convergence_map(map_params)
        alpha_vec_M200c = lensing.deflection_from_convergence(map_params, kappa_map_M200c)
        fname = '/Volumes/Extreme_SSD/codes/master_thesis/code/data/mdpl2_cutouts_for_tszksz_clus_detection_M1.7e+14to2.3e+14_z0.6to0.8_15320haloes_boxsize20.0am.npz'
        cutouts_dic = np.load(fname, allow_pickle = 1, encoding= 'latin1')['arr_0'].item()
        mass_z_key = list(cutouts_dic.keys())[0]
        cutouts = cutouts_dic[mass_z_key]
        scale_fac = fg.compton_y_to_delta_Tcmb(145, uK = True)
        tsz_cutouts, ksz_cutouts  = [], []
        for kcntr, keyname in enumerate( cutouts ):
            tsz_cutout = cutouts[keyname]['y']*scale_fac
            tsz_cutouts.append(tsz_cutout)
            ksz_cutout = cutouts[keyname]['ksz']*random.randrange(-1, 2, 2)
            ksz_cutouts.append(ksz_cutout)
        s, e = int((nx-40)/2), int((ny+40)/2)
        for i in range(nber_maps):
            sim = tools.make_gaussian_realization(map_params, l, cl)
            sim_M200c = lensing.lens_map(map_params, sim, alpha_vec_M200c)
            sim_baseline, sim_tsz, sim_ksz, sim_tsz_ksz = np.copy(sim_M200c), np.copy(sim_M200c), np.copy(sim_M200c), np.copy(sim_M200c)
            tsz_cutout = tools.rotate(tsz_cutouts[random.randint(0, len(tsz_cutouts)-1)], random.randint(-180,180))
            ksz_cutout = tools.rotate(ksz_cutouts[random.randint(0, len(ksz_cutouts)-1)], random.randint(-180,180))
            tsz_ksz_cutout = tsz_cutout+ksz_cutout
            sim_tsz[s:e, s:e] = sim_tsz[s:e, s:e] + tsz_cutout
            sim_ksz[s:e, s:e] = sim_ksz[s:e, s:e] + ksz_cutout
            sim_tsz_ksz[s:e, s:e] = sim_tsz_ksz[s:e, s:e] + tsz_ksz_cutout
            sim_baseline = tools.convolve(sim_baseline, l, np.sqrt(bl), map_params = map_params)
            sim_tsz = tools.convolve(sim_tsz, l, np.sqrt(bl), map_params = map_params)
            sim_ksz = tools.convolve(sim_ksz, l, np.sqrt(bl), map_params = map_params)
            sim_tsz_ksz = tools.convolve(sim_tsz_ksz, l, np.sqrt(bl), map_params = map_params)
            noise_map =  tools.make_gaussian_realization(map_params, l, nl)
            sim_baseline += noise_map
            sim_tsz += noise_map
            sim_ksz += noise_map
            sim_tsz_ksz += noise_map
            sims_baseline.append(sim_baseline)
            sims_tsz.append(sim_tsz)
            sims_ksz.append(sim_ksz)
            sims_tsz_ksz.append(sim_tsz_ksz)             
        return sims_baseline, sims_tsz, sims_ksz, sims_tsz_ksz