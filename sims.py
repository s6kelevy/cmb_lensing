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


def cmb_mock_data(map_params, l, cl, cluster = None, centroid_shift = None, nber_ch = 1, cluster_corr_cutouts_arr = None, cl_extragal_arr = None, bl_arr = None, nl_arr = None):
    nx, dx, ny, dy = map_params
    if cluster is not None:
        M, c, z = cluster
        kappa = lensing.NFW(M, c, z, 1100).kappa_map(map_params)
        alpha_vec = lensing.alpha_from_kappa(map_params, kappa)
    if cluster_corr_cutouts_arr is not None:
        cluster_corr_cutout = cluster_corr_cutouts_arr[0][0]
        nx_cutout, ny_cutout = cluster_corr_cutout.shape[0], cluster_corr_cutout.shape[1]
        s, e = int((nx-nx_cutout)/2), int((ny+ny_cutout)/2)
    sim = tools.make_gaussian_realization(map_params, l, cl) 
    if cluster is not None:
        sim = lensing.lens_map(map_params, sim, alpha_vec, centroid_shift = centroid_shift) 
    sims_ch_arr = [np.copy(sim) for k in range(nber_ch)]
    if cluster_corr_cutouts_arr is not None:
        rand_sel = random.randint(0, len(cluster_corr_cutouts_arr[0])-1)
        rand_ang = random.randint(-180,180)
        for j in range(nber_ch):
            cluster_corr_cutout = tools.rotate(cluster_corr_cutouts_arr[j][rand_sel], rand_ang)
            sims_ch_arr[j][s:e, s:e] = sims_ch_arr[j][s:e, s:e]+cluster_corr_cutout    
    if cl_extragal_arr is not None:
        extragal_maps = tools.make_gaussian_realization(map_params, l, cl_extragal_arr)
        for j in range(nber_ch):
            sims_ch_arr[j] += extragal_maps[j]
    if bl_arr is not None:
        for j in range(nber_ch):    
            sims_ch_arr[j] = tools.convolve(sims_ch_arr[j], l, np.sqrt(bl_arr[j]), map_params = map_params)
    if nl_arr is not None:
        for j in range(nber_ch):
            noise_map = tools.make_gaussian_realization(map_params, l, nl_arr[j])
            sims_ch_arr[j] += noise_map
    if nber_ch == 1:
        sims_ch_arr = sims_ch_arr[0]
    return sims_ch_arr 


def cmb_mock_data_dict(freq_arr, mapparams, l, cl, cluster = None, centroid_shift = None, cluster_corr_cutouts_dict = None, cl_extragal_dict = None, bl_dict = None, nl_dict = None):
    nber_ch = len(freq_arr)
    if cluster_corr_cutouts_dict is not None:
        cluster_corr_cutouts_arr = [cluster_corr_cutouts_dict[freq] for freq in sorted(cluster_corr_cutouts_dict.keys() )] 
    else:
        cluster_corr_cutouts_arr = None
    if cl_extragal_dict is not None:
        cl_extragal_arr = [cl_extragal_dict[freq] for freq in sorted(cl_extragal_dict.keys() )]
    else:
        cl_extragal_arr = None
    if bl_dict is not None:
        bl_arr = [bl_dict[freq] for freq in sorted(bl_dict.keys() )]
    else:
        bl_arr = None
    if nl_dict is not None:
        nl_arr = [nl_dict[freq] for freq in sorted(nl_dict.keys() )]
    else:
        nl_arr = None
    sims = cmb_mock_data(mapparams, l, cl, cluster, centroid_shift, nber_ch, cluster_corr_cutouts_arr, cl_extragal_arr, bl_arr, nl_arr)
    map_dic = {}
    for i, freq in enumerate(freq_arr):
        map_dic[freq] = sims[i]
    return map_dic


def cmb_test_data(nber_maps, validation_analyis = False, clus_position_analysis = False, extragal_bias_analysis = False):
    nx, dx, ny, dy = 240, 0.25, 240, 0.25
    map_params = [nx, dx, ny, dy]
    l, cl = CosmoCalc().cmb_power_spectrum()
    if validation_analyis is True or clus_position_analysis is True:  
        kappa_map_2e14 = lensing.NFW(2e14, 3, 1, 1100).kappa_map(map_params)
        kappa_map_6e14 = lensing.NFW(6e14, 3, 1, 1100).kappa_map(map_params)
        kappa_map_10e14 = lensing.NFW(10e14, 3, 1, 1100).kappa_map(map_params)
        alpha_vec_2e14 = lensing.alpha_from_kappa(map_params, kappa_map_2e14)
        alpha_vec_6e14 = lensing.alpha_from_kappa(map_params, kappa_map_6e14)
        alpha_vec_10e14 = lensing.alpha_from_kappa(map_params, kappa_map_10e14)
    if extragal_bias_analysis is True:  
        c500 = concentration.concentration(2e14,'500c', 0.7)
        M200c, _, c200c = mass_defs.changeMassDefinition(2e14, c500, 0.7, '500c', '200c', profile='nfw')
        kappa_map_M200c = lensing.NFW(M200c, c200c, 0.7, 1100).kappa_map(map_params)
        alpha_vec_M200c = lensing.alpha_from_kappa(map_params, kappa_map_M200c)
        fname = 'sim_data/mdpl2_cutouts_for_tszksz_clus_detection_M1.7e+14to2.3e+14_z0.6to0.8_15320haloes_boxsize20.0am.npz'
        cutouts_dic = np.load(fname, allow_pickle = 1, encoding= 'latin1')['arr_0'].item()
        mass_z_key = list(cutouts_dic.keys())[0]
        cutouts = cutouts_dic[mass_z_key]
        scale_fac = fg.compton_y_to_delta_Tcmb(145, uK = True)
        tsz_cutouts, ksz_cutouts, tsz_ksz_cutouts  = [], [], []
        for kcntr, keyname in enumerate( cutouts ):
            tsz_cutout = cutouts[keyname]['y']*scale_fac
            tsz_cutouts.append(tsz_cutout)
            ksz_cutout = cutouts[keyname]['ksz']*random.randrange(-1, 2, 2)
            ksz_cutouts.append(ksz_cutout)
            tsz_ksz_cutout = tsz_cutout + ksz_cutout
            tsz_ksz_cutouts.append(tsz_ksz_cutout) 
        s, e = int((nx-40)/2), int((ny+40)/2)
    l, bl = exp.beam_power_spectrum(1.4)
    l, nl = exp.noise_power_spectrum(2.0)
    sims_2e14, sims_6e14, sims_10e14 = [], [], []
    sims_no_shift, sims_clus_position_analysis = [], []
    sims_baseline, sims_tsz, sims_ksz, sims_tsz_ksz = [], [], [], []
    if validation_analyis is True:
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
        for i in range(nber_maps):
            sim = tools.make_gaussian_realization(map_params, l, cl)
            sim_no_shift = lensing.lens_map(map_params, sim, alpha_vec_6e14)
            sim_clus_position_analysis = lensing.lens_map(map_params, sim, alpha_vec_6e14, centroid_shift = 0.5)
            sim_no_shift = tools.convolve(sim_no_shift, l, np.sqrt(bl), map_params = map_params)
            sim_clus_position_analysis = tools.convolve(sim_clus_position_analysis, l, np.sqrt(bl), map_params = map_params)
            noise_map =  tools.make_gaussian_realization(map_params, l, nl)
            sim_no_shift += noise_map
            sim_clus_position_analysis += noise_map
            sims_no_shift.append(sim_no_shift)
            sims_clus_position_analysis.append(sim_clus_position_analysis)
        return sims_no_shift, sims_clus_position_analysis
    if extragal_bias_analysis is True:
        for i in range(nber_maps):
            sim = tools.make_gaussian_realization(map_params, l, cl)
            sim_M200c = lensing.lens_map(map_params, sim, alpha_vec_M200c)
            sim_baseline, sim_tsz, sim_ksz, sim_tsz_ksz = np.copy(sim_M200c), np.copy(sim_M200c), np.copy(sim_M200c), np.copy(sim_M200c)
            tsz_cutout = tools.rotate(tsz_cutouts[random.randint(0, len(tsz_cutouts)-1)], random.randint(-180,180))
            ksz_cutout = tools.rotate(ksz_cutouts[random.randint(0, len(ksz_cutouts)-1)], random.randint(-180,180))
            tsz_ksz_cutout = tools.rotate(tsz_ksz_cutouts[random.randint(0, len(tsz_ksz_cutouts)-1)], random.randint(-180,180))
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