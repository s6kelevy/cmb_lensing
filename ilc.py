import numpy as np
import foregrounds as fg
import experiments as exp
import tools


def power_spectra_dic(map_dic = None, map_params = None, components = 'all', experiment = None):
    
    if map_dic is None:
        specs_dic, corr_noise_bands, rho = exp.specs(experiment)
        freq_arr = sorted(specs_dic.keys())
        l, nl_dic = exp.noise_power_spectra_dic(experiment, deconvolve = True, use_cross_power = True)
        l, cl_extragal_dic = fg.extragalactic_power_spectrum_dic(freq_arr, components = 'all', use_cross_power = True)
        cl_dic = {}
        for freq1 in freq_arr:
            for freq2 in freq_arr:
                cl_extragal = cl_extragal_dic[(freq1,freq2)]
                nl = nl_dic[(freq1,freq2)]
                # accounting for the case where beam deconvolution has made end nl pretty large  
                ini_nl = np.median(nl[:100])
                end_nl = np.median(nl[-100:])
                if end_nl>ini_nl:             
                        badinds = np.where(nl>=5000)[0]
                        nl[badinds] = 5000
                # computing total power spectra 
                cl_noise = cl_extragal + nl  
                cl_noise[np.isinf(cl_noise)] = 0.
                cl_dic[(freq1, freq2)] = cl_noise       
    else:
        freqarr = sorted(map_dic.keys())
        cl_dic = {}
        for cntr1, freq1 in enumerate( freqarr ):
            for cntr2, freq2 in enumerate( freqarr ):
                map1, map2 = map_dic[freq1], map_dic[freq2]
                l, cl = tools.power_spectra(mapparams, map1, map2 = None, binsize = None)
                cl = np.concatenate( (np.zeros(lmin), cl) )
                cl[np.isnan(cl)] = 0.
                cl[np.isinf(cl)] = 0.
                cl_dic[(freq1, freq2)] = cl       
    
    return l, cl_dic



'''
def power_spectra_dic(map_dic = None, mapparams = None, components = 'all', experiment = None):
    if map_dic is None:
        # creating total power spectra dictionary
        specs_dic, corr_noise_bands, rho = exp.specs(experiment)
        freq_arr = sorted( specs_dic.keys() )
        l, nl_dic = exp.noise_power_spectra_dic(experiment, deconvolve = True, use_cross_power = True)
        cl_dic = {}
        for freq1 in freq_arr:
            for freq2 in freq_arr:
                # obtaining extragalactic foreground power spectrum
                l, cl_extragal = fg.extragalactic_power_spectrum(freq1, freq2, components)
            
                # obtaining deconvolved noise power spectra
                if (freq1,freq2) in nl_dic:
                    nl = nl_dic[(freq1,freq2)]
                else:
                    nl = nl_dic[freq1]
                    if freq1 != freq2: 
                        nl = np.copy(nl) * 0.
           
                # accounting for the case where beam deconvolution has made end nl pretty large  
                ini_nl = np.median(nl[:100])
                end_nl = np.median(nl[-100:])
                if end_nl>ini_nl:             
                        badinds = np.where(nl>=5000)[0]
                        nl[badinds] = 5000
            
                # computing total power spectra and adding it to dictionary
                cl_noise = cl_extragal + nl    
                cl_noise[np.isnan(cl_noise)] = 0.
                cl_dic[(freq1, freq2)] = cl_noise       
    else:
        freqarr = sorted(map_dic.keys())
        cl_dic = {}
        for cntr1, freq1 in enumerate( freqarr ):
            for cntr2, freq2 in enumerate( freqarr ):
                map1, map2 = map_dic[freq1], map_dic[freq2]
                l, cl = tools.power_spectra(mapparams, map1, map2 = None, binsize = None)
                cl = np.concatenate( (np.zeros(lmin), cl) )
                cl[np.isnan(cl)] = 0.
                cl[np.isinf(cl)] = 0.
                cl_dic[(freq1, freq2)] = cl       
    return l, cl_dic

'''

def create_clmat(freqarr, elcnt, cl_dic):
    
    nc = len(freqarr)
    teb_len, pspec_arr = 1, ['TT']
    clmat = np.zeros( (teb_len * nc, teb_len * nc) )

    for pspecind, pspec in enumerate( pspec_arr ):
        curr_cl_dic = cl_dic
        for ncnt1, freq1 in enumerate(freqarr):
            for ncnt2, freq2 in enumerate(freqarr):
                j, i = ncnt2, ncnt1
                clmat[j, i] = curr_cl_dic[(freq1, freq2)][elcnt]
    clmat = np.mat(clmat)
   
    return clmat 


def get_clinv(clmat): 
    
    clinv = np.linalg.pinv(clmat)
    
    return clinv


def residuals_and_weights(map_dic = None, mapparams = None, components = 'all', experiment = None, cov_from_sims = True): 
    
    if cov_from_sims is True:
        specs_dic, corr_noise_bands, rho = exp.specs(experiment)
        freq_arr = sorted(specs_dic.keys() )   
        l, cl_dic = power_spectra_dic(components = components, experiment = experiment)
    else:
        freq_arr = sorted(map_dic.keys())
        l, cl_dic = power_spectra_dic(map_dic, mapparams) 
    a_cmb = np.mat(np.ones(len(freq_arr))).T 
    cl_residual = np.zeros(len(l))
    weights = np.zeros((len(freq_arr), 1, len(l)))
    for i in range(len(l)):
        clmat = create_clmat(freq_arr, i, cl_dic) 
        clinv = get_clinv(clmat)
        num = np.dot(clinv, a_cmb) 
        den = np.dot(a_cmb.T, np.dot(clinv, a_cmb))
        cl_residual[i] = np.linalg.pinv(den)
        weights[:, :,i] = np.dot(num, cl_residual[i])  
      
    cl_residual = np.asarray(cl_residual)
    cl_residual[np.isinf(cl_residual)] = 0.
    cl_residual[np.isnan(cl_residual)] = 0.
    weights_arr = weights[:len(freq_arr), 0]
 
   # calculate individual residual powers
    if components == 'all':
        components = ['radio', 'cib', 'tsz', 'ksz', 'tsz_cib']
    cl_radio_dic, cl_cib_dic, cl_tsz_dic, cl_ksz_dic, cl_tsz_cib_dic = {}, {}, {}, {}, {}
    for freq1 in freq_arr:
        for freq2 in freq_arr:
            if 'radio' in components:
                l, cl_radio = fg.extragalactic_power_spectrum(freq1, freq2, components = ['radio'])
                cl_radio_dic[(freq1, freq2)] = cl_radio_dic[(freq2, freq1)] = cl_radio 
  
            if 'cib' in components:
                l, cl_cib = fg.extragalactic_power_spectrum(freq1, freq2, components = ['cib'])
                cl_cib_dic[(freq1, freq2)] = cl_cib_dic[(freq2, freq1)]  = cl_cib
            
            if 'tsz' in components:
                l, cl_tsz = fg.extragalactic_power_spectrum(freq1, freq2, components = ['tsz'])
                cl_tsz_dic[(freq1, freq2)] = cl_tsz_dic[(freq2, freq1)] = cl_tsz
           
            if 'ksz' in components:
                l, cl_ksz = fg.extragalactic_power_spectrum(freq1, freq2, components = ['ksz'])
                cl_ksz_dic[(freq1, freq2)] = cl_ksz_dic[(freq2, freq1)] = cl_ksz
      
            if 'tsz_cib' in components:
                l, cl_tsz_cib = fg.extragalactic_power_spectrum(freq1, freq2, components = ['tsz_cib'])
                cl_tsz_cib_dic[(freq1, freq2)] = cl_tsz_cib_dic[(freq2, freq1)] = cl_tsz_cib 
    
    l, nl_dic = exp.noise_power_spectra_dic(experiment, deconvolve = True, use_cross_power = True)   
    signal_arr = components + ['noise']    
    res_ilc_dic = {}
    for i in range(len(l)):
        for s in signal_arr:
            if s == 'radio':
                curr_cl_dic = cl_radio_dic  
            elif s == 'cib':
                curr_cl_dic = cl_cib_dic
            elif s == 'tsz':
                curr_cl_dic = cl_tsz_dic  
            elif s == 'ksz':
                curr_cl_dic = cl_ksz_dic  
            elif s == 'tsz_cib':
                curr_cl_dic = cl_tsz_cib_dic
            elif s == 'noise':
                curr_cl_dic = nl_dic

            clmat = create_clmat(freq_arr, i, curr_cl_dic) 
            currw_ilc = np.mat(weights_arr[:, i])
            curr_res_ilc = np.asarray(np.dot(currw_ilc, np.dot(clmat, currw_ilc.T)))[0][0]
              
            if s not in res_ilc_dic:
                res_ilc_dic[s] = []
            res_ilc_dic[s].append( curr_res_ilc )
    
    residual_and_weights = [l, cl_residual, res_ilc_dic, weights_arr]
        
    return residual_and_weights


def ilc_map(map_dic, opbeam, mapparams, experiment, components = 'all', cov_from_sims = True):
    
    # collecting multifrequency maps
    freq_arr = sorted(map_dic.keys())
    map_arr = []
    for freq in freq_arr:
        curr_map = map_dic[freq]
        map_arr.append(curr_map)
       
    
    # obtaining weights     
    res_weights = residuals_and_weights(map_dic, mapparams, components, experiment, cov_from_sims) 
    l, cl_residual, res_ilc_dic, weights_arr = res_weights
   
    # rebeaming
    l, bl_dic = exp.beam_power_spectrum_dic(experiment, opbeam)
    bl_rebeam_arr = exp.rebeam(bl_dic)

    # computing 2D versions 
    grid, _ = tools.make_grid(mapparams, harmonic = True)
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


    # compute ilc map 
    weighted_maps_arr = []
    for mm in range(len(map_arr)):
        curr_map = map_arr[mm]
        rebeamed_weights_arr[mm][np.isnan(rebeamed_weights_arr[mm])]=0.
        rebeamed_weights_arr[mm][np.isinf(rebeamed_weights_arr[mm])]=0.
        map_weighted = np.fft.fft2(curr_map) * rebeamed_weights_arr[mm]
        weighted_maps_arr.append(map_weighted)
    ilc_map_fft = np.sum(weighted_maps_arr, axis = 0)
    ilc_map = np.fft.ifft2(ilc_map_fft).real

    return ilc_map, res_weights

