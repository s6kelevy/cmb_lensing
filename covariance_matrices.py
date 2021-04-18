%load_ext autoreload 
%autoreload 2
import numpy as np
import random
from colossus.cosmology import cosmology
cosmology.setCosmology('planck18')
from colossus.halo import concentration, mass_defs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from pylab import rcParams
from matplotlib import rc;rc('text', usetex=True);rc('font', weight='bold');matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
rcParams['font.family'] = 'serif'
rc('text.latex',preamble=r'\usepackage{/Volumes/Extreme_SSD/codes/master_thesis/code/configs/apjfonts}')
sz_ft = 20
sz_lb = 14
color_arr = ['indigo', 'royalblue', 'lightseagreen', 'darkgreen', 'goldenrod', 'darkred']
from tqdm import tqdm
import cosmo
import lensing_estimator
from cosmo import CosmoCalc
import lensing
import foregrounds as fg
import experiments as exp
import sims
import stats
import tools


##################################################################################################################################


nber_clus = 3000
nber_rand = 50000
nber_cov = 1000
nber_runs = 25
map_params = [180, 0.5, 180, 0.5]
l, cl = CosmoCalc().cmb_power_spectrum()
l, bl = exp.beam_power_spectrum(beam_fwhm = 1.0)
noiseval_arr = [0.1, 0.25, 0.5, 1, 2, 3, 5]
nl_arr = []
cl_noise_arr = []
for noiseval in noiseval_arr:
    l, nl = exp.white_noise_power_spectrum(noiseval_white = noiseval)
    l, nl_deconvolved =exp.white_noise_power_spectrum(noiseval_white = noiseval, beam_fwhm = 1.0)
    nl_arr.append(nl)
    cl_noise_arr.append(nl_deconvolved)
z = 0.7


c500 = concentration.concentration(2e14, '500c', 0.7)
M200c, _, _ = mass_defs.changeMassDefinition(2e14, c500, 0.7, '500c', '200c', profile='nfw')
cluster = [M200c, cosmo.concentration_parameter(M200c, 0.7, 0.674), 0.7] 
fname = '/mdpl2_cutouts_for_tszksz_clus_detection_M1.7e+14to2.3e+14_z0.6to0.8_15320haloes_boxsize10.0am_dx0.5am.npz'
cutouts_dic = np.load(fname, allow_pickle = 1, encoding= 'latin1')['arr_0'].item()
mass_z_key = list(cutouts_dic.keys())[0]
cutouts = cutouts_dic[mass_z_key]
scale_fac = fg.compton_y_to_delta_Tcmb(freq = 150, uK = True)
tsz_cutouts, ksz_cutouts, tsz_ksz_cutouts = [], [], []
for kcntr, keyname in enumerate( cutouts ):
    tsz_cutout = cutouts[keyname]['y']*scale_fac
    tsz_cutouts.append(tsz_cutout)
    ksz_cutout = cutouts[keyname]['ksz']*random.randrange(-1, 2, 2)
    ksz_cutouts.append(ksz_cutout)
    tsz_ksz_cutout = tsz_cutout + ksz_cutout
    tsz_ksz_cutouts.append(tsz_ksz_cutout) 
mass_int = np.arange(0, 4, 0.1) 


##################################################################################################################################


covariance_matrix_tsz, _ = lensing_estimator.covariance_and_correlation_matrix(nber_cov, nber_clus, map_params, l, cl, cluster_corr_cutouts =  tsz_cutouts, bl = bl, nl = nl_arr[4], cl_noise = cl_noise_arr[4], correct_for_tsz = False)
np.save('/Volumes/Extreme_SSD/codes/master_thesis/code/results/covariance_matrix_tsz.npy', covariance_matrix_tsz)
    
covariance_matrix_tsz_corrected, _ = lensing_estimator.covariance_and_correlation_matrix(nber_cov, nber_clus, map_params, l, cl, cluster_corr_cutouts =  tsz_cutouts, bl = bl, nl = nl_arr[4], cl_noise = cl_noise_arr[4], correct_for_tsz = True)
np.save('/Volumes/Extreme_SSD/codes/master_thesis/code/results/covariance_matrix_tsz_corrected.npy', covariance_matrix_tsz_corrected)

covariance_matrix_ksz, _ = lensing_estimator.covariance_and_correlation_matrix(nber_cov, nber_clus, map_params, l, cl, cluster_corr_cutouts =  ksz_cutouts, bl = bl, nl = nl_arr[4], cl_noise = cl_noise_arr[4], correct_for_tsz = False)
np.save('/Volumes/Extreme_SSD/codes/master_thesis/code/results/covariance_matrix_ksz.npy', covariance_matrix_ksz)
    
covariance_matrix_tsz_ksz, _ = lensing_estimator.covariance_and_correlation_matrix(nber_cov, nber_clus, map_params, l, cl, cluster_corr_cutouts =  tsz_ksz_cutouts, bl = bl, nl = nl_arr[4], cl_noise = cl_noise_arr[4], correct_for_tsz = False)
np.save('/Volumes/Extreme_SSD/codes/master_thesis/code/results/covariance_matrix_tsz_ksz.npy', covariance_matrix_tsz_ksz)
    
covariance_matrix_tsz_ksz_corrected, _ = lensing_estimator.covariance_and_correlation_matrix(nber_cov, nber_clus, map_params, l, cl, cluster_corr_cutouts =  tsz_ksz_cutouts, bl = bl, nl = nl_arr[4], cl_noise = cl_noise_arr[4], correct_for_tsz = True)
np.save('/Volumes/Extreme_SSD/codes/master_thesis/code/results/covariance_matrix_tsz_ksz_corrected.npy', covariance_matrix_tsz_ksz_corrected)    