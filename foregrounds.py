# importing relevant modules
import numpy as np
import random
from scipy.io import readsav
import tools


#################################################################################################################################


# defining relevant constants
c = 3e8  # speed of light in m/s
h = 6.626e-34
k_B = 1.38e-23
T_cmb = 2.72548  # CMB temperature in K
T_cib = 20  # CIB temperature in K
l_norm = 3000
freq0 = 150


#################################################################################################################################


def fn_dB_dT(freq):
    x = (h * freq * 1e9) / (k_B * T_cmb)
    dBdT = x**4. * np.exp(x) / (np.exp(x)-1)**2.
    return dBdT  
    
    
def fn_BnuT(freq, temp):
    x = (h * freq * 1e9) / (k_B * temp)
    t1 = 2 * h * freq**3/ c**2
    t2 = 1./ (np.exp(x)-1.)
    return t1 * t2


def tsz_spec(freq):    
    x = (h * freq * 1e9) / (k_B * T_cmb)
    g_nu = x * (np.exp(x/2) + np.exp(-x/2)) / (np.exp(x/2) - np.exp(-x/2)) - 4.
    return np.mean(g_nu) 


def compton_y_to_delta_Tcmb(freq, uK = False):
    g_nu = tsz_spec(freq)
    scale_fact = g_nu*T_cmb
    if uK is True:
        scale_fact = scale_fact*1e6
    return scale_fact

       #################################################################################################################################


def get_foreground_power_spt(component):
    components = ['tSZ', 'DG-Cl','DG-Po','RG', 'kSZ']
    filename = 'sim_data/george_plot_bestfit_line.sav'
    data = readsav(filename)
    freqs = np.asarray([(95, 95), (95, 150), (95, 220), (150, 150), (150, 220), (220, 220)])
    dl_all = data['ml_dls'][(freqs[:, 0] == 150) & (freqs[:, 1] == freq0)][0]
    labels = data['ml_dl_labels'].astype('str')
    el = np.asarray(data['ml_l'], dtype=int)
    spec = dl_all[labels == component][0]

    # pad to l=0
    spec = np.concatenate((np.zeros(min(el)), spec))
    el = np.concatenate((np.arange(min(el)), el))

    el = el[:10000]
    spec = spec[:10000]

    return el, spec


def get_cl_radio(freq1, freq2 = None):
    
    if freq2 is None:
        freq2 = freq1
        
    l, dl_rg_freq0 = get_foreground_power_spt('RG')
    epsilon_nu1_nu2 =  (fn_dB_dT(freq0)**2)/(fn_dB_dT(freq1) * fn_dB_dT(freq2))

    dl_rg =  dl_rg_freq0[l == l_norm][0] * epsilon_nu1_nu2 * ((freq1 * freq2)/(freq0**2))**(-0.9) * (l/l_norm)**2
    cl_rg = (2*np.pi)/(l * (l+1))*dl_rg
    cl_rg[np.isnan(cl_rg)] = 0.
    if (freq1>230 or freq2>230):
        cl_rg *= 0.

    return l, cl_rg


def get_cl_cib(freq1, freq2 = None):
    
    if freq2 is None:
        freq2 = freq1
     
    l, dl_po_freq0 = get_foreground_power_spt('DG-Po')
    l, dl_clus_freq0 = get_foreground_power_spt('DG-Cl')
    epsilon_nu1_nu2 = (( fn_dB_dT(freq0) )**2.)/(fn_dB_dT(freq1) * fn_dB_dT(freq2))
    bnu1 = fn_BnuT(freq1, temp = T_cib)
    bnu2 = fn_BnuT(freq2, temp = T_cib)
    bnu0 = fn_BnuT(freq0, temp = T_cib)
    etanu1_po = ((freq1*1e9)**1.505) * bnu1
    etanu2_po = ((freq2*1e9)**1.505) * bnu2
    etanu0_po = ((freq0*1e9)**1.505) * bnu0
    etanu1_clus = ((freq1*1e9)**2.51) * bnu1
    etanu2_clus = ((freq2*1e9)**2.51) * bnu2
    etanu0_clus = ((freq0*1e9)**2.51) * bnu0

    
    dl_po = dl_po_freq0[l == l_norm][0] * epsilon_nu1_nu2 * ((etanu1_po * etanu2_po)/(etanu0_po**2)) * (l/l_norm)**2
    dl_clus = dl_clus_freq0[l == l_norm][0] * epsilon_nu1_nu2 * ((etanu1_clus * etanu2_clus)/(etanu0_clus**2)) * (l/l_norm)**0.8
    dl_cib = dl_po + dl_clus
    cl_cib =  (2*np.pi)/(l * (l+1)) * dl_cib 
    cl_cib[np.isnan(cl_cib)] = 0.
    cl_cib[np.isinf(cl_cib)] = 0.

    return l, cl_cib


def get_cl_tsz(freq1, freq2 = None):
    l, dl_tsz_freq0 = get_foreground_power_spt('tSZ')

    tsz_fac_freq0 = tsz_spec(freq0)
    tsz_fac_freq1 = tsz_spec(freq1)
    tsz_fac_freq2 = tsz_spec(freq2)


    dl_tsz = dl_tsz_freq0 * tsz_fac_freq1 * tsz_fac_freq2/ (tsz_fac_freq0**2.)
    cl_tsz =  (2*np.pi)/(l * (l+1)) * dl_tsz
    cl_tsz[np.isnan(cl_tsz)] = 0.
    cl_tsz[np.isinf(cl_tsz)] = 0.

    return l, cl_tsz


def get_cl_ksz():
    
    l, dl_ksz_freq0 = get_foreground_power_spt('kSZ')
    cl_ksz =  (2*np.pi)/(l * (l+1)) * dl_ksz_freq0
    cl_ksz[np.isnan(cl_ksz)] = 0.
    cl_ksz[np.isinf(cl_ksz)] = 0.
    
    return l, cl_ksz


def get_cl_tsz_cib(freq1, freq2 = None):
    
    if freq2 is None:
        freq2 = freq1
      
    #get tSZ and CIB spectra for freq1
    l, cl_tsz_freq1_freq1 = get_cl_tsz(freq1, freq1)
    l, cl_cib_freq1_freq1 = get_cl_cib(freq1, freq1)
    
 
    #get tSZ and CIB spectra for freq2
    l, cl_tsz_freq2_freq2 = get_cl_tsz(freq2, freq2)
    l, cl_cib_freq2_freq2 = get_cl_cib(freq2, freq2)
    
    cl_tsz_cib =  - 0.1 * ( np.sqrt(cl_tsz_freq1_freq1 * cl_cib_freq2_freq2) + np.sqrt(cl_tsz_freq2_freq2 * cl_cib_freq1_freq1) )

    return l, cl_tsz_cib


def extragalactic_power_spectrum(freq, freq2 = None, components = 'all'):
    
    if freq2 is None:
        freq2 = freq
    
    if components == 'all':
        components = ['radio', 'cib', 'tsz', 'ksz', 'tsz_cib']
  
    cl_extragal = np.zeros(10000)
    if 'radio' in components:
        l, cl_radio = get_cl_radio(freq, freq2)
        cl_extragal += cl_radio
    if 'cib' in components:   
        l,  cl_cib = get_cl_cib(freq, freq2)
        cl_extragal += cl_cib
    if 'tsz' in components:   
        l,  cl_tsz = get_cl_tsz(freq, freq2)
        cl_extragal += cl_tsz
    if 'ksz' in components:   
        l,  cl_ksz = get_cl_ksz()
        cl_extragal += cl_ksz    
    if 'tsz_cib' in components:   
        l,  cl_tsz_cib = get_cl_tsz_cib(freq, freq2)
        cl_extragal += cl_tsz_cib
    
    return l, cl_extragal


def extragalactic_power_spectrum_dict(freq_arr, components = 'all'):
    cl_extragal_dic = {}
    for freq in freq_arr:
        l, cl_extragal = extragalactic_power_spectrum(freq, components = components)
        cl_extragal_dic[freq] = cl_extragal
    return l, cl_extragal_dic



  