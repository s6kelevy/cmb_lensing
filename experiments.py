import numpy as np
import tools


def get_exp_specs(name = None):
    if name is None or name == 'spt3g':
        specs_dic = {
            #freq: [beam_arcmins, white_noise_T, elknee_T, alphaknee_T] 
            90:  [1.7, 3.0, 1200, -3],
            150: [1.2, 2.2, 2200, -4], 
            220: [1.0, 8.8, 2300, -4],
            }
        corr_noise = 0
        corr_noise_bands = {90:[90], 150:[150], 220: [220]}
        rho = 1
        
    else:    
        if name == 's4wide':
            specs_dic = {
                #freq: [beam_arcmins, white_noise, red_noise, elknee, alphaknee] 
                30:  [7.3, 21.8, 21.8, 471,  -3.5],
                40:  [5.5, 12.4, 12.4, 428,  -3.5], 
                95:  [2.3, 2.0,  2.0,  2154, -3.5],
                145: [1.5, 2.0,  2.0,  4364, -3.5],
                220: [1.0, 6.9,  6.9,  7334, -3.5],
                270: [0.8, 16.7, 16.7, 7308, -3.5],
                }
            corr_noise = 0
            corr_noise_bands = {30:[30], 40:[40], 95:[95], 145:[145], 220: [220], 270: [270]}
            rho = 1
         
        if name == 'ccatp':
            specs_dic = {
                #freq: [beam_arcmins, white_noise, red_noise, elknee, alphaknee] 
                220: [0.95, 14.6,     434.8,     1000, -3.5],
                280: [0.75, 27.5,     1140.2,    1000, -3.5], 
                350: [0.58, 104.8,    5648.8,    1000, -3.5],
                410: [0.50, 376.6,    14174.2,   1000, -3.5],
                }
            corr_noise = 0
            corr_noise_bands = {220: [220], 280:[280], 350:[350], 410:[410]}   
            rho = 1
        
        if name == 'so':
            specs_dic = {
                #freq: [beam_arcmins, white_noise, red_noise, elknee, alphaknee] 
                27:  [7.4, 52.1, 34377.5,  1000, -3.5],
                39:  [5.1, 27.1, 21468.7,  1000, -3.5], 
                93:  [2.2, 5.8,  52136.0,  1000, -3.5],
                145: [1.4, 6.5,  133143.4, 1000, -3.5],
                225: [1.0, 15.0, 448227.3, 1000, -3.5],
                280: [0.9, 37.0, 605277.8, 1000, -3.5],
                }
            corr_noise = 0
            corr_noise_bands = {27:[27], 39:[39], 93:[93], 145:[145], 225: [225], 280: [280]}
            rho = 1
       
    return specs_dic, corr_noise, corr_noise_bands, rho



def beam_power_spectrum(beam_fwhm, l):
    fwhm_radians = np.radians(beam_fwhm/60)
    sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
    bl = np.exp(-1 * l * (l+1) * sigma ** 2)  
    return bl


def noise_power_spectrum(noiseval_white, el, noiseval_red = None, elknee = -1, alphaknee = 0, beam_fwhm = None, noiseval_white2 = None,  noiseval_red2 = None, elknee2 = -1, alphaknee2 = 0, beam_fwhm2 = None, rho = None):
    
    cross_band_noise = 0
    if noiseval_white2 is not None and beamval2 is not None:
        assert rho is not None
        cross_band_noise = 1
    
    # computing white noise power spectrum
    delta_white_radians = np.radians(noiseval_white/60)
    nl = np.tile(delta_white_radians**2, int(max(el)) + 1 )
    nl = np.asarray( [nl[int(l)] for l in el] )
    if cross_band_noise:
        delta_white_radians2 = np.radians(noiseval_white2/60)
        nl2 = np.tile(delta_white_radians2**2., int(max(el)) + 1 )
        nl2 = np.asarray( [nl2[int(l)] for l in el] )
       
    # adding atmospheric noise power spectrum
    if elknee != -1:
        if noiseval_red is None:
            n_red = delta_white_radians**2
        else:
            delta_red_radians = np.radians(noiseval_red/60)
            n_red = delta_red_radians**2 
        nl += n_red*(el/elknee)**alphaknee
        if cross_band_noise and elknee2 != -1.:
            if noiseval_red2 is None:
                n_red2 = delta_white_radians2**2
            else:
                delta_red_radians2 = np.radians(noiseval_red2/60)
                n_red2 = delta_red_radians2**2  
            nl2 += n_red2*(el/elknee2)**alphaknee2
      
    # deconvolving noise power spectrum
    if beam_fwhm is not None:
        bl = beam_power_spectrum(beam_fwhm, el)
        nl *= bl**(-1)
        if cross_band_noise and beam_fwhm2 is not None: 
            bl2 =  beam_power_spectrum(beam_fwhm2, el)
            nl2 *= bl2**(-1)

    if cross_band_noise:
        nl = rho * (nl * nl2)**0.5
    
    return nl
