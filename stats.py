# importing relevant modules
import numpy as np
import scipy.special
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import emcee


#################################################################################################################################


def jackknife_resampling(data, noofsims):
    total = len(data)
    each_split_should_contain = int(total * 1./noofsims)
    fullarr = np.arange(total)
    inds_to_pick = np.arange(len(fullarr))
    already_picked_inds = []
    jk_samples = []
    for n in range(noofsims):
        inds = np.random.choice(inds_to_pick, size = each_split_should_contain, replace = 0)
        inds_to_delete = np.where (np.in1d(inds_to_pick, inds) == True)[0]
        inds_to_pick = np.delete(inds_to_pick, inds_to_delete)
        tmp = np.in1d(fullarr, inds)
        non_inds = np.where(tmp == False)[0]
        jk_samples.append( (non_inds) )
     
    resamples = []
    for i in range(noofsims):
        resample = [data[jk_samples[i][j]] for j in range(total-each_split_should_contain)]
        resamples.append(resample)
    return resamples, jk_samples


#################################################################################################################################


def covariance_and_correlation_matrix(sample):
    
    matrix = np.concatenate(sample)
    if sample[0].ndim == 1:
        nber_data_points = sample[0].shape[0]
    else:
        nber_data_points = sample[0].shape[0]*sample[0].shape[1]
    matrix = matrix.flatten().reshape(len(sample), nber_data_points)
    covariance_matrix = np.cov(matrix, rowvar=False) 
    correlation_matrix = np.corrcoef(matrix, rowvar=False)
    
    return covariance_matrix, correlation_matrix


#################################################################################################################################


def fitting_func_gaussian(p, p0, X, DATA = None, return_fit = 0):
    fitfunc = lambda p, x: p[1]*(np.exp(-(x-p[2])**2/(2*p[3]**2)))
    if not return_fit:
        return fitfunc(p, X) - DATA
    else:
        return fitfunc(p, X)


def likelihood_finer_resol(M, L, delta=0.001):
    M_ip = np.arange(min(M),max(M),delta) 
    best_fit = M[np.argmax(L)]
    gau_width = abs(best_fit - M[np.argmin(abs(L))])/2.35 * 2.
    p0 = [0.,np.max(L),best_fit,gau_width]
    p1, success = optimize.leastsq(fitting_func_gaussian, p0, args=(p0, M, L))
    L_ip = fitting_func_gaussian(p1, p1, M_ip, return_fit = 1)
    return M_ip, L_ip


def likelihood_function(data, models, param_int, make_finer = 1, normalize = True):
    x, y, cov = data 
    ln_like = []
    for i in range(len(models)):
        diff = (y - models[i]).flatten()
        chi_2 = np.dot(diff.T, np.dot(np.linalg.pinv(cov), diff))#np.dot(diff.flatten(), np.linalg.solve(cov, diff.flatten()))
        ln_L =  -0.5*chi_2
        ln_like.append(ln_L)
    ln_like = ln_like - max(ln_like) 
    L = np.exp(ln_like)
    if normalize is True:
        L /= max(L)   
    if make_finer: 
        x_ip, L_ip = likelihood_finer_resol(param_int, L) 
    else:
        x_ip = np.copy(param_int)
        L_ip = np.copy(L)
    if normalize is True:    
        L_ip /= max(L_ip) 
    return x_ip, L_ip


def random_sampler(x, y, nsamples = 100000, burn_in = 5000):
    norm = integrate.simps(y, x)
    y = y/norm 
    cdf = np.asarray([integrate.simps(y[:i+1], x[:i+1]) for i in range(len(x))])
    cdf_inv = interpolate.interp1d(cdf, x)
    random_sample = cdf_inv(np.random.rand(nsamples))
    return random_sample[burn_in:] 


def ml_params(x, likelihood_curve, nsamples = 1000000, burn_in = 5000):
    sample = random_sampler(x, likelihood_curve, nsamples = nsamples, burn_in = burn_in)
    values = np.percentile(sample, [16, 50, 84])
    median_value = values[1]
    low_error = median_value - values[0]
    high_error = values[2] - median_value
    error = (high_error+low_error)/2
    return median_value, error


def run_ml(data, models, param_int, make_finer = 1, normalize = True, nsamples = 1000000, burn_in = 5000):
    x, likelihood_curve = likelihood_function(data,  models, param_int, make_finer = make_finer, normalize = normalize)
    likelihood = [x, likelihood_curve]
    median_value, error = ml_params(x, likelihood_curve, nsamples = nsamples, burn_in = burn_in)
    return likelihood, median_value, error


#################################################################################################################################


def ln_prior(params, priors):
    for i in range(len(priors)):
        if priors[i][0] < params[i] <  priors[i][1]:
            return 0.0
    return -np.inf
       
    
def ln_likelihood(params, x, y, cov):
    model = f(params, x)
    diff = y - model
    chi_2 = np.dot(diff, np.linalg.solve(cov, diff))
    return -0.5*chi_2


def ln_probability(params, x, y, yerr, priors):
    lp = ln_prior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(params, x, y, yerr)


def run_mcmc(data, guess, offset, nwalkers, priors, nber_steps, burn_in):
    x, y, cov = data 
    ndim = len(guess)
    pos = guess + offset * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_probability, args=(x, y, cov, priors))
    sampler.run_mcmc(pos, nber_steps, progress=True) 
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    chains = [samples, flat_samples]
    values = [np.percentile(flat_samples[:, i], [16, 50, 84]) for i in range(len(priors))]
    median_values = [values[i][1] for i in range(len(priors))] 
    low_errors = [(median_values[i] -values[i][0]) for i in range(len(priors))]
    high_errors = [(median_values[i] -values[i][2]) for i in range(len(priors))]
    errors = [(high_errors[i]-low_errors[i])/2 for i in range(len(priors))]
    return chains, mean_values, errors


#################################################################################################################################


def combined_likelihood(x, likelihood_arr, normalize = True):
    comb_lk = np.ones(len(likelihood_arr[0]))
    for i in range(len(likelihood_arr)):
        comb_lk *= likelihood_arr[i]
    if normalize is True:
        comb_lk = comb_lk/max(comb_lk)
    median_value, error = ml_params(x, comb_lk)      
    return comb_lk, median_value, error


def signal_to_noise(pdf):
    lnpdf = np.log(pdf)
    delta_chisq = 2*(max(lnpdf) - lnpdf[0])
    snr = np.sqrt(delta_chisq)
    return snr