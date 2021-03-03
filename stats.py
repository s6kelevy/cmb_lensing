import numpy as np
import scipy.special
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import emcee


#################################################################################################################################


def covariance_matrix(sims_for_covariance, nber_pixels):
    matrix = np.concatenate(sims_for_covariance)
    matrix = matrix.flatten().reshape(len(sims_for_covariance), nber_pixels)
    cov = np.cov(matrix, rowvar=False) 
    corr = np.corrcoef(matrix, rowvar=False)
    return cov, corr


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


def likelihood_function(data, models, param_int, make_finer = 1):
    x, y, cov = data 
    ln_like = []
    for i in range(len(models)):
        diff = y - models[i]
        chi_2 = np.dot(diff.flatten(), np.linalg.solve(cov, diff.flatten()))
        ln_L =  -0.5*chi_2
        ln_like.append(ln_L)
    ln_like = ln_like - max(ln_like) 
    L = np.exp(ln_like)
    L /= max(L)   
    if make_finer: 
        x_ip, L_ip = likelihood_finer_resol(param_int, L) 
    else:
        x_ip = np.copy(param_int)
        L_ip = np.copy(L)
    L_ip /= max(L_ip) 
    return x_ip, L_ip


def random_sampler(x, y, howmanysamples = 100000, burn_in = 5000):
    norm = integrate.simps(y, x)
    y = y/norm 
    cdf = np.asarray([integrate.simps(y[:i+1], x[:i+1]) for i in range(len(x))])
    cdf_inv = interpolate.interp1d(cdf, x)
    random_sample = cdf_inv(np.random.rand(howmanysamples))
    return random_sample[burn_in:] 


def ml_params(x, likelihood_curve, howmanysamples = 1000000, which_percentile = 16.):
    mean_value = x[np.argmax(likelihood_curve)]
    randoms = random_sampler(x,likelihood_curve, howmanysamples)
    low_err = mean_value - np.percentile(randoms, which_percentile)
    high_err = np.percentile(randoms, 100. - which_percentile) - mean_value
    errors = [low_err, high_err]
    return mean_value, errors


def run_ml(data, models, param_int, make_finer = 1, howmanysamples = 1000000, which_percentile = 16.):
    x, likelihood_curve = likelihood_function(data,  models, param_int)
    likelihood = [x, likelihood_curve]
    mean_value, errors = ml_params(x, likelihood_curve, howmanysamples, which_percentile)
    return likelihood, mean_value, errors


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
    cred_int = [np.percentile(flat_samples[:, i], [16, 84]) for i in range(len(priors))]
    mean_values = [(cred_int[i][0] + cred_int[i][1]) / 2 for i in range(len(priors))]
    errors = [params[i] - cred_int[i][0] for i in range(len(priors))]
    return chains, mean_values, errors