# importing relevant modules
import numpy as np
import scipy.ndimage


#################################################################################################################################


def make_grid(map_params, Fourier = False):    

    # creating 1D x and y arrays and their limits
    nx, dx, ny, dy = map_params
    if Fourier is False:
        x_min, x_max = -nx*dx/2, nx*dx/2 # in arcmins
        y_min, y_max = -ny*dy/2, ny*dy/2 # in arcmins
        x, y = np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny) # in arcmins
    else:
        dx_rad, dy_rad = np.radians(dx/60.), np.radians(dy/60.)
        x, y = np.fft.fftfreq(nx, dx_rad), np.fft.fftfreq(ny, dy_rad)
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        x, y = 2*np.pi*x, 2*np.pi*y
        x_min, x_max = 2*np.pi*x_min, 2*np.pi*x_max
        y_min, y_max = 2*np.pi*y_min, 2*np.pi*y_max    
    
    # creating coordinate matrices based on map parameters
    grid = np.meshgrid(x, y)
    
    # defining the bounding box that the image will fill
    extent = [x_min, x_max, y_min, y_max]
    
    return grid, extent


def convert_to_2d(grid, x, signal_1d):
      
    X, Y = grid[0], grid[1]
    R = np.hypot(X, Y)
    signal_2d = np.interp(R.flatten(), x, signal_1d, right = 0).reshape(R.shape) 
    
    return signal_2d


def convolve(signal, l, kernel, map_params = None):
    
    signal_fft = np.fft.fft2(signal)
    if map_params is not None:
        grid, _ = make_grid(map_params, Fourier = True)
        kernel = convert_to_2d(grid, l, kernel) 
        kernel[np.isnan(kernel)] = 0
    signal_convolved = np.fft.ifft2(kernel*signal_fft).real 
    
    return signal_convolved


def wiener_filter(l, psd_signal, psd_noise):

    wiener_filter = psd_signal/(psd_signal+psd_noise)
    wiener_filter[np.isnan(wiener_filter)] = 0

    return wiener_filter


def low_pass_filter(l, l_cut):

    low_pass_filter = np.ones(len(l))
    low_pass_filter[np.where(l.astype(int)>l_cut)] = 0
    
    return low_pass_filter
    
    
def gradient(signal, dx = 1):
   
    gradient_map = np.nan_to_num(np.gradient(signal, dx))
    gradient_xmap, gradient_ymap = gradient_map[1], gradient_map[0]
    magnitude_map = np.hypot(gradient_xmap, gradient_ymap) 
    orientation_map = np.degrees(np.arctan2(gradient_ymap, gradient_xmap))
    
    return gradient_xmap, gradient_ymap, magnitude_map, orientation_map


def rotate(image, angle):
   
    rotated_map = scipy.ndimage.rotate(np.nan_to_num(image), np.nan_to_num(angle), reshape = False, mode = 'reflect') 
    
    return rotated_map


def central_cutout(map_params, image, size):

    nx, dx, _, _ = map_params
    nber_pixels = int(size/dx)
    s, e = int((nx-nber_pixels)/2), int((nx+nber_pixels)/2)
    cutout = image[s:e, s:e]     
    
    return cutout


def radial_profile(image, grid = None, bin_min = 0, bin_max = 10, bin_size = 1):
    
    # obtaining the different radii based on the underlying grid of the image
    if grid is None:
        y, x = np.indices((image.shape))
        center = np.array([(x.max()-x.min())/2, (x.max()-x.min())/2])
        radius = np.hypot(x-center[0], y-center[1])
    else:
        x, y = grid
        radius = np.hypot(x, y)
        
    # constructing the radial profile of the image
    bins = np.arange(bin_min, bin_max, bin_size)
    bin_ctr, rad_prf, std = np.zeros(len(bins)), np.zeros(len(bins)), np.zeros(len(bins))
    hit_count = []
    for bin_nbr, bin_val in enumerate(bins):
        ind = np.where((radius >= bin_val)&(radius<bin_val+bin_size))
        bin_ctr[bin_nbr] = (bin_val+bin_size/2)
        hits = len(np.where(np.abs(image[ind])>0)[0])
        if hits>0:
            rad_prf[bin_nbr] = np.sum(image[ind])/hits
             
    return bin_ctr, rad_prf


def make_gaussian_realization(map_params, l, psd):  
   
    # creating a random Gaussian realization and its Fourier transform
    nx, dx, ny, dy = map_params
    dx_rad, dy_rad = np.radians(dx/60), np.radians(dy/60)
    gauss_map = np.random.randn(nx, ny)/np.sqrt(dx_rad * dy_rad)
    
    # convoling Gaussian realization with given power spectra
    if isinstance(psd, list) is False:
        is_list = False
        psd = [psd]
    else:
        is_list = True
    gauss_realizations_arr = []
    for power_spectrum in psd:
        gauss_realization = convolve(gauss_map, l, np.sqrt(power_spectrum), map_params = map_params)
        gauss_realization -= np.mean(gauss_realization)
        gauss_realizations_arr.append(gauss_realization)
    
    if is_list is False:
        return gauss_realizations_arr[0]
     
    return gauss_realizations_arr


def power_spectra(map_params, image, image2 = None, binsize = None):
    
    nx, dx, ny, dy = map_params
    dx_rad = np.radians(dx/60.)

    grid, _ = make_grid(map_params, Fourier = True)
    lx, ly = grid
    if binsize == None:
        bin_size = lx.ravel()[1] -lx.ravel()[0]

    if image2 is None:
        image_psd = abs( np.fft.fft2(image) * dx_rad)** 2 / (nx * ny)
    else: 
        assert image.shape == image2.shape
        image_psd = np.fft.fft2(image) * dx_rad * np.conj( np.fft.fft2(image2) ) * dx_rad / (nx * ny)

    l, psd = radial_profile(image_psd, grid, bin_min = 0, bin_max = 10000, bin_size = bin_size)

    return l, psd