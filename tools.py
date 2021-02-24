# importing relevant modules
import numpy as np
import scipy.ndimage


#################################################################################################################################


def make_grid(mapparams, Fourier = False):    
    # Creates coordinate arrays for vectorized evaluations over grids.
    import matplotlib.pyplot as plt

    # creating 1d x and y arrays and their limits
    nx, dx, ny, dy = mapparams
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


def interpolate_to_2d(grid, x, y):
    # Converts a 1D signal to a 2D signal.
    X, Y = grid[0], grid[1]
    R = np.hypot(X, Y)
    interp_to_2d = np.interp(R.flatten(), x, y).reshape(R.shape) 
    return interp_to_2d


def make_gaussian_realization(mapparams, l, psd):  
    # Creates a Gaussian relalisation of the input power spectrum. 
  

    # computing the 2D Fourier amplitudes
    nx, dx, ny, dy = mapparams
    dx_rad, dy_rad = np.radians(dx/60.), np.radians(dy/60.)
    grid, _ = make_grid(mapparams, Fourier = True)
    psd2d = interpolate_to_2d(grid, l, psd)
    psd2d_sqrt_norm = np.sqrt(psd2d)/np.sqrt(dx_rad * dy_rad)
    psd2d_sqrt_norm[np.isnan(psd2d_sqrt_norm)] = 0.     
       
    # creating a random Gaussian realisation and its Fourier transform
    gauss_map = np.random.randn(nx, ny)
    gauss_map_fft = np.fft.fft2(gauss_map)
    
    
    # creating the flat-sky map               
    sim = np.fft.ifft2(psd2d_sqrt_norm*gauss_map_fft).real   
    sim = sim - np.mean(sim)
    return sim


def gaussian_filter(image, mapparams, l, bl):
    # Convoles image with a Gaussian filter.
    
    # computing the 2D beam power spectrum
    grid, _ = tools.make_grid(mapparams, Fourier = True)
    bl2d = tools.interpolate_to_2d(mapparams, l, bl) 
    
    
    # convolving the image 
    image_fft = np.fft.fft2(image) 
    smoothed_image = np.fft.ifft2(image_fft * np.sqrt(bl2d)).real
    
    
    return smoothed_image


def wiener_filter(psd_signal, psd_noise, l, image, mapparams):
    # Filters out the noise from the corrupted signal.
   
    
    # computing the Wiener filter
    wiener_filter = psd_signal/(psd_signal+psd_noise)
    wiener_filter[np.isnan(wiener_filter)] = 0.
    
    grid, _ = make_grid(mapparams, Fourier = True)
    wiener_filter2d = interpolate_to_2d(grid, l, wiener_filter)
    
    
    # filtering the image
    image_fft = np.fft.fft2(image)
    filtered_image = (np.fft.ifft2(image_fft*wiener_filter2d)).real
    
    
    return filtered_image 


def low_pass_filter(l_cut, l, image, mapparams):
    low_pass_filter = np.ones(len(l))
    low_pass_filter[np.where(l.astype(int)>l_cut)] = 0.
    grid, _ = make_grid(mapparams, Fourier = True)
    low_pass_filter2d = interpolate_to_2d(grid, l, low_pass_filter)
    
    # filtering the image
    image_fft = np.fft.fft2(image)
    filtered_image = (np.fft.ifft2(image_fft*low_pass_filter2d)).real
    
    
    return filtered_image  

    
def gradient(image, dx):
    # Computes the relevent gradient information of an image.
    gradient_map = np.nan_to_num(np.gradient(image, dx))
    gradient_xmap, gradient_ymap = gradient_map[1], gradient_map[0]
    magnitude_map = np.hypot(gradient_xmap, gradient_ymap) 
    orientation_map = np.degrees(np.arctan2(gradient_ymap, gradient_xmap))
    return gradient_xmap, gradient_ymap, magnitude_map, orientation_map


def rotate(image, angle):
    # Rotates an image by a given angle. 
    rotated_map = scipy.ndimage.rotate(np.nan_to_num(image), angle, reshape = False, mode = 'reflect')  
    return rotated_map


def central_cutout(image, size, mapparams):
    # ...
    nx, dx, ny, dy = mapparams
    nber_pixels = size/dx
    s, e = int((ny-nber_pixels)/2), int((ny+nber_pixels)/2)
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
            std[bin_nbr] = np.std(image[ind])
        hit_count.append(hits)
        
        
    # computing error
    hit_count = np.array(hit_count)
    std_mean = np.sum(std*hit_count)/np.sum(hit_count)
    error = std_mean/(hit_count)**0.5
    
    
    return bin_ctr, rad_prf, error