import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from skimage import io 
from sklearn import preprocessing


def boxfilter(n):

    """ returns a box filter of size n by n, where n is odd number"""
    try:
        assert n%2==1
    except AssertionError:
        print("Dimension must be odd.")
        exit(1)
     
     # creating a numpy array with shape NxN with values 0.4      
    array = np.full(shape = (n,n), fill_value=0.4)
    return array
    
    
def gauss1d(sigma):
    
    """returns a 1D Gaussian filter for a given value of sigma
        the filter is a 1D Numpy array with length 6 times sigma rounded up to the next odd integer.  
        - x is an array of the distances from the center
        - if sigma is 1.6 then the filter length is odd(1.6*6)=11
        => 1D array of x values [-5,- 4,-3,-2,-1,0,1,2,3,4,5]
        and apply the array x to the given density function """
    
    # calculating the array’s length
    length = math.ceil(sigma * 6)
    if length % 2 == 0:
        length += 1
        
    # creating numpy array with values in the range [-(length//2), (length//2)]
    x = np.arange(start= -(math.floor(length/2)), stop=math.ceil(length/2), step=1)
    
    # calculating the gauss's function  
    gauss = np.exp(- x*x / (2 * (sigma*sigma)))
    
    # normalizing the vector
    vector_normalized = gauss / np.sum(gauss) 
        
    # print("gauss1d: ", vector_normalized)
    # print(sum(vector_normalized))
    return vector_normalized
    
    
def gauss2d(sigma):
        
    """returns a 2D Gaussian filter for a given value of sigma"""
   
    # creating the 2d filter using the 1d gauss's filter 
    gauss = gauss1d(sigma)
    gauss2d = np.outer(gauss, gauss)
    
    return gauss2d
    

def convolve2d(array, filter):
    
    """ takes in an image (stored in `array`) and a filter, 
    	and performs convolution to the image with zero paddings 
        => input image size = output image size
        Both input variables are in type `np.float32"""
    
    img_height = array.shape[0]
    img_width = array.shape[1]
    # img_color = array.shape[2]
    
    f_height = filter.shape[0]
    f_width = filter.shape[1]
    
    # rotation of the filter: flipped left-right and up-down
    filter = np.flipud(np.fliplr(filter))
    # filter = filter[:, :, np.newaxis]   
    
    # array of 0s with size of the input image, the convoluted (output) image will be stored here 
    convoluted_image = np.zeros_like(array)
    
    """adding padding of 0s to the input image
    1) create array of 0s which size = size(input image) + padding
        - left + right = 2 * padding; up + down = 2 * padding
    2) put the array (containing image's data) in the middle
    How much padding do we need?
    (in): n + 2p - f + 1 = n (out) => 2p = f - 1
    - the height of image + the height of filter - 1
    - the width of image + the width of filter - 1"""
    
    if(array.ndim == 2):
          
        padded_image = np.zeros(shape = (img_height + (f_height - 1), img_width + (f_width - 1)))

        padded_image[(f_height // 2) : -(f_height // 2),
                        (f_width // 2) : -(f_width // 2)] = array
        
        # if the array is 2D
        for x in range(img_height):
            for y in range(img_width):
                convoluted_image[x, y] = (padded_image[x:x+f_height, y:y+f_width] * filter).sum()
                
    elif(array.ndim == 3):
        
        padded_image = np.zeros(shape = (img_height + (f_height - 1), img_width + (f_width - 1), array.shape[2]))

        padded_image[(f_height // 2) : -(f_height // 2),
                        (f_width // 2) : -(f_width // 2)] = array
    
        """the input image is 3d array, so 
        - 1st loop is iterating colours
        - 2nd loop is iterating height
        - 3rd loop is iterating width"""
            
        # if the array is 3D
        for c in range(array.shape[2]):
            for x in range(img_height):
                for y in range(img_width):
                    convoluted_image[x, y, c] = (padded_image[x:x+f_height, y:y+f_width, c] * filter).sum()   
    
    # returning the image's array in integer format          
    # if img_color == 1:
    #     return np.resize(convoluted_image, shape=(convoluted_image.shape[0],  convoluted_image.shape[1])).astype(np.uint8)
    # else:
    #     return convoluted_image.astype(np.uint8)
    return convoluted_image


def gaussconvolve2d(array, sigma):
    
    """applies Gaussian convolution to a 2D array for
        the given value of sigma. The result should be a 2D array.
        array: image like array"""
        
    kernel = gauss2d(sigma)
        
    return convolve2d(array, kernel)
