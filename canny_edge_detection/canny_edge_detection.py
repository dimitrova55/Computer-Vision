from PIL import Image
import math
import numpy as np
import hw2
from scipy import ndimage


"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def gauss1d(sigma):
    pass

def gauss2d(sigma):
    pass

def convolve2d(array,filter):
    pass

def gaussconvolve2d(array,sigma):
    pass

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    # defining the Sobel's filters    
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
    # Ix = ndimage.convolve(img, Gx)
    # Iy = ndimage.convolve(img, Gy)
    
    # N.B. the image's array must be grayscale 2D array!!!
    
    Ix = hw2.convolve2d(img, Gx)
    Iy = hw2.convolve2d(img, Gy)
    
    G = np.hypot(Ix, Iy)
    
    # converting from float to uint8
    G = G / G.max() * 255
    
    # arctan of Iy / Ix
    theta = np.arctan2(Iy, Ix)               
                
    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    res = np.zeros_like(G, dtype=np.int32)
    
    # converting from radians to degrees
    theta = theta * 180. / np.pi
    # making the negative thetas positive
    theta[theta < 0] += 180

    
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            try:
                # the points are initialized to be black
                point1 = 255
                point2 = 255

                """
                determing two points around the current one according to the degree of theta
                between 0~45, 45~90, 90~135, 135~180 degrees we have 45 degree difference 45 / 2 = 22.5 degrees => 
                for 0 degree angle the range is: [0; 0+22.5) or [180-22.5; 180] = [0 <= theta < 22.5] or [157.5 <= theta <= 180]
                for 45 degree angle the range is: [45-22.5; 45+22.5] = [22.5 <= theta < 67.5]
                for 90 degree angle : [90-22.5; 90+22.5] = [67.5 <= theta < 112.5]
                """
            
               #theta = 0
                if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180):
                    point1 = G[i, j+1]
                    point2 = G[i, j-1]
                #theta 45
                elif (22.5 <= theta[i,j] < 67.5):
                    point1 = G[i+1, j-1]
                    point2 = G[i-1, j+1]
                #theta 90
                elif (67.5 <= theta[i,j] < 112.5):
                    point1 = G[i+1, j]
                    point2 = G[i-1, j]
                #theta 135
                elif (112.5 <= theta[i,j] < 157.5):
                    point1 = G[i-1, j-1]
                    point2 = G[i+1, j+1]
                    
                """
                comparing current point's pixel intensity with the two other points'; 
                if it is higher - preserve, otherwise make it white """

                if (G[i,j] >= point1) and (G[i,j] >= point2):
                    res[i,j] = G[i,j]
                else:
                    res[i,j] = 0

            except IndexError as e:
                pass
    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        NMS: non max edge suppression
    Returns:
        res: double_thresholded image.
    """
    res = np.zeros_like(img, dtype=np.int32)
    
    diff = img.max() - img.min()
    highThreshold = img.min() + diff * 0.15
    lowThreshold = img.min() + diff * 0.03
       
    weak = np.int32(80)
    strong = np.int32(255)
    nonrelevant = np.int32(0)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    res[zeros_i, zeros_j] = nonrelevant
    
    return res

def dfs(img, res, i, j, visited=[]):
    # 호출된 시점의 시작점 (i, j)은 최초 호출이 아닌 이상 
    # strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    res[i, j] = 255

    # visited points
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """
    Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing threshold response.
    Returns:
        res: hysteresised image.
    """
    strong = 255
    weak = 80
    res = np.zeros_like(img)
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
           if (img[i,j] == strong):
               dfs(img, res, i, j)
                
    return res


img = Image.open("iguana.bmp")
# img.show()

"""
img_gray3d = img.convert(mode='LA')
img_gray3d.show()

# grayscale 3D array
img_array = np.asarray(img_gray3d)

img_blurred = hw2.gaussconvolve2d(img_array, sigma=1.6)
image = Image.fromarray(img_blurred)
image.show()
"""

# redefine: colour 3D array
img_array = np.asarray(img) 
# Extracting each one of the RGB components 2D array
r_img, g_img, b_img = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

# The following operation will take weights and parameters to convert the colour image to grayscale
# It will return grayscale 2D array, not 3D
gamma = 1.400  # a parameter
r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma

img_blurred = hw2.gaussconvolve2d(grayscale_image, sigma=1.6)
image = Image.fromarray(img_blurred)
image.show()


img_sobel_G, img_sobel_theta = sobel_filters(img_blurred)
# image = Image.fromarray(img_sobel_G.astype(np.uint8))
image = Image.fromarray(img_sobel_G)
image.show()


non_max_sup_img = non_max_suppression(img_sobel_G, img_sobel_theta)
image = Image.fromarray(non_max_sup_img)
image.show()


threshold_img = double_thresholding(non_max_sup_img)
image = Image.fromarray(threshold_img)
image.show()


image = hysteresis(threshold_img)
image = Image.fromarray(image)
image.show()


