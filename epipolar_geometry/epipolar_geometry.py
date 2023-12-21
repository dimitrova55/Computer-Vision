import numpy as np
import matplotlib.pyplot as plt

"""
img1 = plt.imread('./data/warrior_a.jpg')
img2 = plt.imread('./data/warrior_b.jpg')

cor1 = np.load("./data/warrior_a.npy")
cor2 = np.load("./data/warrior_b.npy")
"""

img1 = plt.imread('./data/graffiti_a.jpg')
img2 = plt.imread('./data/graffiti_b.jpg')

cor1 = np.load("./data/graffiti_a.npy")
cor2 = np.load("./data/graffiti_b.npy")



def compute_fundamental(x1,x2):
    """ 
    x1, x2 : 3xN vectors
    - use 8-point algorithm to compute the fundamental matrix
    - get a least square solution for the fundamental matrix F,
    and ensure that rank(F) = 2 using SVD where the last singular value is set to be zero."""
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
     
    # build matrix for equations in Page 51   
    # compute the solution in Page 51        
    # constrain F: make rank 2 by zeroing out last singular value (Page 52)
    
    # Create the constraint matrix A
    A = np.zeros((x1.shape[1], 9))
    for i in range(x1.shape[1]):
        A[i] = [x1[0][i]*x2[0][i], x1[1][i]*x2[0][i], x2[0][i],
                x1[0][i]*x2[1][i], x1[1][i]*x2[1][i], x2[1][i],
                x1[0][i], x1[1][i],1]
        
    # Solve the homogeneous linear system using SVD
    U, S, Vh = np.linalg.svd(A)
    F = Vh[-1].reshape(3, 3)

    # Enforce the rank-2 constraint on F
    U, S, Vh = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vh
    # F = np.dot(U, np.dot(np.diag(S), Vh))

    F = F/F[2, 2]
    
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1
    
    return F


def compute_epipoles(F):
    """
    compute  the  epipoles  of  two  images  based  on  the fundamental matrix
    Epipoles must be normalized using homogenous coordinate.
    F * e1 = 0
    F * e2 = 0
    """
    # e1 = None
    # e2 = None
    
    # Compute the null space of F to get e1
    _, _, V = np.linalg.svd(F)
    e1 = V[-1].T

    # Compute the null space of F transpose to get e2
    _, _, V = np.linalg.svd(F.T)
    e2 = V[-1].T

    # Normalize the epipoles using homogeneous coordinates
    e1 = e1 / e1[2]
    e2 = e2 / e2[2]
    
    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    
    """
    draw the epipolar lines on the images img1, img2
    cor1, cor2 : shape 3 x N
    """
    F = compute_norm_fundamental(cor1, cor2)

    # e1, e2 = compute_epipoles(F)
    
    # Compute epipolar lines in the first image
    lines1 = np.dot(F.T , cor2)
    lines1 = lines1 / np.linalg.norm(lines1[:2], axis=0)


    # Compute epipolar lines in the second image
    lines2 = np.dot(F, cor1)
    lines2 = lines2 / np.linalg.norm(lines2[:2], axis=0)
    
    # Calculate image dimensions
    img1_height, img1_width = img1.shape[:2]
    img2_height, img2_width = img2.shape[:2]

    # Draw the epipolar lines on the first image
    _, ax = plt.subplots()
    ax.imshow(img1)
    for line in lines1.T:
        a, b, c = line
        x = np.array([0, img1.shape[1]])
        y = -(a * x + c) / b

        # Clip the line within the image boundaries
        mask = (y >= 0) & (y < img1_height)
        x = x[mask]
        y = y[mask]

        ax.plot(x, y, color='blue')

    # Draw the epipolar lines on the second image
    _, ax = plt.subplots()
    ax.imshow(img2)
    for line in lines2.T:
        a, b, c = line
        x = np.array([0, img2_width])
        y = -(a * x + c) / b

        # Clip the epiline within the image boundaries
        mask = (y >= 0) & (y < img2_height)
        x = x[mask]
        y = y[mask]

        ax.plot(x, y, color='blue')

    plt.show()
    
    return

draw_epipolar_lines(img1, img2, cor1, cor2)