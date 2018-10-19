import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys

#returns a view on all submatrices of size n*n of A
def get_all_sub_matrices(A, n):
    sub_shape = (n,n)
    view_shape = tuple(np.subtract(A.shape, sub_shape) + 1) + sub_shape
    arr_view = np.lib.stride_tricks.as_strided(A,  view_shape, A.strides * 2)
    return arr_view.reshape((-1,) + sub_shape)

#A is a 2d matrix
def integral_image(A):
    return np.cumsum(np.cumsum(A,axis=0),axis=1)

def equalize_hist(img):
    hist = np.zeros(256)
    
    numbers,counts = np.unique(img, return_counts=True)

    hist[numbers] = counts

    cum_hist = np.cumsum(hist)
    cum_hist = cum_hist/cum_hist[-1]

    print(cum_hist)

    img = np.floor(cum_hist[img]*255)

    return img

def gaussian(x, sigma):
     return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power(x/sigma, 2.)/2)

#get Gaussian kernel of size (ksize,ksize). ksize is an int here
def getGaussianKernel1d(sigma):
    half_width = np.ceil(3*sigma)
    half_of_kernel = np.arange(0,np.ceil(3*sigma)+1)
    half_of_kernel = gaussian(half_of_kernel,sigma)
    kernel_1d = np.concatenate((half_of_kernel[::-1],half_of_kernel[1:]))
    return kernel_1d

def getGaussianKernel(sigma):
    kernel_1d = getGaussianKernel1d(sigma)
    return np.outer(kernel_1d,kernel_1d)



if __name__ == '__main__':
    img_path = sys.argv[1]
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel_size = 3

    box_filter = 1/(kernel_size**2)*np.ones((kernel_size,kernel_size))

    cv.getGaussianKernel

    #flip box fil
    # ter

    temp = get_all_sub_matrices(img,kernel_size)*box_filter[np.newaxis,:,:]

    cut = kernel_size//2

    #filtered_image = 2*img[1:-1,1:-1] -  np.sum(np.sum(temp,axis=1),axis=1).reshape(img.shape[0]-(kernel_size-1),img.shape[1]-(kernel_size-1))

    #filtered_image = np.einsum('ij,ijkl->kl',box_filter,temp)

    #filtered_image = filtered_image.astype(np.uint8)



    #print(img.shape)
    #print(filtered_image.shape)

    #cv.imshow('Input Image', img)
    #cv.imshow('Filtered Image', filtered_image)

    #cv.waitKey(0)
    #cv.destroyAllWindows()

#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    print('Task 1:');

    print("Just the sum: {}".format(np.average(img)))
    print("CV Integral: {}".format(cv.integral(img)[-1,-1]/img.size))
    print("Own Integral: {}".format(integral_image(img)[-1,-1]/img.size))

    height, width = img.shape
    posx = np.random.randint(0,height-100, size=10) 
    posy = np.random.randint(0,width-100, size=10)
    
    #A is integral image
    def get_average(A, x, y):
        big = A[x+100,y+100]
        sub1,sub2,small = 0,0,0
        if x-1 >= 0:
            sub1 = A[x-1,y+100]
        if y-1 >=0:
            sub2 = A[x+100,y-1]
        if x-1 >= 0 and y-1 >= 0:
            small = A[x-1,y-1]
        return big - sub1 - sub2 + small

    #naive
    start = time.time()
    for i in range(10):
        np.average(img[posx[i]:posx[i]+100,posy[i]:posy[i]+100])

    end = time.time()
    print(end - start)

    #own implementation
    start = time.time()
    integral = integral_image(img)
    mid = time.time()
    for i in range(10):
        get_average(integral,posx[i],posy[i])

    end = time.time()
    print(mid - start, end - mid, end - start)

    #cv implementation
    start = time.time()
    integral = cv.integral(img)
    for i in range(10):
        get_average(integral,posx[i],posy[i])

    end = time.time()
    print(end - start)
#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================
    print('Task 2:');
    cv.imshow('Original Image', img)
    own_equalization = equalize_hist(img).astype(np.uint8)
    cv.imshow('Own Equalization', equalize_hist(img).astype(np.uint8))
    cv.imshow('Original Image 2', img)
    cv_equalization = img.copy()
    cv.equalizeHist(img, cv_equalization)
    cv.imshow('CV Equalization', cv_equalization)



    print('Maximum error is:', np.unique((own_equalization-cv_equalization)))

    cv.waitKey(0)
    cv.destroyAllWindows()

#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:')
    sigma = 2*np.sqrt(2)
    kernel_size = int(2*np.ceil(3*sigma)+1)#by rule of thumb for ~99.7%
    print(kernel_size)
    cv.imshow('Opencv Gaussian Blur', cv.GaussianBlur(img,(kernel_size,kernel_size),sigma))
    cv.imshow('Own Filter with filter2d()', cv.filter2D(img,-1,getGaussianKernel(sigma)))
    kernel_1d = getGaussianKernel1d(sigma)
    cv.imshow('Own Filter with setpFilter2d()', cv.sepFilter2D(img,-1,kernel_1d,kernel_1d))
    cv.waitKey(0)
    cv.destroyAllWindows()

#    =========================================================================
#    ==================== Task 6 =================================
#    =========================================================================
    print('Task 6:');





#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================
    print('Task 7:');





#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================
    print('Task 8:');



