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

#A is integral image
def get_average(A, x, y,height=100,width=100):
        A = A.astype(int) #with uint8 overflow could occur
        big = A[x+height,y+width]
        sub1,sub2,small = 0,0,0
        if x-1 >= 0:
            sub1 = A[x-1,y+100]
        if y-1 >=0:
            sub2 = A[x+height,y-1]
        if x-1 >= 0 and y-1 >= 0:
            small = A[x-1,y-1]
        return big - sub1 - sub2 + small

def equalize_hist(img):
    hist = np.zeros(256)
    
    numbers,counts = np.unique(img, return_counts=True)

    hist[numbers] = counts

    cum_hist = np.cumsum(hist)
    cum_hist = cum_hist/cum_hist[-1]    

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
    #convert it once to greyscale for all tasks
    img_path = sys.argv[1]
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    #a)
    cv.imshow('Integral Image', integral_image(img))
    cv.waitKey(0)
    cv.destroyAllWindows()
    #b)
    print('Task 1:');
    print("Mean grey values by different methods:")
    print("Just the sum: {}".format(np.average(img)))
    print("CV Integral: {}".format(cv.integral(img)[-1,-1]/img.size))
    print("Own Integral: {}".format(integral_image(img)[-1,-1]/img.size))
    #c)
    height, width = img.shape
    #compute ten random starting points before hand
    posx = np.random.randint(0,height-100, size=10) 
    posy = np.random.randint(0,width-100, size=10)

    #naive
    start = time.time()
    for i in range(10):
        np.average(img[posx[i]:posx[i]+100,posy[i]:posy[i]+100])

    end = time.time()
    print("Just the sum took", end - start, "s")

    #own implementation
    start = time.time()
    integral = integral_image(img)
    for i in range(10):
        get_average(integral,posx[i],posy[i])

    end = time.time()
    print("CV Integral took", end - start, "s")

    #cv implementation
    start = time.time()
    integral = cv.integral(img)
    for i in range(10):
        get_average(integral,posx[i],posy[i])

    end = time.time()
    print("Own Integral tool", end - start, "s")
#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================
    print('Task 2:');
    cv.imshow('Original Image', img)
    own_equalization = equalize_hist(img).astype(np.uint8)
    cv.imshow('Own Equalization', equalize_hist(img).astype(np.uint8))
    cv_equalization = img.copy()
    cv.equalizeHist(img, cv_equalization)
    cv.imshow('CV Equalization', cv_equalization)



    print('Maximum pixel error:', np.max(np.abs(own_equalization.astype(float)-cv_equalization.astype(float))))
    cv.waitKey(0)
    cv.destroyAllWindows()

#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:')
    sigma = 2*np.sqrt(2)
    kernel_size = int(2*np.ceil(3*sigma)+1)#by rule of thumb for ~99.7%
    
    cv.imshow('Opencv Gaussian Blur', cv.GaussianBlur(img,(kernel_size,kernel_size),sigma))
    cv.imshow('Own Filter with filter2d()', cv.filter2D(img,-1,getGaussianKernel(sigma)))
    kernel_1d = getGaussianKernel1d(sigma)
    cv.imshow('Own Filter with setpFilter2d()', cv.sepFilter2D(img,-1,kernel_1d,kernel_1d))
    cv.waitKey(0)
    cv.destroyAllWindows()


#    =========================================================================
#    ==================== Task 5 =================================
#    =========================================================================
    print('Task 5:');
    cv.imshow("Original Image", img)
    
    sigma = 2
    kernel_size = int(2*np.ceil(3*sigma)+1) #rule of thumb from lecture
    img_2gauss = cv.GaussianBlur(cv.GaussianBlur(img, (kernel_size,kernel_size),sigma), (kernel_size,kernel_size),sigma)

    sigma = 2*np.sqrt(2)
    kernel_size = int(2*np.ceil(3*sigma)+1)
    img_gauss = cv.GaussianBlur(img, (kernel_size,kernel_size),sigma)

    cv.imshow('Gauss twice with sigma=2', img_2gauss)
    cv.imshow('Gauss with sigma = 2*sqrt(2)', img_gauss)

    print('Maximum pixel error:', np.max(np.abs(img_2gauss.astype(float) - img_gauss.astype(float))))

    cv.waitKey(0)
    cv.destroyAllWindows()


#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================
    print('Task 7:');
    cv.imshow("Original Image", img)
    noise = np.random.randint(0,20,size=img.shape)
    noise_img = img.copy()
    noise_img[noise <= 2] = 255 #15% white
    noise_img[(noise > 2) & (noise <= 5)] = 0 #15% black

    cv.imshow('Salt and Pepper', noise_img)

    error_gauss, error_median, error_bilateral = 256,256,256
    img = img.astype(float)
    for kernel in [1,3,5,7,9]:
        kernel = kernel *2 +1 #we assumed that the kernel sizes from the task actually meant the half-width, as otherwise the 1 wouldn't make much sense
        sigma = (kernel/2 -1) / 3 #inverse rule of thumb from the lecture
        
        tmp_gauss = cv.GaussianBlur(noise_img, (kernel,kernel),sigma).astype(float)
        tmp_median = cv.medianBlur(noise_img, kernel).astype(float)
        tmp_bilateral = cv.bilateralFilter(noise_img,kernel,10*kernel,10*kernel).astype(float) #use some heuristic from the internet for the sigmas
        
        current_error = np.average(np.abs(img - tmp_gauss))
        if current_error < error_gauss:
            img_gauss = tmp_gauss
            error_gauss = current_error

        current_error = np.average(np.abs(img - tmp_median))
        if current_error < error_median:
            img_median = tmp_median
            error_median = current_error
        
        current_error = np.average(np.abs(img - tmp_bilateral))
        if current_error < error_bilateral:
            img_bilateral = tmp_bilateral
            error_bilateral = current_error
        
    print("Gaussian best error: {}\nMedian best error: {}\nBilateral best error: {}".format(error_gauss, error_median, error_bilateral))

    cv.imshow("Optimized Gaussian", img_gauss.astype(np.uint8))
    cv.imshow("Optimized Median", img_median.astype(np.uint8))
    cv.imshow("Optimized Bilateral", img_bilateral.astype(np.uint8))

    cv.waitKey(0)
    cv.destroyAllWindows()
    img = img.astype(np.uint8)



#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================
    print('Task 8:');
    #a) regular filtering
    kernel_1 = np.array([[0.0113, 0.0838, 0.0113], 
                            [0.0838, 0.6183, 0.0838],
                            [0.0113, 0.0838, 0.0113]])

    kernel_2 = np.array([[-0.8984, 0.1472, 1.1410], 
                            [-1.9075, 0.1566, 2.1359],
                            [-0.8659, 0.0573, 1.0337]])


    filtered_img_1 = cv.filter2D(img,-1,kernel_1)
    filtered_img_2 = cv.filter2D(img,-1,kernel_2)

    cv.imshow('1. filter', filtered_img_1)
    cv.imshow('2. filter', filtered_img_2)

    #b) SVD approximation filtering
    w_1,u_1,v_1 = cv.SVDecomp(kernel_1)
    w_2,u_2,v_2 = cv.SVDecomp(kernel_2)

    if w_1[0] != 0:
        filtered_apx_1 = cv.sepFilter2D(img, -1, np.sqrt(w_1[0])*u_1[:,0], np.sqrt(w_1[0])*v_1[0])
        cv.imshow('1. Approximation', filtered_apx_1)

    if w_2[0] != 0:
        filtered_apx_2 = cv.sepFilter2D(img, -1, np.sqrt(w_2[0])*v_2[0],np.sqrt(w_2[0])*u_2[:,0])
        cv.imshow('2. Approximation', filtered_apx_2)

    #Remark: Actually both kernels are not separable, as sigma_1[1:] and sigma_2[1:] are non zero.
    #        But sigma_1[1:] is almost 0, so one could argue it is almost separable.
    #        kernel_2 is clearly not separable.

    #c) Pixelwise error
    print('Error 1. kernel', np.max(np.abs(filtered_img_1.astype(float) - filtered_apx_1.astype(float))))
    print('Error 2. kernel', np.max(np.abs(filtered_img_2.astype(float) - filtered_apx_2.astype(float))))

    cv.waitKey(0)
    cv.destroyAllWindows()