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
    hist = np.zeros(255)
    
    numbers,counts = np.unique(img, return_counts=True)

    hist[numbers] = counts

    cum_hist = np.cumsum(hist)
    cum_hist = cum_hist/cum_hist[-1]

    img = cum_hist[img]



if __name__ == '__main__':
    img_path = sys.argv[1]
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel_size = 3

    box_filter = 1/(kernel_size**2)*np.ones((kernel_size,kernel_size))

    #flip box filter

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





#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:');





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



