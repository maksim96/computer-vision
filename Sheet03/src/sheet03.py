import numpy as np
import cv2 as cv
import random


##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    '''
    ...
    your code ...
    ...
    '''


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
    '''
    ...
    your code ...
    ...
    '''
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    #detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    theta_res = None # set the resolution of theta
    d_res = None # set the distance resolution
    #_, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    
    data.astype(float)

    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]
    

    # initialize centers using some random points from data
    centers = data.copy()
    np.random.shuffle(centers)
    centers = centers[:k]
    
    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        closest = np.argmin(
            np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2)), axis = 0
        )

        
        # update clusters' centers and check for convergence
        new_centers = np.array([data[closest==i].mean(axis=0) for i in range(k)])
        if (new_centers == centers).all():
            convergence = True
        else:
            centers = new_centers

        
        iterationNo += 1
        print('iterationNo = ', iterationNo)

    index = closest
    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    arr_img = img.copy()
    arr_img = arr_img.reshape((arr_img.size,1))
    
    for i in [2,4,6]:
        index, _ = myKmeans(arr_img, i)
        for k in range(i):
            img[index.reshape(img.shape) == k] = k/i * 255
        cv.imshow("k={} clustering".format(i), img)
        cv.waitKey(0)
    cv.destroyAllWindows()

def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    arr_img = img.copy()
    arr_img = arr_img.reshape((int(arr_img.size/3),3))
    
    for i in [2,4,6]:
        index, _ = myKmeans(arr_img, i)
        for k in range(i):
            img[index.reshape(img.shape[:-1 ]) == k] = k/(i-1) * 255
        cv.imshow("k={} clustering".format(i), img)
        cv.waitKey(0)
    cv.destroyAllWindows()


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    arr_img = img.copy()
    arr_img = arr_img.reshape((int(arr_img.size/3),3))
    
    ind = np.indices(img.shape[:-1])
    ind = np.dstack((ind[0], ind[1]))
    print(img.shape, arr_img.shape, ind.shape)
    ind = ind.reshape((int(arr_img.size/3), 2))

    arr_img = np.concatenate((arr_img, ind), axis=1)

    # todo: add relative image location scaled to 255

    # normalize the data to a scale from 0 to 255
    arr_img = arr_img.astype(float)
    arr_img[:,-1] = (arr_img[:,-1]/arr_img[:,-1].argmax(0)) * 255
    arr_img[:,-2] = (arr_img[:,-2]/arr_img[:,-2].argmax(0)) * 255

    for i in [2,4,6]:
        index, _ = myKmeans(arr_img, i)
        for k in range(i):
            img[index.reshape(img.shape[:-1 ]) == k] = k/(i-1) * 255
        cv.imshow("k={} clustering".format(i), img)
        cv.waitKey(0)
    cv.destroyAllWindows()


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = None  # construct the D matrix
    W = None  # construct the W matrix
    '''
    ...
    your code ...
    ...
    '''


##############################################
##############################################
##############################################


task_1_a()
task_1_b()
task_2()
task_3_a()
task_3_b()
task_3_c()
task_4_a()

