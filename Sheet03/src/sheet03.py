import numpy as np
import cv2 as cv
import random


##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    edges = cv.Canny(cv.cvtColor(img, cv.COLOR_RGB2GRAY), 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 50)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        # this point is in the image
        x0 = a * rho
        y0 = b * rho
        # "img.shape*"" to start and end the line outside of the image
        x1 = int(x0 + img.shape[1] * (-b))
        y1 = int(y0 + img.shape[0] * (a))
        x2 = int(x0 - img.shape[1] * (-b))
        y2 = int(y0 - img.shape[0] * (a))

        img = cv.line(img, (x1, y1), (x2, y2), [0, 255, 0])
    cv.imshow('lines', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


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
    thetas = (np.arange(180 / theta_step_sz) * theta_step_sz).astype(int)
    thetas_rad = np.deg2rad(thetas)
    theta_idx = range(int(180 / theta_step_sz))
    for x in range(img_edges.shape[1]):
        for y in range(img_edges.shape[0]):
            if img_edges[y, x] == 0:
                continue

            ds = ((x * np.cos(thetas_rad) - y * np.sin(thetas_rad)) / d_resolution).astype(int)

            accumulator[theta_idx, ds] += 1

    best_values = np.where(accumulator > threshold)
    best_values[0]/theta_step_sz
    best_values[1]/d_resolution
    detected_lines = np.array(best_values).T
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = cv.imshow('gray', cv.cvtColor(img, cv.COLOR_RGB2GRAY))
    edges = cv.Canny(cv.cvtColor(img, cv.COLOR_RGB2GRAY), 50, 150, apertureSize=3)
    detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    cv.imshow('Accumulator', accumulator.astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()


##############################################
#     Task 2        ##########################
##############################################

def gaussian(x, sig):
    return np.exp(np.dot(x,x)/ (2 * np.power(sig, 2.)))

def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(img_gray, 50, 150, apertureSize=3)
    theta_res = 2
    d_res = 1
    _,accumulator = myHoughLines(edges, d_res, theta_res, 50)
    #mean shift
    #the computation isn't correct, we don't know how to fix it
    h = 5
    modes = np.zeros((accumulator.shape[0],accumulator.shape[1],2))
    dists = np.array(range(-h//2+1,h//2+1))
    x_i = np.transpose([np.tile(dists, len(dists)), np.repeat(dists, len(dists))])
    dist_matrix = (np.linalg.norm(x_i, axis=1)**2/h).reshape(h,h)
    g_matrix = -np.exp(-dist_matrix/2)
    for x in range(2,accumulator.shape[1]-2):
        for y in range(2,accumulator.shape[0]-2):
            convergence = False
            xt,yt = x,y
            while not convergence:
                if np.any(accumulator[yt-h//2:yt+h//2+1,xt-h//2:xt+h//2+1]>0):
                    m_x = np.sum(x_i*(accumulator[yt-h//2:yt+h//2+1,xt-h//2:xt+h//2+1]*g_matrix).flatten()[:,np.newaxis], axis=0)
                    m_x /= np.sum(accumulator[yt-h//2:yt+h//2+1,xt-h//2:xt+h//2+1]*g_matrix)
                    m_x - np.array([yt,xt])
                    if xt == int(xt + m_x[0]) and yt == int(yt + m_x[1]):
                        convergence = True
                        modes[y,x] = [yt,xt]

                    xt = int(xt + m_x[1])
                    yt = int(yt + m_x[0])
                    if xt >=  accumulator.shape[1] -2 or yt >= accumulator.shape[0] -2 :
                        break
                else:
                    break


    
   



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
    W = np.array([[0, 1, 0.2, 1, 0, 0, 0, 0],
                  [1, 0, 0.1, 0, 1, 0, 0, 0],
                  [0.2, 0.1, 0, 1, 0, 1, 0.3, 0],
                  [1, 0, 1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 0, 0, 1, 0],
                  [0, 0, 0.3, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0, 1, 0]])
    D_vector = np.sum(W, axis=0)
    # a)
    D = np.diag(D_vector)
    D_sqrt = np.diag(np.sqrt(D_vector))
    D_inv_sqrt = np.diag(1 / np.sqrt(D_vector))
    _, eigen_values, eigen_vectors = cv.eigen(np.dot(np.dot(D_inv_sqrt, D - W), D_inv_sqrt))
    second_smallest = eigen_vectors[-2]
    y = np.dot(D_sqrt, second_smallest)  # this is the generalized eigen vector of (D-W)y=lambda D y
    # b)
    names = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    C_1 = names[y >= 0]
    C_2 = names[y < 0]
    print('C_1:', C_1)
    print('C_2:', C_2)

    # Instead of counting the edges etc. we are using the reformulated version
    normalized_cut_value = np.dot(np.dot(y, (D - W)), y) / (np.dot(np.dot(y, W), y))
    print('Normalized cut value:', normalized_cut_value)

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

