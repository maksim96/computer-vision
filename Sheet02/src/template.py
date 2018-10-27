import cv2
import numpy as np
import time


def get_convolution_using_fourier_transform(image, kernel):
    img_freq = np.fft.fft2(image)
    resized_kernel = np.zeros(image.shape)
    resized_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel
    kernel_freq = np.fft.fft2(kernel, s=img_freq.shape)
    # cv2.imshow('Image Frequencies', (20*np.log(np.abs(np.fft.fftshift(img_freq)))).astype(np.uint8))
    # cv2.imshow('Kernel Frequencies', np.abs(np.fft.fftshift(kernel_freq)))

    return np.fft.ifft2(img_freq * kernel_freq).astype(np.uint8)


def task1():
    print("Task1")
    image = cv2.imread('../data/einstein.jpeg', 0)
    kernel = cv2.getGaussianKernel(ksize=31, sigma=5)
    kernel = np.outer(kernel, kernel)

    conv_result = cv2.filter2D(image, -1, kernel)
    fft_result = get_convolution_using_fourier_transform(image, kernel)

    cv2.imshow('Blurring filter', conv_result)
    cv2.imshow('Frequency Domain', fft_result)#

    # compare results
    print('Mean difference:', np.average(np.abs(conv_result.astype(float) - fft_result.astype(float))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalized_cross_correlation(image, template):
    normalized_template = template.astype(float) - np.average(template.astype(float))
    variance_template = np.linalg.norm(normalized_template)
    norm_cross_corr = np.zeros(image.shape)
    image = image.astype(float)
    for i in range(image.shape[0] - (template.shape[0])):
        for j in range(image.shape[1] - (template.shape[1])):
            normalized_image_patch = image[i:i+template.shape[0],j:j+template.shape[1]] - np.average(image[i:i+template.shape[0],j:j+template.shape[1]])
            norm_cross_corr[i,j] = np.dot(normalized_template.flatten(),normalized_image_patch.flatten())
            image_patch_variance = np.linalg.norm(normalized_image_patch)
            norm_cross_corr[i,j] /= variance_template*image_patch_variance
    return norm_cross_corr.clip(min=0,max=255)


def task2():
    print("Task2")
    image = cv2.imread('../data/lena.png', 0)
    template = cv2.imread('../data/eye.png', 0)

    result_ncc = normalized_cross_correlation(image, template)
    #as in our version of ncc there is no value above 70% we noramlize it
    result_ncc = result_ncc/np.max(result_ncc)*255


    cv2.imshow('Normalized Cross Correlation', result_ncc.astype(np.uint8))
    #rectangles drawing. didn't get the task
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# draw rectangle around found location in all four results
# show the results

#returns a list holding each layer of the pyramid -opencv version
def build_gaussian_pyramid_opencv(image, num_levels):
    pyramid = [image]
    for i in range(1,num_levels):
        pyramid.append(cv2.pyrDown(pyramid[i-1]))

    return pyramid

#returns a list holding each layer of the pyramid -our version
def build_gaussian_pyramid(image, num_levels, kernel_size):
    pyramid = [image]
    for i in range(1,num_levels):
        blur = cv2.GaussianBlur(pyramid[i-1],(kernel_size,kernel_size),-1)
        pyramid.append(blur[::2,::2])
    return pyramid

def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
    return None

def task3():
    print("Task3")
    image = cv2.imread('../data/traffic.jpg', 0)
    template = cv2.imread('../data/traffic-template.png', 0)

    num_levels = 4

    cv_pyramid = build_gaussian_pyramid_opencv(image, num_levels)
    mine_pyramid = build_gaussian_pyramid(image, num_levels, kernel_size=5) #kernel_size of 5 results in the same pyramid as cv.pyrDown

    # compare and print mean absolute difference at each level
    for i in range(num_levels):
        print('Average difference at level', i, ' ', np.average(np.abs(cv_pyramid[i].astype(float) - mine_pyramid[i].astype(float))))

    pyramid_template = build_gaussian_pyramid(template, num_levels, 5)
    result = template_matching_multiple_scales(mine_pyramid, pyramid_template, 0.7)

    start = time.time()
    ncc = normalized_cross_correlation(image,template)
    end = time.time()

    print('Best match at position:', np.unravel_index(ncc.argmax(), ncc.shape))
    print('Time:', end - start)
    start = time.time()
    search_box = np.array([0, mine_pyramid[3].shape[0], 0, mine_pyramid[3].shape[1]])
    for i in range(num_levels-1,-1,-1):
        current_ncc = normalized_cross_correlation(mine_pyramid[i][search_box[0]:search_box[1], search_box[2]:search_box[3]], pyramid_template[i])
        current_best = np.unravel_index(current_ncc.argmax(), current_ncc.shape) + search_box[[0,0]]
        search_box = np.array((current_best[0] - pyramid_template[i].shape[0], current_best[0] + pyramid_template[i].shape[0], current_best[1] - pyramid_template[i].shape[1], current_best[1] + pyramid_template[i].shape[1]))
        search_box *= 2 #adjust for the next pyramid layer
    end = time.time()

    print('Best match at position:', current_best)
    print('Time:', end - start)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# show result

def get_derivative_of_gaussian_kernel(size, sigma):
    return None, None

def task4():
    image = cv2.imread('../data/einstein.jpeg', 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = None  # convolve with kernel_x
    edges_y = None  # convolve with kernel_y

    magnitude = None  # compute edge magnitude
    direction = None  # compute edge direction

    cv2.imshow('Magnitude', magnitude)
    cv2.imshow('Direction', direction)

def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    return None

def task5():
    image = cv2.imread('../data/traffic.jpg', 0)

    edges = None  # compute edges
    edge_function = None  # prepare edges for distance transform

    dist_transfom_mine = l2_distance_transform_2D(edge_function, positive_inf, negative_inf)
    dist_transfom_cv = None  # compute using opencv

# compare and print mean absolute difference

task1()
task2()
task3()
#task4()
task5()




