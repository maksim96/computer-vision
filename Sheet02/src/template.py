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
    image = cv2.imread('../data/einstein.jpeg', 0)
    kernel = cv2.getGaussianKernel(ksize=31, sigma=5)
    print(kernel * 255)
    kernel = np.outer(kernel, kernel)

    conv_result = cv2.filter2D(image, -1, kernel)
    fft_result = get_convolution_using_fourier_transform(image, kernel)

    cv2.imshow('Blurring filter', conv_result)
    cv2.imshow('Frequency Domain', fft_result)

    # compare results
    print('Mean difference:', np.average(np.abs(conv_result.astype(float) - fft_result.astype(float))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalized_cross_correlation(image, template):
    return None


def task2():
    image = cv2.imread('../data/lena.png', 0)
    template = cv2.imread('../data/eye.png', 0)

    result_ncc = normalized_cross_correlation(image, template)


# draw rectangle around found location in all four results
# show the results

def build_gaussian_pyramid_opencv(image, num_levels):
    pyramid = []
    for i in range(num_levels):
        pyramid.append(cv2.pyrDown(image))
		cv2.imshow(str(i),cv2.pyrDown(image))

    return pyramid

def build_gaussian_pyramid(image, num_levels, sigma):
    pyramid = []
    for i in range(num_levels):
        blur = cv2.GaussianBlur(image,int(sigma*3)+1,sigma)
        pyramid.append(blur[::2,::2])
    return pyramid

def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
    return None

def task3():
    image = cv2.imread('../data/traffic.jpg', 0)
    template = cv2.imread('../data/traffic-template.jpg', 0)

    num_levels = 4

    cv_pyramid = build_gaussian_pyramid_opencv(image, num_levels)
    mine_pyramid = build_gaussian_pyramid(image, num_levels, 2/3)

    # compare and print mean absolute difference at each level
    for i in range(num_levels):
		print(np.average(np.abs(cv_pyramid[i].astype(float) - mine_pyramid[i].astype(float))))

    pyramid_template = build_gaussian_pyramid(template, 4, 2/3)
    result = template_matching_multiple_scales(pyramid_mine, pyramid_template, 0.7)

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
task4()
task5()




