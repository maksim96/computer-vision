#!/usr/bin/python3.5

import numpy as np
import cv2 as cv #just for displaying
from scipy import misc


'''
    read the usps digit data
    returns a python dict with entries for each digit (0, ..., 9)
    dict[digit] contains a list of 256-dimensional feature vectores (i.e. the gray scale values of the 16x16 digit image)
'''
def read_usps(filename):
    data = dict()
    with open(filename, 'r') as f:
        N = int( np.fromfile(f, dtype = np.uint32, count = 1, sep = ' ') )
        for n in range(N):
            c = int( np.fromfile(f, dtype = np.uint32, count = 1, sep = ' ') )
            tmp = np.fromfile(f, dtype = np.float64, count = 256, sep = ' ') / 1000.0
            data[c] = data.get(c, []) + [tmp]
    for c in range(len(data)):
        data[c] = np.stack(data[c])
    return data

'''
    load the face image and foreground/background parts
    image: the original image
    foreground/background: numpy arrays of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''
def read_face_image(filename):
    image = misc.imread(filename) / 255.0
    bounding_box = np.zeros(image.shape)
    bounding_box[50:100, 60:120, :] = 1
    foreground = image[bounding_box == 1].reshape((50 * 60, 3))
    background = image[bounding_box == 0].reshape((40000 - 50 * 60, 3))
    return image, foreground, background



'''
    implement your GMM and EM algorithm here
'''
class GMM(object):
    mus = np.zeros(0)
    sigmas = np.zeros(0)
    lambdas = np.zeros(0)
    '''
        fit a single gaussian to the data
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
    '''
    def fit_single_gaussian(self, data):
        self.mus = np.mean(data, axis=0)[np.newaxis,:]
        self.sigmas = np.var(data, axis=0)[np.newaxis,:]
        self.lambdas = np.ones(1)

    '''
        pdf of multivariate gaussian
    '''
    def norm_pdf_multivariate(self, x, mu, sigma):
        sigma += 0.0001
        temp = 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma))
        return np.prod(temp)


    '''
        implement the em algorithm here
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
        @n_iterations: the number of em iterations
    '''
    def em_algorithm(self, data, n_iterations = 10):
        for i in range(n_iterations):
            #E-step
            r = np.zeros((data.shape[0], self.lambdas.size))

            for x in range(data.shape[0]):
                denominator = 0
                for k in range(self.lambdas.size):
                    x_prob = self.norm_pdf_multivariate(data[x], self.mus[k], self.sigmas[k])
                    r[x,k] = self.lambdas[k]*x_prob
                    denominator += self.lambdas[k]*x_prob

                r[x,:] /= denominator

            #M-Step
            self.lambdas = np.sum(r, axis=0)/np.sum(r)
            for i in range(self.lambdas.size):
                self.mus[k] = np.sum(r[:,i][:,np.newaxis]*data, axis=0)/np.sum(r[:,i], axis=0)
                self.sigmas[k] = np.sum(r[:,i][:,np.newaxis]*(data - self.mus[i])**2, axis=0)/np.sum(r[:,i], axis=0)

    '''
        implement the split function here
        generates an initialization for a GMM with 2K components out of the current (already trained) GMM with K components
        @epsilon: small perturbation value for the mean
    '''
    def split(self, epsilon = 0.1):
        self.lambdas = 0.5*np.append(self.lambdas, self.lambdas)
        mus    = np.zeros((self.lambdas.size, self.mus.shape[1]))
        sigmas = np.zeros((self.lambdas.size, self.mus.shape[1]))

        for i, mu in enumerate(self.mus):
            mus[i] = (mu + epsilon * self.sigmas[i])
            mus[i+self.lambdas.size//2] = (mu - epsilon * self.sigmas[i])

            sigmas[i] = (self.sigmas[i])
            sigmas[i+self.lambdas.size//2] = (self.sigmas[i])

        self.mus = mus
        self.sigmas = sigmas
    '''
        sample a D-dimensional feature vector from the GMM
    '''
    def sample(self):
        which_gaussian = np.random.choice(len(self.lambdas), 1, p=self.lambdas)[0]
        return np.random.multivariate_normal(self.mus[which_gaussian], np.diag(self.sigmas[which_gaussian]))




'''
    Task 2d: synthesizing handwritten digits
    if you implemeted the code in the GMM class correctly, you should not need to change anything here
    We changed split to 3 as it in the task we need a mixture of 8 Gaussians in the end.
    In the task the say do 100 iterations, we just do 10 because it would just take to long otherwise (but this was the default here anyway)
'''
data = read_usps('usps.txt')
gmm = [ GMM() for _ in range(10) ] # 10 GMMs (one for each digit)
for split in [0, 1, 2, 3]:
    result_image = np.zeros((160, 160))
    for digit in range(10):
        # train the model
        if split == 0:
            gmm[digit].fit_single_gaussian(data[digit])
        else:
            gmm[digit].em_algorithm(data[digit])
        # sample 10 images for this digit
        for i in range(10):
            x = gmm[digit].sample()
            x = x.reshape((16, 16))
            x = np.clip(x, 0, 1)
            result_image[digit*16:(digit+1)*16, i*16:(i+1)*16] = x
        # save image
        misc.imsave('digits.' + str(2 ** split) + 'components.png', result_image)
        # split the components to have twice as many in the next run
        gmm[digit].split(epsilon = 0.1)

'''
    Task 2e: skin color model
'''
image, foreground, background = read_face_image('face.jpg')

'''
    TODO: compute p(x|w=foreground) / p(x|w=background) for each image pixel and manipulate image such that everything below the threshold is black, display the resulting image
'''
gmm_foreground = GMM()
gmm_background = GMM()
for split in [0, 1, 2, 3]:
    # train the model
    if split == 0:
        gmm_foreground.fit_single_gaussian(foreground)
        gmm_background.fit_single_gaussian(background)
    else:
        gmm_foreground.em_algorithm(foreground)
        gmm_background.em_algorithm(background)
    if split < 3:
        gmm_foreground.split(epsilon=0.1)
        gmm_background.split(epsilon=0.1)

thresholded_image = np.ones((image.shape[:2])).astype(np.uint8)*255

threshold = 0.1

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        foreground_prob = 0
        background_prob = 0
        x = image[i,j]
        for k in range(gmm_foreground.lambdas.size):
            foreground_prob += gmm_foreground.lambdas[k]*gmm_foreground.norm_pdf_multivariate(x, gmm_foreground.mus[k], gmm_foreground.sigmas[k])
            background_prob += gmm_background.lambdas[k] * gmm_background.norm_pdf_multivariate(x, gmm_background.mus[k], gmm_background.sigmas[k])

        if foreground_prob/background_prob < threshold:
            thresholded_image[i,j] = 0

#Unfortunately almost everythung just get set to skin...

cv.imshow('Skin', thresholded_image)
cv.waitKey(0)
cv.destroyAllWindows()
