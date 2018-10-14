import cv2 as cv
import numpy as np
import random
import sys

if __name__ == '__main__':
    img_path = sys.argv[1]

    # 2a: read and display the image
    img = cv.imread(img_path)
    cv.imshow('Original Image', img)

    # 2b: display the intenstity image
    intensity_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Intensity Image', intensity_img)

    # 2c: for loop to perform the operation
    height, width, channels = img.shape
    img_copy_c = img.copy()
    for x in range(height):
        for y in range(width):
            for c in range(channels):
                img_copy_c[x, y, c] = max(0, img[x, y, c] - 0.5 * intensity_img[x, y])
    cv.imshow('2c)', img_copy_c)
    # 2d: one-line statement to perfom the operation above
    img_copy_d = (img - 0.5 * intensity_img[:, :, np.newaxis]).clip(min=0).astype(
        np.uint8)  # clip sets all negative values to zero
    print(img_copy_d[0, 0])
    print(img_copy_c.dtype)
    cv.imshow('2d)', img_copy_d)
    # 2e: Extract a random patch
    patch_size = 16
    patch = img[(height - patch_size) // 2:(height + patch_size) // 2,
            (width - patch_size) // 2:(width + patch_size) // 2]
    cv.imshow('Patch', patch)
    # put it at random position
    # np.random.seed(0) #deterministic behaviour
    rand_x = np.random.randint(0, width - patch_size)
    rand_y = np.random.randint(0, height - patch_size)

    img[rand_y:rand_y + patch_size, rand_x:rand_x + patch_size] = patch
    cv.imshow('Patched Image', img)

    # 2f: Draw random rectangles and ellipses
    for i in range(20):
        if i < 10:
            # position and size is chosen uniformly at random over all possible rectangles,
            # which fit into the image
            x = np.random.randint(0, width - 1)
            y = np.random.randint(0, height - 1)
            box_width = np.random.randint(1, width - x)
            box_height = np.random.randint(1, height - y)
            color = np.random.randint(0, 255, size=3)
            img[y:y + box_height, x:x + box_width] = color
        else:
            # position and size is chosen uniformly at random over all possible ellipses (without rotation),
            # which fit into the image
            x = np.random.randint(1, width - 1)
            y = np.random.randint(1, height - 1)
            main_axis = np.random.randint(1, min(x, width - x) + 1)
            second_axis = np.random.randint(1, min(y, height - y) + 1)
            color = np.random.randint(0, 255, size=3)
            color = list(map(int, color))  # without this work around, the ellipse method won't get called properly
            img = cv.ellipse(img, (x, y), (main_axis, second_axis), 0, 0, 360, color, -1) #-1 indicates to fill the ellipse

    cv.imshow('Rectangles and Ellipses', img)

    # destroy all windows
    cv.waitKey(0)
    cv.destroyAllWindows()
