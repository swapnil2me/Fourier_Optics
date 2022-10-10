import numpy as np

import matplotlib.pyplot as plt


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_circle(extent, res, radius):
    l_per_res = extent / res
    layout = np.ones((res, res))
    rad_pix = int(np.divide(radius, l_per_res))
    mask = create_circular_mask(res, res, center=None, radius=rad_pix)
    layout[~mask] = 0
    return layout, mask


def generate_map(extent, res, square_list):
    l_per_res = extent / res
    layout = np.zeros((res, res))
    center = res / 2 + 1
    for s in square_list:
        corners = np.ndarray.astype(np.divide(vertices_square(s), l_per_res) + center, 'int')
        layout[min(corners[:, 0]):max(corners[:, 0]), min(corners[:, 1]):max(corners[:, 1])] = 1
    return np.rot90(layout)


def vertices_square(s):
    return (s['x'] + s['lx'] / 2, s['y'] + s['ly'] / 2), \
           (s['x'] - s['lx'] / 2, s['y'] + s['ly'] / 2), \
           (s['x'] - s['lx'] / 2, s['y'] - s['ly'] / 2), \
           (s['x'] + s['lx'] / 2, s['y'] - s['ly'] / 2)


def get_fft_of_image(img, plot=False, block=False):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img, cmap='jet')
        ax1.set_title('Input Image')
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        ax2.imshow(magnitude_spectrum / np.max(magnitude_spectrum), cmap='jet')
        ax2.set_title('Magnitude Spectrum')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        plt.show(block=block)

    return fshift, magnitude_spectrum


def apply_filter(spectrum, fltr, plot=False, block=False):
    fshift = np.multiply(spectrum, fltr)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(20 * np.log(np.abs(spectrum)), cmap='jet')
        ax1.set_title('Input Spectrum')
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        ax2.imshow(20 * np.log(np.abs(fshift)), cmap='jet')
        ax2.set_title('Masked Spectrum')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        plt.show(block=block)

    return fshift, magnitude_spectrum


def get_image_from_fftfshift_lpf(fshift_lpf, plot=False, block=False):
    f_ishift = np.fft.ifftshift(fshift_lpf)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_back, cmap='gray')
        ax1.set_title('Gray')
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        ax2.imshow(img_back, cmap='jet')
        ax2.set_title('Jet')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        plt.show(block=block)

    return img_back


def laplas_filter(rows, cols):
    mask_laplas = np.zeros((rows, cols), dtype=np.float32)
    for u in range(rows):
        for v in range(cols):
            mask_laplas[u, v] = 4 * np.pi * np.pi * ((u - rows / 2) ** 2 + (v - cols / 2) ** 2)
    mask_laplas = mask_laplas / np.max(mask_laplas)

    return mask_laplas


def gaussian_filter(rows, cols, sigma=[1, 1], mu=[0, 0]):
    sigma = [sigma[0] * rows, sigma[1] * cols]
    mu = [mu[0] + rows / 2, mu[1] + cols / 2]
    mask_gauss = np.zeros((rows, cols), dtype=np.float32)
    for u in range(rows):
        for v in range(cols):
            mask_gauss[u, v] = np.exp(
                -(((u - mu[0]) ** 2 / (2 * sigma[0] ** 2)) + ((v - mu[1]) ** 2 / (2 * sigma[1] ** 2))))
    return mask_gauss