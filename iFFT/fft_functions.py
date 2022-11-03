import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import rotate

import matplotlib.pyplot as plt

colors = [(65/255, 171/255, 93/255), (1, 1, 1), (252/255, 78/255, 42/255)]

cmap = LinearSegmentedColormap.from_list('cmap_name', colors, N=120)

def simple_analysis(input_image, input_mask, simulator=None):
    if simulator:
        F1 = simulator
        field1 = np.real(F1.E * np.conjugate(F1.E))
        field1 = rotate(field1, -180)
        extent = [-F1.extent_x / 2, F1.extent_x / 2, -F1.extent_y / 2, F1.extent_y / 2]
    else:
        extent = [-1, 1, -1, 1]

    fshift, mag_spectrum = get_fft_of_image(input_image, plot=False)
    fshift_masked, mag_spectrum_masked = apply_filter(fshift, input_mask, plot=False)
    outi_filtered = get_image_from_fftfshift_lpf(fshift_masked, plot=False)
    fshift_out, mag_spectrum_out = get_fft_of_image(outi_filtered, plot=False)

    fig, ax = plt.subplots(2, 3)
    ax1 = ax[0][0]
    ax2 = ax[1][0]

    ax3 = ax[0][1]
    ax4 = ax[1][1]

    ax5 = ax[0][2]
    ax6 = ax[1][2]

    color_map = 'seismic'

    ax1.imshow(input_image, cmap=color_map, origin='lower', extent=extent)
    ax1.set_title('Input Image')
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    ax2.imshow(mag_spectrum / np.max(mag_spectrum), cmap=color_map, origin='lower', extent=extent)
    ax2.set_title('Input Spectrum')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    ax3.imshow(input_mask / np.max(input_mask), cmap='gray', origin='lower', extent=extent)
    ax3.set_title('Mask')
    ax3.set_xlabel('')
    ax3.set_ylabel('')

    if simulator:
        ax4.imshow(field1 / np.max(field1), cmap=cmap, origin='lower', extent=extent)
        ax4.set_title('Simulation')
        ax4.set_xlabel('')
        ax4.set_ylabel('')
    else:
        ax4.imshow(mag_spectrum_masked / np.max(mag_spectrum_masked), cmap='RdBu', origin='lower', extent=extent)
        ax4.set_title('Masked Spectrum')
        ax4.set_xlabel('')
        ax4.set_ylabel('')

    ax5.imshow(outi_filtered / np.max(outi_filtered), cmap='RdBu', origin='lower', extent=extent)
    ax5.set_title('Output')
    ax5.set_xlabel('')
    ax5.set_ylabel('')

    ax6.imshow(mag_spectrum_out / np.max(mag_spectrum_out), cmap=color_map, origin='lower', extent=extent)
    ax6.set_title('Output Spectrum')
    ax6.set_xlabel('')
    ax6.set_ylabel('')
    plt.show(block=False)

    out1 = field1 / np.max(field1)
    return rotate(out1, 180)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_circle(extent, res, radius, center=None):
    l_per_res = extent / res
    layout = np.ones((res, res))
    rad_pix = np.divide(radius, l_per_res).astype('int')
    if center:
        center_pix = np.divide(center, l_per_res).astype('int') + int(res / 2)
    else:
        center_pix = None
    mask = create_circular_mask(res, res, center=center_pix, radius=rad_pix)
    layout[~mask] = 0
    return layout, mask


def generate_annular_map(extent, res, ann_data_list):
    layout = np.zeros((res, res))
    for ann_data in ann_data_list:
        r = ann_data['r']
        w = ann_data['w']
        r_0 = r
        r_i = r - w
        c1, mask = get_circle(extent, res, r_0, (ann_data['x'], ann_data['y']))
        c2, mask = get_circle(extent, res, r_i, (ann_data['x'], ann_data['y']))
        s = {'x': ann_data['x'] + ann_data['r'] / 2,
             'y': -ann_data['y'],
             'lx': ann_data['r'],
             'ly': ann_data['r'] * 2}
        c1_xor_c2 = np.logical_xor(c1, c2).astype('int')
        rect_mask = generate_rectangle_map(extent, res, [s])
        c1_xor_c2 = np.logical_and(c1_xor_c2, rect_mask).astype('int')
        c1_xor_c2 = np.rot90(c1_xor_c2, ann_data['rot'])
        layout = np.logical_or(layout, c1_xor_c2).astype('int')
    return layout


def generate_rectangle_map(extent, res, square_list, circle_list=None):
    l_per_res = extent / res
    layout = np.zeros((res, res))
    center = res / 2 + 1
    for s in square_list:
        corners = np.ndarray.astype(np.divide(vertices_square(s), l_per_res) + center, 'int')
        layout[min(corners[:, 0]):max(corners[:, 0]), min(corners[:, 1]):max(corners[:, 1])] = 1

    if circle_list:
        for circle in circle_list:
            circle_map, mask = get_circle(extent, res, circle['r'], (circle['x'], circle['y']))
            layout = np.logical_or(layout, circle_map).astype('int')

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
