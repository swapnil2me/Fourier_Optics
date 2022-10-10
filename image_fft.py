import os

import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import diffractsim

diffractsim.set_backend("CPU")  # Change the string to "CUDA" to use GPU acceleration
from diffractsim import plot_colors
from diffractsim import MonochromaticField, ApertureFromImage, ApertureFromLayout, mm, nm, cm
from iFFT import generate_map, get_fft_of_image, apply_filter, get_image_from_fftfshift_lpf


y = np.arange(0.25, 1.5, 0.0010)
filenames = []

lamda = 10 * nm
cd = 10 * lamda
d = 0.5 * lamda


for i, v in enumerate(y):
    square1 = {'x': 0, 'y': 0, 'lx': 1, 'ly': 8}
    square2 = {'x': v, 'y': 0, 'lx': 0.25, 'ly': 8}
    square3 = {'x': -v, 'y': 0, 'lx': 0.25, 'ly': 8}
    extent = 10
    res = 1024

    layout1 = generate_map(extent, res, [square1, square2, square3])
    img = Image.fromarray(layout1 * 255)
    rows, cols = img.size
    crow, ccol = int(rows / 2), int(cols / 2)

    F1 = MonochromaticField(
        wavelength=lamda, extent_x=cd, extent_y=cd, Nx=res, Ny=res
    )
    F1.add(ApertureFromLayout(layout1, image_size=(cd, cd), simulation=F1))
    F1.propagate(d)
    rgb = F1.get_colors()
    field1 = np.real(F1.E * np.conjugate(F1.E))

    mask = np.zeros((rows, cols), np.uint8)
    kx = 10
    ky = 10
    mask[crow - kx:crow + kx, ccol - ky:ccol + ky] = 1
    fshift, mag_spectrum = get_fft_of_image(img, plot=False)
    fshift, magnitude_spectrum = apply_filter(fshift, mask, plot=False)
    get_image_from_fftfshift_lpf(fshift)
    outi = get_image_from_fftfshift_lpf(fshift, plot=False)
    fshift, mag_spectrum = get_fft_of_image(outi, plot=False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Gray')
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    ax2.imshow(outi, cmap='jet')
    ax2.set_title('Jet')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    ax3.imshow(field1, cmap='jet')
    ax3.set_title('Jet')
    ax3.set_xlabel('')
    ax3.set_ylabel('')

    fig.suptitle(f'{v:.2f}', fontsize='xx-large')
    # plt.show(block=True)

    # plt.show()

    # create file name and append it to a list
    filename = f'frame/{i}.png'
    filenames.append(filename)

    # save frame
    plt.savefig(filename)
    plt.close()

# build gif
with imageio.get_writer('mygif_dist_withProp_10k_45fps_0.25width.gif', mode='I', fps=45) as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)

# Remove files
# for filename in set(filenames):
#     os.remove(filename)
