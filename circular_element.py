from diffractsim import plot_colors
from diffractsim import MonochromaticField, ApertureFromImage, ApertureFromLayout, mm, nm, cm
from iFFT import generate_map, get_fft_of_image, apply_filter, get_image_from_fftfshift_lpf, get_circle
import os
import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import diffractsim


diffractsim.set_backend("CPU")  # Change the string to "CUDA" to use GPU acceleration


y = np.arange(0.25, 1.5, 0.0010)
filenames = []

lamda = 10 * nm
cd = 10 * lamda
d = 0.5 * lamda

extent = 10
res = 512
layout1, cMask = get_circle(extent, res, 0.5)
img = Image.fromarray(layout1 * 255)

F1 = MonochromaticField(
    wavelength=lamda, extent_x=cd, extent_y=cd, Nx=res, Ny=res
)
F1.add(ApertureFromLayout(layout1, image_size=(cd, cd), simulation=F1))
F1.propagate(d)
rgb = F1.get_colors()
field1 = np.real(F1.E * np.conjugate(F1.E))

rows, cols = img.size
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols), np.uint8)
kx = 1
ky = 1
mask[crow - kx:crow + kx, ccol - ky:ccol + ky] = 1
fshift, mag_spectrum = get_fft_of_image(img, plot=False)
fshift, magnitude_spectrum = apply_filter(fshift, mask, plot=False)
get_image_from_fftfshift_lpf(fshift)
outi = get_image_from_fftfshift_lpf(fshift, plot=False)
fshift, mag_spectrum = get_fft_of_image(outi, plot=False)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(img, cmap='gray')
ax1.set_title('Gray')
ax1.set_xlabel('')
ax1.set_ylabel('')

ax2.imshow(magnitude_spectrum, cmap='jet')
ax2.set_title('Jet')
ax2.set_xlabel('')
ax2.set_ylabel('')

ax3.imshow(outi, cmap='jet')
ax3.set_title('Jet')
ax3.set_xlabel('')
ax3.set_ylabel('')

ax4.imshow(field1, cmap='jet')
ax4.set_title('Jet')
ax4.set_xlabel('')
ax4.set_ylabel('')

fig.suptitle(f'{1:.2f}', fontsize='xx-large')
plt.show()