import numpy as np
from matplotlib import pyplot as plt

from diffractsim import MonochromaticField, ApertureFromLayout, nm
from iFFT import get_circle, simple_analysis, generate_rectangle_map
from PIL import Image
import diffractsim
from scipy import ndimage
from skimage import data, segmentation, measure
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk)
diffractsim.set_backend("CPU")  # Change the string to "CUDA" to use GPU acceleration

lamda = 193 * nm
cd = 10 * lamda
d = 1 * lamda
extent = 10
res = 1024

sq_list = [{'x': 0, 'y': 0, 'lx': 2, 'ly': 2},
           # {'x': 2, 'y': 0, 'lx': 1, 'ly': 8},
           # {'x': 4, 'y': 0, 'lx': 1, 'ly': 8},
           # {'x': 6, 'y': 0, 'lx': 1, 'ly': 8},
           # {'x': -2, 'y': 0, 'lx': 1, 'ly': 8},
           # {'x': -4, 'y': 0, 'lx': 1, 'ly': 8},
           # {'x': -6, 'y': 0, 'lx': 1, 'ly': 8},
           # {'x': 0, 'y': 0, 'lx': 8, 'ly': 1},
           # {'x': 0, 'y': 2, 'lx': 8, 'ly': 1},
           # {'x': 0, 'y': 4, 'lx': 8, 'ly': 1},
           # {'x': 0, 'y': 6, 'lx': 8, 'ly': 1},
           # {'x': 0, 'y': -2, 'lx': 8, 'ly': 1},
           # {'x': 0, 'y': -4, 'lx': 8, 'ly': 1},
           # {'x': 0, 'y': -6, 'lx': 8, 'ly': 1}
           ]

layout = generate_rectangle_map(extent, res, sq_list)

img = Image.fromarray(layout * 255)
layout1, cMask = get_circle(extent, res, 0.15, (0, 0))
F1 = MonochromaticField(wavelength=lamda, extent_x=cd, extent_y=cd, Nx=res, Ny=res)
F1.add(ApertureFromLayout(img, image_size=(cd, cd), simulation=F1))
F1.propagate(d)

sim_img = simple_analysis(img, cMask, F1)
print(np.max(sim_img))

binary_img = sim_img > 0.5
open_img = ndimage.binary_opening(binary_img)
# Remove small black hole
close_img = ndimage.binary_closing(open_img)
fig, ax = plt.subplots(2, 3)

footprint = disk(50)
dilated = dilation(close_img, footprint)

ax[0][0].imshow(binary_img, cmap=plt.cm.gray)
ax[0][0].set_title('binary_img')
ax[0][1].imshow(open_img, cmap=plt.cm.gray)
ax[0][1].set_title('open_img')
ax[0][2].imshow(close_img, cmap=plt.cm.gray)
ax[0][2].set_title('close_img')

ax[1][0].contour(close_img, [0.5], linewidths=2, colors='r')
ax[1][0].set_title('contour')
contours = measure.find_contours(close_img, 0.8)
contours_d = measure.find_contours(dilated, 0.8)
# print(contours)
for i, contour in enumerate(contours):
    coords1 = measure.approximate_polygon(contour, tolerance=2.5)
    coords2 = measure.approximate_polygon(contour, tolerance=39.5)
    ax[1][1].plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
    ax[1][1].plot(contours_d[i][:, 1], contours_d[i][:, 0], linewidth=2, color='r')
    # ax[1][1].plot(coords1[:, 1], coords1[:, 0], '-r', linewidth=2)
    # ax[1][1].plot(coords2[:, 1], coords2[:, 0], '-g', linewidth=2)

plt.show()