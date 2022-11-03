import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import measure
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk)

import diffractsim
from diffractsim import MonochromaticField, ApertureFromLayout, nm
from iFFT import get_circle, simple_analysis, generate_rectangle_map, generate_annular_map

diffractsim.set_backend("CPU")  # Change the string to "CUDA" to use GPU acceleration

lamda = 193 * nm
cd = 10 * lamda
d = 1 * lamda
extent = 10
res = 1024

# for circles, the first quadrant is lower right of the layout
circle_list = [{'x': 1, 'y': 1.5, 'r': 0.5},
               {'x': -1.5, 'y': -1, 'r': 0.5},
               ]

# for rectangles, the y-axis is flipped
rect_list = [{'x': 1.5, 'y': 0, 'lx': 1, 'ly': 2},
             {'x': 0.5, 'y': -1.5, 'lx': 3, 'ly': 1},
             ]

sraf_list = [{'x': 0, 'y': -0.75, 'lx': 1.75, 'ly': 0.2},
             {'x': 0.75, 'y': -2.25, 'lx': 3.5, 'ly': 0.2},
             {'x': 0.75, 'y': 0.25, 'lx': 0.2, 'ly': 1.5},
             {'x': 2.25, 'y': -0.5, 'lx': 0.2, 'ly': 2.95},
             {'x': 0.20, 'y': -0.25, 'lx': 0.25, 'ly': 0.25},
             ]

slot_list = [{'x': 1.5, 'y': -0.75, 'lx': 0.25, 'ly': 0.55},
             {'x': 0.75, 'y': -1.5, 'lx': 0.55, 'ly': 0.25},
             {'x': 1.5, 'y': 1.0, 'lx': 0.25, 'ly': 0.55},
             {'x': -1.0, 'y': -1.5, 'lx': 0.55, 'ly': 0.25},
             # {'x': 1.50, 'y': -1.75, 'lx': 0.25, 'ly': 0.25},
             ]

layout = generate_rectangle_map(extent, res, rect_list, circle_list)

sraf = generate_rectangle_map(extent, res, sraf_list)
slot = generate_rectangle_map(extent, res, slot_list)

layout = np.logical_or(layout, sraf).astype('int')
layout = np.logical_xor(layout, slot).astype('int')

ann_data = [{'x': 1.15, 'y': -1.5, 'r': 0.85, 'w': 0.15, 'rot': 2},
            {'x': 1.15, 'y': 1.5, 'r': 0.85, 'w': 0.15, 'rot': 1}]
ann1 = generate_annular_map(extent, res, ann_data)

layout = np.logical_or(layout, ann1).astype('int')

img = Image.fromarray(layout * 255)

lpf, lpf_bool = get_circle(extent, res, 0.15, (0, 0))

Sim1 = MonochromaticField(wavelength=lamda, extent_x=cd, extent_y=cd, Nx=res, Ny=res)
Sim1.add(ApertureFromLayout(img, image_size=(cd, cd), simulation=Sim1))
Sim1.propagate(d * 1.15)

sim_img = simple_analysis(img, lpf, Sim1)
binary_img = sim_img > 0.6
footprint_d = disk(30)
footprint_e = disk(20)

dilated = dilation(binary_img, footprint_d)
eroded = erosion(binary_img, footprint_e)
contours_d = measure.find_contours(dilated, 0.8)
contours_e = measure.find_contours(eroded, 0.8)
fig, ax = plt.subplots(1, 3)

ax[0].imshow(sim_img, cmap='RdBu')
ax[0].set_title('Simulation')
ax[1].imshow(binary_img, cmap=plt.cm.gray)
ax[1].set_title('Threshold')
ax[2].imshow(binary_img, cmap=plt.cm.gray)
ax[2].set_title('Threshold')

for i, contour in enumerate(contours_d):
    # coords1 = measure.approximate_polygon(contour, tolerance=2.5)
    # coords2 = measure.approximate_polygon(contour, tolerance=39.5)
    ax[2].plot(contour[:, 1], contour[:, 0], linewidth=2, color='g')
    ax[2].plot(contours_e[i][:, 1], contours_e[i][:, 0], linewidth=2, color='r')

    # ax[0][2].plot(contours_d[i][:, 1], contours_d[i][:, 0], linewidth=2, color='r')
    # ax[1][1].plot(coords1[:, 1], coords1[:, 0], '-r', linewidth=2)
    # ax[1][1].plot(coords2[:, 1], coords2[:, 0], '-g', linewidth=2)

plt.show()
