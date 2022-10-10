import diffractsim

diffractsim.set_backend("CPU")  # Change the string to "CUDA" to use GPU acceleration
from diffractsim import plot_colors
from diffractsim import MonochromaticField, ApertureFromImage, ApertureFromLayout, mm, nm, cm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def vertices_square(s):
    return (s['x'] + s['lx'] / 2, s['y'] + s['ly'] / 2), \
           (s['x'] - s['lx'] / 2, s['y'] + s['ly'] / 2), \
           (s['x'] - s['lx'] / 2, s['y'] - s['ly'] / 2), \
           (s['x'] + s['lx'] / 2, s['y'] - s['ly'] / 2)


def generate_map(extent, res, square_list):
    l_per_res = extent / res
    layout = np.zeros((res, res))
    center = res / 2 + 1
    for s in square_list:
        corners = np.ndarray.astype(np.divide(vertices_square(s), l_per_res) + center, 'int')
        layout[min(corners[:, 0]):max(corners[:, 0]), min(corners[:, 1]):max(corners[:, 1])] = 1
    return np.rot90(layout)


# scale = 1e-6
# one_grid_point = scale / 1000
extent = 10
res = 1024

square1 = {'x': 0,
           'y': 0,
           'lx': 1,
           'ly': 8}

square1a = {'x': 0.5,
            'y': 3.05,
            'lx': 0.5,
            'ly': 2}

square1b = {'x': -0.5,
            'y': 3.05,
            'lx': 0.5,
            'ly': 2}

square1c = {'x': 0.5,
            'y': -3.05,
            'lx': 0.5,
            'ly': 2}

square1d = {'x': -0.5,
            'y': -3.05,
            'lx': 0.5,
            'ly': 2}

square1e = {'x': 0,
            'y': 0,
            'lx': 1.35,
            'ly': 4}

square2 = {'x': 0.85,
           'y': 0,
           'lx': 0.1,
           'ly': 8}

square3 = {'x': -0.85,
           'y': 0,
           'lx': 0.1,
           'ly': 8}

square4 = {'x': 0.94,
           'y': 0,
           'lx': 0.25,
           'ly': 8}

square5 = {'x': -0.94,
           'y': 0,
           'lx': 0.25,
           'ly': 8}

layout1 = generate_map(extent, res,
                       [square1, square4, square5])
layout2 = generate_map(extent, res, [square1])

img = Image.fromarray(layout1 * 255)
# img.show()
# plt.imshow(layout1)
# plt.show(block=False)

lamda = 10 * nm
cd = 10 * lamda
d = 0.5 * lamda

F1 = MonochromaticField(
    wavelength=lamda, extent_x=cd, extent_y=cd, Nx=res, Ny=res
)
F1.add(ApertureFromLayout(layout1, image_size=(cd, cd), simulation=F1))
F1.propagate(d)
rgb = F1.get_colors()
field1 = np.real(F1.E * np.conjugate(F1.E))

F2 = MonochromaticField(
    wavelength=lamda, extent_x=cd, extent_y=cd, Nx=res, Ny=res
)
F2.add(ApertureFromLayout(layout2, image_size=(cd, cd), simulation=F2))
F2.propagate(d)
rgb = F2.get_colors()
field2 = np.real(F2.E * np.conjugate(F2.E))

F1.plot_colors(field1, xlim=[-cd / 2, cd / 2], ylim=[-cd / 2, cd / 2], units=nm)
F1.plot_colors(layout1, xlim=[-cd / 2, cd / 2], ylim=[-cd / 2, cd / 2], units=nm)
F2.plot_colors(field2, xlim=[-cd / 2, cd / 2], ylim=[-cd / 2, cd / 2], units=nm, do_block=True)

#
# plt.imshow(layout1)
# plt.show(block=True)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax1.set_title('Gray')
ax1.set_xlabel('')
ax1.set_ylabel('')

ax2.imshow(field2, cmap='jet')
ax2.set_title('Jet')
ax2.set_xlabel('')
ax2.set_ylabel('')
plt.show()
print("done!")
