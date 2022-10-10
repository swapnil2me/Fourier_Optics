import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage, ApertureFromLayout, mm, nm, cm
import numpy as np


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


scale = 1e-6
one_grid_point = scale / 1000
extent = 10
res = 512

square1 = {'x': 0,
           'y': 0,
           'lx': 5,
           'ly': 5}

square2 = {'x': 0,
           'y': 1.5,
           'lx': 1,
           'ly': 1}

square3 = {'x': 1,
           'y': 0.75,
           'lx': 1,
           'ly': 0.5}

layout = generate_map(extent, res, [square1, square2, square3])


F = MonochromaticField(
    wavelength=192 * nm, extent_x=18 * mm, extent_y=18 * mm, Nx=126, Ny=126
)

F.add(ApertureFromLayout(layout, image_size=(5.6 * mm, 5.6 * mm), simulation = F))


F.propagate(5*cm)
rgb = F.get_colors()
field =  np.real(F.E * np.conjugate(F.E))
# F.plot_colors(field, xlim=[-cd, cd], ylim=[-cd, cd])
F.plot_colors(field, xlim=[-7* mm, 7* mm], ylim=[-7* mm, 7* mm])
print("done!")