import numpy as np
from PIL import Image


def edges(element):
    pass


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
           'lx': 1,
           'ly': 1}

square2 = {'x': 0,
           'y': 1.5,
           'lx': 1,
           'ly': 1}

square3 = {'x': 1,
           'y': 0.75,
           'lx': 1,
           'ly': 0.5}

layout = generate_map(extent, res, [square1, square2, square3])
img = Image.fromarray(layout * 255)
img.show()
