import imageio
from matplotlib.colors import LinearSegmentedColormap

from diffractsim import MonochromaticField, ApertureFromLayout, nm
from iFFT import get_circle, simple_analysis, generate_rectangle_map, generate_annular_map
from PIL import Image
import numpy as np
import diffractsim
import matplotlib.pyplot as plt

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
             ]

layout = generate_rectangle_map(extent, res, rect_list, circle_list)

sraf = generate_rectangle_map(extent, res, sraf_list)
slot = generate_rectangle_map(extent, res, slot_list)

layout = np.logical_or(layout, sraf).astype('int')
layout = np.logical_xor(layout, slot).astype('int')

ann_data = {'x': 0, 'y': 0, 'r': 2, 'w': 0.5, 'rot': 1}
layout = generate_annular_map(extent, res, [ann_data])

img = Image.fromarray(layout * 255)

lpf, lpf_bool = get_circle(extent, res, 0.15, (0, 0))

Sim1 = MonochromaticField(wavelength=lamda, extent_x=cd, extent_y=cd, Nx=res, Ny=res)
Sim1.add(ApertureFromLayout(img, image_size=(cd, cd), simulation=Sim1))
# Sim1.propagate(d)
# sim_img = simple_analysis(img, lpf, Sim1)
# plt.show()

colors = [(10 / 255, 137 / 255, 1 / 255), (1, 1, 1), (255 / 255, 154 / 255, 47 / 255)]
cmap = LinearSegmentedColormap.from_list('tricolor', colors, N=120)

filenames = []
for i in range(360):
    if i < 220:
        Sim1.propagate(d / 200)
    elif i >= 220 & i < 300:
        Sim1.propagate(d / 2000)
    else:
        Sim1.propagate(d / 20000)
    field1 = np.real(Sim1.E * np.conjugate(Sim1.E))
    fig = plt.figure(figsize=(16, 9), facecolor=[10 / 255, 137 / 255, 1 / 255])
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(field1, cmap=cmap)
    ax.axis('off')
    filename = f'diya_anim_tricolor/{i}.png'
    filenames.append(filename)

    # save frame
    plt.savefig(filename)
    plt.close()
# simple_analysis(img, lpf, Sim1)

# build gif
with imageio.get_writer('diya_anim_tricolor/diya.gif', mode='I', fps=60) as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)
