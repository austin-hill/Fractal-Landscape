'''Uses the diamond-square algorithm to generate and plot a 3d fractal landscape'''

import numpy as np
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def diamond_step(corner_1, corner_2, corner_3, corner_4, n_iter):
    return (corner_1+corner_2+corner_3+corner_4)/4 + np.random.randn()*2**(-n_iter)

def square_step(corner_1, corner_2, corner_3, corner_4, n_iter):
    return (corner_1+corner_2+corner_3+corner_4)/4 + np.random.randn()*2**(-n_iter)

def edge_square_step(corner_1, corner_2, corner_3, n_iter):
    return (corner_1+corner_2+corner_3)/4 + np.random.randn()*2**(-n_iter)

def diamond_square(fl, dim, k):

    for n_iterations in range(k):

        # diamond step
        step = 2**(k-n_iterations)
        half_step = int(step/2)
        for i in range(0, dim-1, step):
            for j in range(0, dim-1, step):
                fl[i+half_step, j+half_step] = diamond_step(fl[i, j], fl[i+step, j], fl[i, j+step],
                                                            fl[i+step, j+step], n_iterations)

        # square step
        for i in range(half_step, dim, step):
            for j in range(0, dim, step):
                if j == 0:
                    fl[i, j] = edge_square_step(fl[i+half_step, j], fl[i-half_step, j],
                                                fl[i, j+half_step], n_iterations)
                elif j == dim-1:
                    fl[i, j] = edge_square_step(fl[i+half_step, j], fl[i-half_step, j],
                                                fl[i, j-half_step], n_iterations)
                else:
                    fl[i, j] = square_step(fl[i+half_step, j], fl[i-half_step, j],
                                           fl[i, j+half_step], fl[i, j-half_step], n_iterations)
        for i in range(0, dim, step):
            for j in range(half_step, dim, step):
                if i == 0:
                    fl[i, j] = edge_square_step(fl[i+half_step, j], fl[i, j-half_step],
                                                fl[i, j+half_step], n_iterations)
                elif i == dim-1:
                    fl[i, j] = edge_square_step(fl[i-half_step, j], fl[i, j+half_step],
                                                fl[i, j-half_step], n_iterations)
                else:
                    fl[i, j] = square_step(fl[i+half_step, j], fl[i-half_step, j],
                                           fl[i, j+half_step], fl[i, j-half_step], n_iterations)

    return fl

# initialise variables
power = 8
size = 2**power+1
x = np.arange(size)
x, y = np.meshgrid(x, x)
# generate fractal - a median filter has been applied to remove occasional spikes which are
# a known artifact of this algorithm
fractal_landscape = medfilt2d(diamond_square(np.zeros((size, size)), size, power), kernel_size=3)

# create figure
fig = plt.figure(figsize=(15, 10))
ax = Axes3D(fig)

# adjust scaling of axes
scale_x = 1
scale_y = 1
scale_z = 0.5
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

# plot the result
ax.plot_wireframe(x, y, fractal_landscape)

plt.show()
