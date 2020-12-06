import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def diamond_step(c1, c2, c3, c4, n):
    return (c1+c2+c3+c4)/4 + np.random.randn()/(n+1)**2

def square_step(v1, v2, v3, v4, n):
    return (v1+v2+v3+v4)/4 + np.random.randn()/(n+1)**2

def edge_square_step(v1, v2, v3, n):
    return (v1+v2+v3)/4 + np.random.randn()/(n+1)**2

def diamond_square(fl, dim, k):
    for n_iterations in range(k):
        
        # diamond step
        step = 2**(k-n_iterations)
        half_step = int(step/2)
        for i in range(0, dim-1, step):
            for j in range(0, dim-1, step):
                fl[i+half_step, j+half_step] = diamond_step(fl[i, j], fl[i+step, j], fl[i, j+step], fl[i+step, j+step], n_iterations)
        
        # square step
        for i in range(half_step, dim, step):
            for j in range(0, dim, step):
                if j == 0:
                    fl[i, j] = edge_square_step(fl[i+half_step, j], fl[i-half_step, j], fl[i, j+half_step], n_iterations)
                elif j == dim-1:
                    fl[i, j] = edge_square_step(fl[i+half_step, j], fl[i-half_step, j], fl[i, j-half_step], n_iterations)
                else:
                    fl[i, j] = square_step(fl[i+half_step, j], fl[i-half_step, j], fl[i, j+half_step], fl[i, j-half_step], n_iterations)
        for i in range(0, dim, step):
            for j in range(half_step, dim, step):
                if i == 0:
                    fl[i, j] = edge_square_step(fl[i+half_step, j], fl[i, j-half_step], fl[i, j+half_step], n_iterations)
                elif i == dim-1:
                    fl[i, j] = edge_square_step(fl[i-half_step, j], fl[i, j+half_step], fl[i, j-half_step], n_iterations)
                else:
                    fl[i, j] = square_step(fl[i+half_step, j], fl[i-half_step, j], fl[i, j+half_step], fl[i, j-half_step], n_iterations)
    return fl

power = 10
size = 2**power+1
fractal_landscape = diamond_square(np.zeros((size, size)), size, power)
x = np.arange(size)
x, y = np.meshgrid(x, x)
        
fig = plt.figure()
ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, fractal_landscape)

plt.show()