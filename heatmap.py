import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt


def diffuse(u0, nsteps=1):
    # plate size, mm
    w = u0.shape[0]
    h = u0.shape[1]
    # intervals in x-, y- directions, mm
    dx = dy = 1
    # Thermal diffusivity of steel, mm2.s-1
    D = 4.

    nx, ny = int(w / dx), int(h / dy)

    dx2, dy2 = dx * dx, dy * dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

    u = np.empty((nx, ny))

    def do_timestep(u0, u):
        # Propagate with forward-difference in time, central-difference in space
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
                (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / dx2
                + (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dy2)

        u0 = u.copy()
        return u0, u

    for m in range(nsteps):
        u0, u = do_timestep(u0, u)

    return u


###########


def generate_heatmap_dummy():
    x = np.arange(-5.01, 5.01, 0.25)
    y = np.arange(-5.01, 5.01, 0.25)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx ** 2 + yy ** 2)
    f = interpolate.interp2d(x, y, z, kind='cubic')
    xnew = np.arange(-5.01, 5.01, 1e-2)
    ynew = np.arange(-5.01, 5.01, 1e-2)
    znew = f(xnew, ynew)

    plt.imshow(znew, cmap='hot', interpolation='nearest')
    plt.show()


# def generate_heatmap(x, y, values):
#     # xs = np.linspace(min(x), max(x), num=100)
#     # ys = np.linspace(min(y), max(y), num=100)
#     #
#     # xx, yy = np.meshgrid(xs, ys)
#     # zz = np.zeros(xx.shape)
#     # zz.fill(np.mean(values))
#     #
#
#     f = interpolate.interp2d(x, y, values, kind='cubic')
#     xnew = np.linspace(min(x), max(x), num=100)
#     ynew = np.linspace(min(y), max(y), num=100)
#     znew = f(xnew, ynew)
#
#     plt.imshow(znew, cmap='hot', interpolation='nearest')
#     plt.show()

# def generate_heatmap(x, y, values):
#
#     points = np.column_stack((x, y))
#     grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
#     values = np.array(values)
#     grid = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=np.mean(values))
#
#     grid = diffuse(grid)
#     return grid

def generate_heatmap(x, y, values, img_size=(100, 100)):
    img = np.zeros(img_size)
    x = np.array(x)
    y = np.array(y)
    x = np.floor((x - np.min(x)) / (np.max(x) - np.min(x)) * img.shape[0] - 1).astype(int)
    y = np.floor((y - np.min(y)) / (np.max(y) - np.min(y)) * img.shape[1] - 1).astype(int)

    values = (values - min(values)) / (np.max(values) - np.min(values))

    img[x, y] = values

    img = diffuse(img, nsteps=10)
    # plt.imshow(img)
    return img


if __name__ == "__main__":
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    grid = generate_heatmap(x, y, z)
    plt.imshow(grid.T, extent=None, origin='lower')
    plt.show()
