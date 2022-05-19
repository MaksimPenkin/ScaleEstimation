"""
 @author   Maksim Penkin
"""


import numpy as np
import matplotlib.pyplot as plt


def touint8(img):
    COEF_8 = 2 ** 8 - 1

    return (np.clip(img, 0., 1.) * COEF_8).astype(np.uint8)


def touint16(img):
    COEF_16 = 2 ** 16 - 1

    return (np.clip(img, 0., 1.) * COEF_16).astype(np.uint16)


def minmax_norm(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))


def centered_norm(img):
    return (img - np.median(img)) / np.std(img)


def plot_line(x_values, y_values, save_path="tmp_plot_line.png",
              x_label='',
              y_label='',
              title='',
              grid=True,
              legend=False):
    fig = plt.figure()
    fig, ax = plt.subplots()

    ax.plot(x_values, y_values, "-o")
    if grid:
        ax.grid()
    ax.set(xlabel=x_label,
           ylabel=y_label,
           title=title)
    if legend:
        ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)


def plot_scatter(x_values, y_values, save_path="tmp_plot_scatter.png",
                 x_label='',
                 y_label='',
                 title='',
                 grid=True,
                 legend=True,
                 x_lim=None,
                 y_lim=None):
    fig = plt.figure()
    fig, ax = plt.subplots()

    ax.scatter(x_values, y_values)
    ax.plot(np.arange(x_lim[0], x_lim[1] + 1),
            [np.mean(y_values)]*(x_lim[1] - x_lim[0] + 1),
            color='orange', label='optimal scale')
    if grid:
        ax.grid()
    ax.set(xlabel=x_label,
           ylabel=y_label,
           title=title)
    if legend:
        ax.legend()
    if x_lim:
        assert isinstance(x_lim, tuple) and (len(x_lim) == 2)
        plt.xlim(*x_lim)
    if y_lim:
        assert isinstance(y_lim, tuple) and (len(y_lim) == 2)
        plt.ylim(*y_lim)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
