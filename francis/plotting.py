import numpy as np
import matplotlib.pyplot as plt


def figax(figsize=(16, 8)):
    return plt.subplots(1, figsize=figsize)


def oax(figsize=(16, 8)):
    return plt.subplots(1, figsize=figsize)[1]


def error_curve(ax, data, x=None, color='C0', e_alpha=.1, lw=None, ms=None, rescale=True):
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    if x is None:
        x = np.arange(len(m))
    y_lim = ax.get_ylim()
    ax.plot(x, m, color=color, lw=lw, ms=ms)
    if rescale:
        y_lim = ax.get_ylim()
    ax.fill_between(x, m - s, m + s, facecolors=color, alpha=e_alpha)
    ax.set_ylim(y_lim)
