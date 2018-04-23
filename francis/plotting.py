import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker


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


def parallel_coordinates(data, column_names, color_coded_column=0, cmap=cm.coolwarm, colors=None, **kwargs):
    args = dict(sharey=False, figsize=(16, 8))
    for k, v in kwargs.items():
        args[k] = v
    fig, axes = plt.subplots(1, data.shape[1] - 1, **args)
    used_colors = dict()
    x = np.arange(data.shape[1])
    d_min, d_max, d_range = [np.min(data, 0), np.max(data, 0), np.ptp(data, 0)]
    rescaled = np.true_divide((data - d_min[None, :]), d_range[None, :])

    for i, ax in enumerate(axes):
        for j in range(data.shape[0]):
            c = cmap(rescaled[j, color_coded_column])
            if colors is not None:
                c = colors[j]
            ax.plot(x, rescaled[j, :], c=c)
            used_colors[j] = c
        ax.set_xlim(x[i], x[i + 1])

    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = d_min[dim], d_max[dim], d_range[dim]
        step = val_range / float(ticks-1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_step = 1 / float(ticks-1)
        ticks = [round(norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(ticks)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([column_names[dim]])

    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([column_names[-2], column_names[-1]])
    ax.spines['right'].set_visible(True)

    plt.subplots_adjust(wspace=0)
    plt.legend(
        [plt.Line2D((0,1),(0,0), color=c) for c in used_colors.values()],
        column_names,
        bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
