import numpy as np
import os
import pendulum
from itertools import islice
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def plot_learning_stats(learning_history, title: str, grid=True,
                        figsize=(10, 8), show=True, *args, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_title(title)
    if grid:
        plt.grid()
    lines = {
        'median': ax.plot(np.median(learning_history, axis=0), label='mediana')[0],
        'max': ax.plot(np.max(learning_history, axis=0), label='max')[0],
        'min': ax.plot(np.min(learning_history, axis=0), label='min')[0]
    }
    ax.legend(lines.values(), lines.keys())
    ax.set_xlabel('episod')
    ax.set_ylabel('ilość kroków')
    if show:
        plt.show()
    return fig, ax


def compare_learning_curves(named_learning_histories: dict, title: str,
                            grid=True, figsize=(12, 10), show=True,
                            *args, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_title(title)
    if grid:
        plt.grid()
    lines = {
        name: ax.plot(np.median(history, axis=0), label=name)[0]
        for name, history in named_learning_histories.items()
    }
    ax.legend(lines.values(), lines.keys())
    ax.set_xlabel('episod')
    ax.set_ylabel('ilość kroków')
    if show:
        plt.show()
    return fig, ax
