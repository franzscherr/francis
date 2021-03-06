import numpy as np
from scipy.signal import savgol_filter
import multiprocessing as mp
import datetime as dt


cpu_count = mp.cpu_count()


def equi_sample(x, y, x_step=10):
    x_new, y_new = [], []
    x_curr = x[0]
    while x_curr < x[-1]:
        ind = np.logical_and(x >= x_curr, x < x_curr + x_step)
        if np.sum(ind) == 0:
            if x_curr == x[0]:
                raise Exception('first window empty')
            else:
                x_new.append(x_curr + x_step / 2)
                y_new.append(y_new[-1])
        else:
            y_new.append(y[ind].mean())
            x_new.append(x[ind].mean())
        x_curr += x_step
    return (np.array(x_new), np.array(y_new))


def inverse_dict(d):
    t = dict()
    for k, v in d.items():
        t[v] = k
    return t


def timestamps_to_datetimes(timestamps, n_processes=cpu_count, convert_to_numpy_array=True):
    if n_processes > 1:
        with mp.Pool(n_processes) as p:
            datetimes = p.map(dt.datetime.fromtimestamp, timestamps)
    else:
        datetimes = list(map(dt.datetime.fromtimestamp, timestamps))

    if convert_to_numpy_array:
        datetimes = np.array(datetimes)
    return datetimes


def datetimes_to_timestamps(datetimes, n_processes=cpu_count, convert_to_numpy_array=True):
    if n_processes > 1:
        with mp.Pool(n_processes) as p:
            timestamps = p.map(dt.datetime.timestamp, datetimes)
    else:
        timestamps = list(map(dt.datetime.timestamp, datetimes))

    if convert_to_numpy_array:
        timestamps = np.array(timestamps)
    return timestamps


def flatten(l):
    r = list()
    try:
        for a in l:
            r.extend(flatten(a))
    except TypeError:
        return [l]
    return r


def reshape(l, shape):
    shape = list(shape)
    l = flatten(l)
    p = 1
    negative_index = []
    for i, s in enumerate(shape):
        if s >= 0:
            p *= s
        else:
            negative_index.append(i)
    if len(negative_index) > 1:
        raise Exception('Cannot have more than one undefined dimension')
    elif len(negative_index) == 1:
        shape[negative_index[0]] = len(l) // p
        p *= shape[negative_index[0]]
    if len(l) != p:
        raise Exception('Total number of elements must agree')
    r = []
    for s in shape[1:][::-1]:
        step = p // s
        for i in range(0, p, s):
            r.append(l[i:i + s])
        p = p // s
        l = r
        r = []
    return l


def m_s(data, axis=None):
    m = np.mean(data, axis=axis)
    s = np.std(data, axis=axis)
    return m, s


def linear_smooth(data, n=50):
    k = np.ones(n) / n
    x_ex = np.concatenate(([data[0]] * n, data))
    return np.convolve(x_ex, k)[n:-n + 1]


def exponential_smooth(data, decay=.99):
    tt = np.zeros_like(data)
    tt[0] = data[0]
    for i, t in enumerate(data[1:]):
        tt[i + 1] = decay * tt[i] + (1 - decay) * t
    return tt


def savgol_smooth(data, n=7):
    return savgol_filter(data, n, 3, mode='nearest')


def tile_dim(data, n, axis=0):
    n_dims = len(data.shape)
    return np.tile(data, [1] * axis + [n] + [1] * (n_dims - axis - 1))


def structure_close(_s1, _s2, rtol=None, atol=None):
    if atol is None and rtol is not None:
        return np.all([np.allclose(a, b, rtol=rtol) for a, b in zip(_s1, _s2)])
    if atol is not None and rtol is None:
        return np.all([np.allclose(a, b, atol=atol) for a, b in zip(_s1, _s2)])
    if atol is not None and rtol is not None:
        return np.all([np.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(_s1, _s2)])
    return np.all([np.allclose(a, b) for a, b in zip(_s1, _s2)])

