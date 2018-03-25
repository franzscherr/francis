import os
import sys
import logging
import pickle as pkl
import datetime as dt
import json
import numpy as np
import scipy as sc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

from IPython import embed

from .serialize import pickle, depickle
from .plotting import figax, oax, error_curve
from .data_tools import equi_sample, datetimes_to_timestamps, timestamps_to_datetimes, inverse_dict, flatten, reshape


exp_user = os.path.expanduser


logger = logging.getLogger('base')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \t- %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler('log.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)


ipython = embed


def now():
    return dt.datetime.now()


def get_console_logger(name):
    l = logging.getLogger(name)
    l.setLevel(logging.DEBUG)
    l.addHandler(ch)
    return l


def get_file_logger(name):
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    l.addHandler(fh)
    return l


__all__ = ['os', 'pkl', 'json', 'dt', 'now',
           'np', 'sc', 'tf', 'slim',
           'ipython',
           'pickle', 'depickle',
           'plt', 'figax', 'oax', 'error_curve', 'exp_user', 'sys',
           'get_console_logger', 'get_file_logger',
           'equi_sample', 'datetimes_to_timestamps', 'timestamps_to_datetimes', 'inverse_dict', 'flatten', 'reshape']
