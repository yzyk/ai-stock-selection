import copy

import numpy as np
from sklearn.metrics import *
import os
import tensorflow as tf
import Constant
from collections import Generator


def performance_measure(y_true, y_pred):
    m_ = eval(Constant.ERROR)
    print('weighted_{}: {:.5f}'.format(
        Constant.ERROR, m_(y_true, y_pred, sample_weight=abs(y_true) / np.sum(abs(y_true)))
    )
    )
    return m_(y_true, y_pred, sample_weight=abs(y_true) / np.sum(abs(y_true)))


def multi_label_measure(y_true, y_pred):
    # print(y_true)
    # print(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred))
    return tf.math.reduce_mean(
        tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    )
