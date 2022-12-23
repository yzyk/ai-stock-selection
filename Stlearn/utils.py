import copy
import linecache

import numpy as np
from sklearn.metrics import *
import os
import tensorflow as tf
import Constant
from collections import Generator
import tracemalloc
import gc

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


class TfDataset(object):
    def __init__(self):
        self.py_func_set_to_cleanup = set()

    def from_generator(self, generator, output_types, output_shapes=None, args=None):
        if not hasattr(tf.compat.v1.get_default_graph(), '_py_funcs_used_in_graph'):
            tf.compat.v1.get_default_graph()._py_funcs_used_in_graph = []
        py_func_set_before = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph)
        result = tf.data.Dataset.from_generator(generator, output_types, output_shapes, args)
        py_func_set_after = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph) - py_func_set_before
        self.py_func_set_to_cleanup |= py_func_set_after
        return result

    def cleanup(self):
        new_py_funcs = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph) - self.py_func_set_to_cleanup
        tf.compat.v1.get_default_graph()._py_funcs_used_in_graph = list(new_py_funcs)
        self.py_func_set_to_cleanup = set()

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
