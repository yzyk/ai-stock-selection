import numpy as np
from sklearn.metrics import *

DATA_PATH = "./data/data.parquet.gzip"

ERROR = 'mean_squared_error'


def performance_measure(y_true, y_pred):
    m_ = eval(ERROR)
    print('weighted_{}: {:.5f}'.format(
        ERROR, m_(y_true, y_pred, sample_weight=abs(y_true) / np.sum(abs(y_true)))
    )
    )
    return m_(y_true, y_pred, sample_weight=abs(y_true) / np.sum(abs(y_true)))
