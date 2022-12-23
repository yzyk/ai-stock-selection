import os
import tensorflow as tf

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    PATH = '/content/drive/MyDrive/capstone/Deliverable/project'
else:
    PATH = "."

DATA_PATH = PATH + "/data/data.parquet.gzip"

MODEL_IMAGE_PATH = PATH + '/plot'

ERROR = 'mean_squared_error'

RF = 0.03

N_DAYS = 252

NUM_FEATURES = 152

NUM_STOCKS = 501

WIN_SIZE = 30

FORWARD_STEPS_SIZE = 15

MAX_EPOCHS = 10

BATCH_SIZE = 256

STOCK_LIST = None

COMPILE_CONFIG = {
    'optimizer': tf.keras.optimizers.Adam(learning_rate=1e-3),
    'loss': ERROR,
    'metrics': [ERROR],
    'weighted_metrics': [ERROR]
}
