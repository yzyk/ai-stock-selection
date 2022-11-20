import os

DATA_PATH = "./data/data.parquet.gzip"

MODEL_IMAGE_PATH = os.path.realpath('./plot')

ERROR = 'mean_squared_error'

RF = 0.03

N_DAYS = 252

NUM_FEATURES = 152

WIN_SIZE = 60

MAX_EPOCHS = 10
