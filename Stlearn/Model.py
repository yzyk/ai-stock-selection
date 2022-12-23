from __future__ import annotations

import Constant
from Zoo import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from utils import *
import matplotlib.pyplot as plt
import json


class Model(ABC):
    _model = None
    _params = None
    _name = None

    @abstractmethod
    def __init__(self, name) -> None:
        self._create_model()
        self._name = name
        pass

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def fit(self, data) -> None:
        pass

    def predict(self, X):
        return self._model.predict(X)

    @abstractmethod
    def evaluate(self, data, s) -> None:
        pass

    @abstractmethod
    def info(self) -> None:
        pass

    @property
    def name(self):
        return self._name


class MlModel(Model):

    def __init__(self, name) -> None:
        super().__init__(name)
        pass

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, data) -> None:
        X_train, y_train = data.get_train()
        self._model.fit(X_train, y_train)
        pass

    def evaluate(self, data, s) -> None:
        X_test = None
        y_test = None
        if s == 'val':
            X_test, y_test = data.get_val()
        if s == 'test':
            X_test, y_test = data.get_test()
        y_pred = self.predict(X_test)
        performance_measure(y_test, y_pred)
        pass

    def info(self):
        print(self._model)
        pass


class DlModel(Model):
    _input_shape = None

    def __init__(self,
                 name, loss=Constant.ERROR, metric=Constant.ERROR,
                 weighted_metric=Constant.ERROR) -> None:
        self._input_shape = (Constant.WIN_SIZE, Constant.NUM_FEATURES)
        super().__init__(name)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=[metric],
            weighted_metrics=[weighted_metric])
        pass

    def get_config(self):
        res = {
            'name': self._name,
            'loss': Constant.ERROR,
            'weighted_metric': Constant.ERROR
        }
        return res

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, data) -> None:
        data_train = data.get_train()
        data_val = data.get_val()
        if type(data_train) is not tuple:
            X_train = data_train
            y_train = None
        else:
            X_train, y_train = data_train
        self._history = self._model.fit(X_train, y_train,
                                        epochs=Constant.MAX_EPOCHS,
                                        validation_data=data_val,
                                        callbacks=[GarbageCollectorCallback()]
                                        )
        fig, axs = plt.subplots(1, 2, figsize=(24, 10))
        self._plot_train(history=self._history, axs=axs)
        pass

    def save(self, dir_path):
        self._model.save_weights(dir_path + '/weights')
        with open(dir_path + '/config.json', 'w') as f:
            json.dump(self.get_config(), f)
        with open(dir_path + '/metric.json', 'w') as f:
            json.dump(self._history.history, f)
        pass

    @classmethod
    def load(cls, dir_path):
        with open(dir_path + '/config.json') as f:
            config = json.load(f)
        with open(dir_path + '/metric.json') as f:
            metric = json.load(f)
        model = cls(**config)
        model._model.load_weights(dir_path + '/weights')
        return model, metric

    def predict(self, X):
        pred = self._model.predict(X)
        if len(pred.shape) == 2:
            pred = pred[:, :, np.newaxis]
        return pred

    def evaluate(self, data, s) -> None:
        X_test = None
        y_test = None
        if s == 'val':
            X_test, y_test = data.get_val()
        if s == 'test':
            X_test, y_test = data.get_test()
        score_ = self._model.evaluate(X_test, y_test)
        print("{n:s}: Test loss: {l:3.5f}".format(
            n=self._name, l=score_[0])
        )
        pass

    def info(self):
        tf.keras.utils.plot_model(
            self._model, to_file=Constant.MODEL_IMAGE_PATH + '/' + self._name + '.png', show_shapes=True, dpi=100
        )
        print("Parameters number in model: ", self._model.count_params())
        return IPython.display.Image(Constant.MODEL_IMAGE_PATH + '/' + self._name + '.png')

    def _plot_train(self, history, axs):
        # Determine the name of the key that indexes into the accuracy metric
        acc_string = 'weighted_' + Constant.ERROR
        # Plot loss
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].set_title(self._name + " " + 'model loss')
        axs[0].set_ylabel('loss')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'validation'], loc='upper left')
        # Plot accuracy
        axs[1].plot(history.history[acc_string])
        axs[1].plot(history.history['val_' + acc_string])
        axs[1].set_title(self._name + ' ' + acc_string)
        axs[1].set_ylabel(acc_string)
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'validation'], loc='upper left')
        pass

    @classmethod
    def plot_train(cls, history, name='test'):
        # Determine the name of the key that indexes into the accuracy metric
        fig, axs = plt.subplots(1, 2, figsize=(24, 10))
        acc_string = 'weighted_' + Constant.ERROR
        # Plot loss
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].set_title(name + " " + 'model loss')
        axs[0].set_ylabel('loss')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'validation'], loc='upper left')
        # Plot accuracy
        axs[1].plot(history.history[acc_string])
        axs[1].plot(history.history['val_' + acc_string])
        axs[1].set_title(name + ' ' + acc_string)
        axs[1].set_ylabel(acc_string)
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'validation'], loc='upper left')
        pass


class LinearRegressionModel(MlModel):

    def _create_model(self):
        self._model = LinearRegression()
        pass


class RandomForestRegressorModel(MlModel):

    def _create_model(self):
        self._model = RandomForestRegressor()
        pass


class AdaBoostRegressorModel(MlModel):

    def _create_model(self):
        self._model = AdaBoostRegressor()
        pass


class SimpleNNModel(DlModel):

    def _create_model(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self._input_shape),
            tf.keras.layers.Dense(16, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, name='dense_head')
        ])

    pass


class CNNModel(DlModel):

    def _create_model(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(8, padding='same', kernel_size=(3,),
                                   activation='relu', input_shape=self._input_shape, name='CNN_1'),
            tf.keras.layers.Conv1D(8, padding='same', kernel_size=(3,),
                                   activation='relu', name='CNN_2'),
            tf.keras.layers.MaxPooling1D(pool_size=(2,)),
            tf.keras.layers.Conv1D(16, padding='same', kernel_size=(3,),
                                   activation='relu', name='CNN_3'),
            tf.keras.layers.Conv1D(16, padding='same', kernel_size=(3,),
                                   activation='relu', name='CNN_4'),
            tf.keras.layers.MaxPooling1D(pool_size=(2,)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(Constant.FORWARD_STEPS_SIZE,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([Constant.FORWARD_STEPS_SIZE, 1])
        ])
        pass


class LSTMModel(DlModel):

    def _create_model(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True, input_shape=self._input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(Constant.FORWARD_STEPS_SIZE),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([Constant.FORWARD_STEPS_SIZE, 1])
        ])
        pass


class LSTMAutoRegressorModel(DlModel):
    def _create_model(self):
        self._model = LSTMAutoRegressor(50)
        pass


class CNNAutoRegressorModel(DlModel):
    def _create_model(self):
        self._model = CNNAutoRegressor(50)

    pass


class StandardVariationAutoEncoderModel(DlModel):
    def __init__(self, name, loss=Constant.ERROR, metric=Constant.ERROR,
                 weighted_metric=Constant.ERROR, num_stocks=Constant.NUM_STOCKS,
                 output_steps=Constant.FORWARD_STEPS_SIZE
                 ):
        self.num_stocks = num_stocks
        self.output_steps = output_steps
        super().__init__(name, loss, metric, weighted_metric)
        pass

    def _create_model(self):
        self._model = StandardVariationalAutoEncoder(30, input_shape=(self.output_steps, self.num_stocks, 1))
        pass

    def predict(self, X=None):
        pred = self._model.predict(X)
        pred = tf.transpose(tf.squeeze(pred))
        pred = tf.expand_dims(pred, axis=-1)
        return pred.numpy()

    def get_config(self):
        config = super().get_config()
        config['num_stocks'] = self.num_stocks
        config['output_steps'] = self.output_steps
        return config


class ConditionalVariationAutoEncoderModel(DlModel):

    def __init__(self, name, loss=Constant.ERROR, metric=Constant.ERROR,
                 weighted_metric=Constant.ERROR, num_stocks=Constant.NUM_STOCKS,
                 win_size=Constant.WIN_SIZE,
                 output_steps=Constant.FORWARD_STEPS_SIZE
                 ):
        self.num_stocks = num_stocks
        self.win_size = win_size
        self.output_steps = output_steps
        super().__init__(name, loss, metric, weighted_metric)

    def _create_model(self):
        self._model = ConditionalVariationalAutoEncoder(encoded_size=15, output_size=30,
                                                        num_stocks=self.num_stocks,
                                                        win_size=self.win_size,
                                                        output_steps=self.output_steps)
        pass

    def predict(self, X=None):
        pred = self._model.predict(X)
        pred = tf.transpose(tf.squeeze(pred))
        pred = tf.expand_dims(pred, axis=-1)
        return pred.numpy()

    def evaluate(self, data, s) -> None:
        data_test = None
        if s == 'val':
            data_test = data.get_val()
        if s == 'test':
            data_test = data.get_test()
        score_ = self._model.evaluate(data_test)
        print("{n:s}: Test loss: {l:3.5f}".format(
            n=self._name, l=score_[0])
        )
        pass

    def get_config(self):
        config = super().get_config()
        config['num_stocks'] = self.num_stocks
        config['win_size'] = self.win_size
        config['output_steps'] = self.output_steps
        return config


class ConditionalVariationAutoEncoderGraphModel(DlModel):

    def __init__(self, name, loss=Constant.ERROR, metric=Constant.ERROR,
                 weighted_metric=Constant.ERROR, num_stocks=Constant.NUM_STOCKS,
                 win_size=Constant.WIN_SIZE,
                 output_steps=Constant.FORWARD_STEPS_SIZE
                 ):
        self.num_stocks = num_stocks
        self.win_size = win_size
        self.output_steps = output_steps
        super().__init__(name, loss, metric, weighted_metric)

    def _create_model(self):
        self._model = ConditionalVariationalAutoEncoderGraph(encoded_size=15, output_size=30,
                                                             num_stocks=self.num_stocks,
                                                             win_size=self.win_size,
                                                             output_steps=self.output_steps)
        pass

    def predict(self, X=None):
        pred = self._model.predict(X)
        pred = tf.transpose(tf.squeeze(pred))
        pred = tf.expand_dims(pred, axis=-1)
        return pred.numpy()

    def evaluate(self, data, s) -> None:
        data_test = None
        if s == 'val':
            data_test = data.get_val()
        if s == 'test':
            data_test = data.get_test()
        score_ = self._model.evaluate(data_test)
        print("{n:s}: Test loss: {l:3.5f}".format(
            n=self._name, l=score_[0])
        )
        pass

    def get_config(self):
        config = super().get_config()
        config['num_stocks'] = self.num_stocks
        config['win_size'] = self.win_size
        config['output_steps'] = self.output_steps
        return config
