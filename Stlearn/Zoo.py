from __future__ import annotations

from Data.DataProcessor import *
import tensorflow_probability as tfp
from utils import *
import Constant


class Prototype:

    @classmethod
    def simple_nn(cls):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(Constant.NUM_FEATURES + 1, name='dense_head')
        ])
        return model

    @classmethod
    def cnn(cls):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(8, padding='same', kernel_size=(3,),
                                   activation='relu', name='CNN_1'),
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
            tf.keras.layers.Dense(Constant.NUM_FEATURES + 1)
        ])
        return model

    @classmethod
    def lstm(cls):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.LSTM(1, return_sequences=True)
            # tf.keras.layers.Dense(1, name='dense_head')
        ])
        return model

    @classmethod
    def cnn_lstm(cls):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(25, padding='same', kernel_size=(1,),
                                   activation='relu', input_shape=(Constant.WIN_SIZE, Constant.NUM_FEATURES)),
            tf.keras.layers.Conv1D(50, padding='same', kernel_size=(1,),
                                   activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(1, return_sequences=True)
        ])


class AutoRegressor(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def warmup(self, inputs):
        return None

    @abstractmethod
    def call(self, inputs, training=None):
        return None


class LSTMAutoRegressor(AutoRegressor):
    def __init__(self, units, output_size=Constant.NUM_FEATURES + 1, out_steps=Constant.FORWARD_STEPS_SIZE):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(output_size)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # print('------------------------------------------------')
        predictions = []
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction[:, :-1]
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions[:, :, -1]


class CNNAutoRegressor(AutoRegressor):
    def __init__(self, units, out_steps=Constant.FORWARD_STEPS_SIZE):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.cnn = Prototype.cnn()

    def warmup(self, inputs):
        prediction = self.cnn(inputs)
        return prediction

    def call(self, inputs, training=None):
        predictions = []
        prediction = self.warmup(inputs)
        predictions.append(prediction)

        for n in range(1, self.out_steps):
            new_X = prediction[:, :-1]
            new_X = tf.reshape(new_X, [-1, 1, Constant.NUM_FEATURES])

            inputs = tf.boolean_mask(inputs, np.array(
                [False] + [True] * (Constant.WIN_SIZE - 1)), axis=1
                                     )
            inputs = tf.concat([inputs, new_X], 1)
            prediction = self.cnn(inputs)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions[:, :, -1]


class AutoEncoder(tf.keras.Model, ABC):

    def __init__(self, encoded_size):
        super().__init__()
        self._encoded_size = encoded_size
        pass

    @abstractmethod
    def _define_encoder(self):
        pass

    @abstractmethod
    def _define_decoder(self):
        pass

    @abstractmethod
    def call(self, inputs, training=None):
        return None


class VariationalAutoEncoder(AutoEncoder):

    def __init__(self, encoded_size, input_shape=None, output_shape=None):
        super().__init__(encoded_size)
        self._base_depth = encoded_size * 2
        self._input_shape = input_shape
        if output_shape is None:
            self._output_shape = self._input_shape
        else:
            self._output_shape = output_shape
        self._define_encoder()
        self._define_decoder()
        pass

    @abstractmethod
    def _define_encoder(self):
        pass

    @abstractmethod
    def _define_decoder(self):
        pass

    @abstractmethod
    def _define_prior(self):
        pass

    @abstractmethod
    def call(self, inputs, training=None):
        return None


class StandardVariationalAutoEncoder(VariationalAutoEncoder):

    def _define_encoder(self):
        self._define_prior()
        self._encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self._input_shape),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
            tf.keras.layers.Conv1D(self._base_depth, 5, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1D(2 * self._base_depth, 5, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1D(4 * self._base_depth, 7, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(self._encoded_size)),
            tfp.layers.IndependentNormal(self._encoded_size)
        ])
        pass

    def _define_decoder(self):
        self._decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=[self._encoded_size]),
            tf.keras.layers.Reshape([1, self._encoded_size]),
            tf.keras.layers.Conv1DTranspose(4 * self._base_depth, 7, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1DTranspose(2 * self._base_depth, 5, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1DTranspose(self._base_depth, 5, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(self._output_shape)),
            tfp.layers.IndependentNormal(self._output_shape)
        ])
        pass

    def _define_prior(self):
        self._prior = tfp.distributions.Independent(tfp.distributions.Normal(
            loc=tf.zeros(self._encoded_size), scale=1),
            reinterpreted_batch_ndims=1)

        pass

    def call(self, inputs, training=None):
        if training:
            X = self._encoder(inputs)
            # print(type(X))
            pred = self._decoder(X)
        else:
            Z = self._prior.sample(tf.shape(inputs)[0])
            # print(type(Z))
            # print(Z.shape)
            pred = self._decoder(Z)
        return pred

    def predict(self, X, steps):
        preds = []
        for step in steps:
            preds.append(tf.squeeze(self(X)))
        return preds[:, :, np.newaxis]


class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):

    class MyLSTMAutoRegressor(AutoRegressor):
        def __init__(self, output_size):
            super().__init__()
            self.init = tf.keras.initializers.GlorotNormal()
            self.output_steps = Constant.FORWARD_STEPS_SIZE
            self.input_reshape = tf.keras.layers.Reshape([Constant.WIN_SIZE, -1])
            self.fp1 = tf.keras.layers.Dense(Constant.NUM_STOCKS * output_size, kernel_initializer=self.init)
            self.lstm_cell = tf.keras.layers.LSTMCell(Constant.NUM_STOCKS * 1)
            self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
            self.fp2 = tf.keras.layers.Dense(Constant.NUM_STOCKS * output_size)
            self._output_reshape = tf.keras.layers.Reshape([self.output_steps, Constant.NUM_STOCKS, output_size])

        def warmup(self, inputs):
            inputs = self.input_reshape(inputs)
            inputs = self.fp1(inputs)
            x, *state = self.lstm_rnn(inputs)
            prediction = self.fp2(x)
            return prediction, state

        def call(self, inputs, training=None):
            predictions = []
            prediction, state = self.warmup(inputs)

            predictions.append(prediction)

            for n in range(1, self.output_steps):
                x = prediction
                x, state = self.lstm_cell(x, states=state, training=training)
                prediction = self.fp2(x)
                predictions.append(prediction)

            predictions = tf.stack(predictions)
            predictions = tf.transpose(predictions, [1, 0, 2])
            return self._output_reshape(predictions)

    def __init__(self, encoded_size=15, output_size=30):
        self.init = tf.keras.initializers.GlorotNormal()
        super().__init__(encoded_size, input_shape=(Constant.FORWARD_STEPS_SIZE, Constant.NUM_STOCKS),
                         output_shape=(output_size, 1))
        self._define_beta_layers(output_size)

    def _define_beta_layers(self, output_size):

        # None, win, stocks, features

        self._beta_layers = self.MyLSTMAutoRegressor(output_size)

        # None, steps, stocks, K

        pass

    def _define_encoder(self):
        self._define_prior()
        self._encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self._input_shape),
            tf.keras.layers.Conv1D(self._base_depth, 5, padding='same', activation=tf.nn.leaky_relu,
                                   kernel_initializer=self.init),
            tf.keras.layers.Conv1D(2 * self._base_depth, 5, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1D(4 * self._base_depth, 7, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(
                self._encoded_size * Constant.FORWARD_STEPS_SIZE)
                                  ),
            tfp.layers.IndependentNormal(self._encoded_size * Constant.FORWARD_STEPS_SIZE)
        ])
        pass

    def _define_decoder(self):
        self._decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=[self._encoded_size * Constant.FORWARD_STEPS_SIZE]),
            tf.keras.layers.Reshape([Constant.FORWARD_STEPS_SIZE, -1]),
            tf.keras.layers.Conv1DTranspose(4 * self._base_depth, 7, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1DTranspose(2 * self._base_depth, 5, padding='same', activation=tf.nn.leaky_relu
                                           ),
            tf.keras.layers.Conv1DTranspose(self._base_depth, 5, padding='same', activation=tf.nn.leaky_relu
                                            ),
            tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(self._output_shape)
                                  ),
            tfp.layers.IndependentNormal(self._output_shape)
        ])
        pass

    def _define_prior(self):
        self._prior = tfp.distributions.Independent(tfp.distributions.Normal(
            loc=tf.zeros(self._encoded_size * Constant.FORWARD_STEPS_SIZE), scale=1),
            reinterpreted_batch_ndims=1)

        pass

    def call(self, inputs, training=None):
        X = inputs[0]
        y = inputs[1]
        # tf.print(tf.shape(X))
        if training:
            beta = self._beta_layers(X)
            y = tf.keras.layers.Reshape([Constant.FORWARD_STEPS_SIZE, -1])(y)
            f = self._encoder(y)
            f = self._decoder(f)
            pred = beta @ f
        else:
            beta = self._beta_layers(X)
            Z = self._prior.sample(tf.shape(X)[0])
            f = self._decoder(Z)
            pred = beta @ f
        return pred

    def predict(self, X):
        beta = self._beta_layers(X)
        Z = self._prior.sample(tf.shape(X)[0])
        f = self._decoder(Z)
        pred = beta @ f
        return pred
