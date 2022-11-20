# Stlearn: Experimental Machine Learning Stock Selection Framework

In this module, I develop Stlearn library to enclasp machine learning codes. I employ Object Oriented Design Patterns to design the framework. I use design patterns for this framework because:
* We should avoid add codes to existing classes to make them support more general information
* Handle possible future changes in requirement in advance 
* Enclasp as many code as possible and only write codes when necessary

To achieve this, I select [Abstract Factory Pattern](https://www.oodesign.com/abstract-factory-pattern) as my design pattern for Stlearn framework. Below, we define system as a machine learning client system that is either faced with commercial user, or faced with researcher that studies the performance of different machine learning models. We define product as required data and machine learning model. This pattern applies to our framework because:
* The system needs to be independent from the way the products are created, client only cares about what model to call and it is solely the developer's responsibility to implement and specify how products are created using specified protocol or API.
* The system should be configured to work with multiple families of products, where different machine learning task consist of pipelines with different dataset and models.
* A family of products is designed to work together, where one type machine learning task requires a specific type of data and models.

Based on general ideas of Abstract Factory Pattern, we design the framework and modules of Stlearn as follow:

<img src="./img/stframework.png" alt="framework" title="framework" width="4000" height="500"/>

In this framework, the client system is expected to use `Data` and `Model` objects and all their derived classes objects as their products to perform machine learning task. The rule of abstract factory discourages the client to directly call constructor of these objects to access specific product objects, because clients are expected not to know anything about how to construct or implement a specific product, and because how to construct and implement a specific product may change from time to time. Instead, the client is expected to go to `StlearnFactory` and all its derived classes to access specific product through unified and constant APIs.

For developer and producer who provides new model or new data to use, all they need to do is to:
* Derive a subclass from `MlModel` or `DlModel`, override the `_create_model()` function, where a specific machine learning or deep learning model which at lease has `fit()` and `predict()` function should be defined and assigned to `self._model` variable
* Derive a subclass from `MlData` or `DlData`, override the `_generate_data()` function, which should read data from a certain data source and then split it into train, validation and test dataset. For our problem, since data has already been provided, derive a new subclass is not necessary at all.
* Derive a subclass from `MlFactory` or `DlFactory`, override `_load()` function to initialize the relevant `Data` and `Model` object and return them accordingly.

# Model

We build a set of modules to support the functionality of machine learning and deep learning models. Fot the sake of convenience for extension in the future and enable the current framework to accommodate more advanced deep learning models, we apply the [Single Responsibility Principle](https://en.wikipedia.org/wiki/Single-responsibility_principle) to design the architecture.

## Design

By Single Responsibility, we argue each class should do only one job. We hereby separate the scope of machine learning/deep learning models from the academic side into the following modules:
* `Zoo`: it is only used to develop and implement machine learning and deep learning model prototypes. It means we only define what the model should look like here, as long as inner structure
* `Model`: it is only used to encapsulate data and behavior of an already-defined sophisticated model defined in `Zoo` or other libraries such as Sklearn or TensorFlow

The reason behind this design is intuitive. In the real world, we have a lot of models that have already been written by libraries or frameworks, such as Random Forest, Linear Regression, CNN. But we also have tons of models that we have to derive some classes and write their mechanisms on our own. Thus we need a separate module `Zoo` to define these prototypes. We then encapsulate these self-defined sophisticated models together with library models into `Model` module.

<img src="./img/Model.png" alt="framework" title="framework" width="4000" height="500"/>

We derive two subclasses from `Model`, which are `MlModel` and `DlModel`, representing abstract classes for machine learning and deep learning models. We then derive subclasses from these two classes, overriding `_create_model()` to define the `_model` variable. 

All the models' output have been transformed into multi-time-step predictions. By multi-time-step predictions, we are saying that for each sample:
* We take in all its available features,
* We predict multiple labels for multiple future time steps, rather than only predict one label

We currently have two solutions to this:
* `Single Shot Model`: We still do one prediction, but we can duplicate, or we can output a sequence then reshape it
* `Auto Regressive Model`: We do one prediction each time, we then feed this prediction as input features into the model to get the prediction for the next time step

## Library-defined Model

We currently encapsulated some well-known library-defined models:
* Machine Learning:
  * Linear Regression
  * Random Forest
  * AdaBoost
* Deep Learning:
  * Simple Neural Networks
  * Convolution Neural Networks (CNN)
  * Long Short Term Memory Neural Networks (LSTM)

All of these models except LSTM that returns sequences are attributed to `Single Shot Model` stated ahead. Please refer to the following image from TensorFlow documentation to have a illustration on its mechanism.

<img src="./img/multistep_conv.png" alt="framework" title="framework" width="300" height="50"/>


## Advanced Model

We also did some researches on advanced deep learning models. The first family of advanced models that we implement is auto regressive models. Please refer to the following image from TensorFlow documentation to have a illustration on its mechanism.

On the high level, the model is warmed up with input features. It then starts to make predictions for multiple time steps. During each time step:
* It takes in inputs and outputs for previous time steps, and memorized states (LSTM only)
* It then make its prediction for this current time step
* It updates all the data mentioned above with the prediction of current time step to next time step

We hereby define abstract class `AutoRegressor` as follow:
```python
class AutoRegressor(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def warmup(self, inputs):
        return None

    @abstractmethod
    def call(self, inputs, training=None):
        return None
```

We define cells as the exact models doing prediction during warmup period and during each output time step.

We then implement this interface by deriving two subclasses, one is to implement the auto regressor model with LSTM as cell, another is to implement the auto regressor model with one dimensional CNN as cell.

For LSTM, during the warm up part we should return its predictions for all features plus label, as well as its memorized states. During prediction part, for each output time step:
* We use predicted features and states from last time step as input features to make the prediction of features and label for the current time step
* We then forward such predictions and states of the current time step to next time step

```python
class LSTMAutoRegressor(AutoRegressor):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(Constant.NUM_FEATURES + 1)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
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
```

For CNN, during the warm up part we should return its predictions for all features plus label. During prediction part, for each output time step:
* Unlike LSTM, CNN cannot memorize the previous features, so what we can do here is create a new feature set of shape (number of look back periods, number of features) for the current time step. We do this by dropping the most previous time step's features and concatenate the prediction of last time step's features into our feature set.
* We then forward such predictions of the current time step to next time step

> **Warning**   
> Only use operations provided by TensorFlow framework here. Using numpy operation or convert tensors to numpy arrays cannot work here as tensors can be symbolic during computation.

> **Warning**   
> Avoid accessing information of any tensors, because they can be symbolic during computation.

```python
class CNNAutoRegressor(AutoRegressor):
    def __init__(self, units, out_steps):
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
```


<img src="./img/multistep_autoregressive.png" alt="framework" title="framework" width="500" height="50"/>


# Portfolios for back testing

We define a portfolio as an combination of investment that:
* longs or shorts or longs shorts at the same time
* applied on a set of stocks, which are chosen based on certain model and data
* held over a certain time period

By concatenating portfolios across time axis, we mean:
* two or more portfolios defined as above
* are solely applied over their designated time period
* one after one continuously over time axis

For the utility of back testing our machine learning tasks in the client side, we also develop a family of `Portfolio` classes. A `Portfolio` object encapsulate all relevant data and behavior for a portfolio. At the highest level, abstract class `Portfolio` has 2 variables:
* `_performance`: it encapsulates all data and behavior for a portfolio's performance metric, represented by self defined `Performance` class objects
* `_scheme`: it is a dictionary specifying what stocks the current portfolio longs and what stocks the current portfolio shorts

Abstract class `Portfolio` also has 3 methods:
* `construct()`: abstract method, requires the subclasses to implement how a specific portfolio is constructed and furthermore updates the `_performance` and `_scheme` accordingly
* `add_portfolio()`: implements how to add a portfolio of the **same class** to the current `Portfolio` object. For some portfolios, this is as easy as add daily returns of new portfolio to the existing one and update the portfolio value accordingly based on new daily returns
* `clone()`: return a **deep copy** of current object

We derive 2 subclasses from abstract class `Portfolio`:
* `MultiStrategyPortfolio`: which is a container to concatenate different portfolios of different strategies (i.e. different subclasses of `Portfolio`) along the time axis
* `AlgoTradePortfolio`: which represents all portfolios traded algorithmically based on certain strategy. These objects are determined by `construct()`, which is further relied on two methods:
  * `_strategy()`: abstract method, it requires subclasses to implement how we choose specific stocks to long or short
  * `_apply_strategy()`: abstract method, it requires subclasses to implement how we long or short specific stocks provided by `_strategy()`

For `AlgoTradePortfolio`, we derive subclasses:
* `LongPortfolio`: which represents all long-only portfolio objects, it overrides `_apply_strategy()` and requires subclasses to implement `_strategy()`
* `DollarNeutralLongShortPortfolio`: which represents all dollar-neutral long short portfolio objects, it overrides `_apply_strategy()` and requires subclasses to implement `_strategy()`.
  * By dollar-neutral, we always assume we do the short sale on certain units of target stocks to long exactly one unit of target stocks with short sale proceeds. We keep these positions for a given period until rebalance. For each time step the portfolio is applying on, we calculate the long position value and short position value, thus net asset value. We calculate returns by formula (current net asset value - last net asset value) / last long value
  * The `add_portfolio()` overrides the super class method and implements a different logic. When add new portfolio to the current one, we use the existing portfolio's proceeds, which is the portfolio value of the last day to update the new portfolio's value and daily returns. If the existing portfolio's proceeds is positive, we use this part all to long new portfolio's choices of long stocks, thus we add the long value of this part to the original new portfolio's value and update daily returns accordingly. If the existing portfolio's proceeds is negative, we add this number to all portfolio values of the new portfolio.

Within `AlgoTradePortfolio` class and its subclasses, we also provide a static class variable `global_portfolio` to record all this kind of portfolios constructed across different time periods. We assume that portfolios are constructed based on consistent strategies (i.e. using the same `Portfolio` subclass) during a given back test period. However, under the circumstance when one insists on applying multiple strategies to different time periods within the back test period, we provide `MultiStrategyPortfolio` to manually combine portfolios of different strategies across time axis within the back test period.

We derive real concrete subclasses from `LongPortfolio` and `DollarNeutralPortfolio` and override `_strategy()` method:
* `LongBestPortfolio`: long the stock with the best performance predicted by the model
* `LongBestThreePortfolio`: long the three stock with best performance predicted by the model
* `LongRandomPortfolio`: long the stock randomly picked
* `LongMarketPortfolio`: long the market portfolio
* `LongShortBestShotPortfolio`: long the stock with the best performance predicted by the model and short the stock with the worst performance predicted by the model

<img src="./img/portfolio.png" alt="framework" title="framework" width="4000" height="500"/>