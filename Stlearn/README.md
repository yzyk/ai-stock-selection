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