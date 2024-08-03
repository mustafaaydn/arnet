## ARNet: Nonlinear autoregression with feed-forward neural networks (or any estimator really)
A Python port of [nnetar.R](https://www.rdocumentation.org/packages/forecast/versions/8.23.0/topics/nnetar): nonlinear
autoregression with feed-forward neural networks. Here, we made it a bit more generic to use any kind of estimator
respecting the sklearn's interface, though the default is to use a single layer MLPRegressor from sklearn.

Following nnetar.R, the ordinal (`p`) and seasonal (`P`) lags to look back are automatically chosen if left
unspecified. Furthermore, number of hidden neurons in the default estimator is also chosen automatically following the
heuristic in nnetar.R.

#### Installation
```sh
pip install arnet
```

Below we present some snippets to illustrate the main usage; there is also [examples with a dataset](https://mustafaaydn.github.io/arnet/index.html) and the [API reference](https://mustafaaydn.github.io/arnet/arnet.html).

### Fit-predict flow
```py
from arnet import ARNet

y_train, y_test = ...

model = ARNet()
model.fit(y_train)
predictions = model.predict(n_steps=y_test.size)
```
[Instantaniate](https://mustafaaydn.github.io/arnet/arnet.html) the model, [fit](https://mustafaaydn.github.io/arnet/arnet.html#arnet.ARNet.fit) to data and [predict](https://mustafaaydn.github.io/arnet/arnet.html#arnet.ARNet.predict); that's all.

If you have side information, i.e., exogenous regressors to help in prediction, you can supply them like so:
```py
X_train, X_test, y_train, y_test = ...

model = ARNet()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

If you have seasonality in the series:
```py
# An automatic `P` (seasonal lags) will be chosen as left unspecified
model = ARNet(seasonality=12)
model.fit(...)
predictons = model.predict(...)
```

Default base model is an [`MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html); if you want to use another one, that's okay too:
```py
from sklearn.ensemble import RandomForestRegressor

model = ARNet(RandomForestRegressor(max_features=0.9))
model.fit(...).predict(...)
```
In fact, if you use [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), you
effectively have the linear AR(p) model as a subset! [Here](https://mustafaaydn.github.io/arnet/index.html#linear-ar-p-as-a-subset) is an example for that.

### Prediction intervals
Following the procedure in [FPP](https://otexts.com/fpp3/nnetar.html#prediction-intervals-5), the model is able to produce prediction intervals for the future given confidence levels:
```py
model = ARNet()
model.fit(y_train)
predictions, intervals = model.predict(n_steps=y_test.size, return_intervals=True, alphas=[80, 95])
```
You can also obtain the simulated paths by issuing `return_paths=True`.

### Validation
There is a [`.validate`](https://mustafaaydn.github.io/arnet/arnet.html#arnet.ARNet.validate) method to perform time series validation (expanding window) on a parameter grid with either a full search or a randomized one:
```py
X, y = ...
param_grid = {"p": [2, 5, 8, None], "estimator__solver": ["sgd", "adam"]}
n_iter = -1  # -1 (or None) means full grid search; any positive integer would mean a randomized search

search = ARNet.validate(y, param_grid, X=X, n_iter=n_iter)
best_model = search["best_estimator_"]
# Do something with the best model
```

### Plotting
There is also a helper static function for [plotting](https://mustafaaydn.github.io/arnet/arnet.html#arnet.ARNet.plot) lines -- it might be helpful in visualizing the true values along with the predictions and intervals.
```py
preds_in_sample = model.fitted_values_  # fitted values are available as a post-fit attribute
preds_out_sample, intervals = model.predict(n_steps=y_test.size, return_intervals=True, alphas=[80, 95])

ARNet.plot(lines=[y_train, preds_in_sample, y_test, preds_out_sample],
           labels=["y-train", "in-sample preds", "y-test", "out-sample preds"],
                   true_indexes=[0, 2],
                   intervals=intervals)
```
Here is an example plot output:
![example plot](tests/figures/example_plot.PNG)

For examples with a dataset in action, please see [here](https://mustafaaydn.github.io/arnet/index.html); for the API reference, see [here](https://mustafaaydn.github.io/arnet/arnet.html). Happy forecasting!
