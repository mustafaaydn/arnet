"""
arnet.py: Generalized Python version of `nnetar.R <https://www.rdocumentation.org/packages/forecast/versions/8.23.0/topics/nnetar>`_
"""
import random
import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.utils.validation import check_X_y, check_scalar
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.seasonal import STL

NoneType = type(None)
_SENTINEL = object()
_DIFF_MLP_DEFAULTS = {"hidden_layer_sizes": None, "max_iter": 2_000}


class ARNet:
    """
    Nonlinear autoregression with feed-forward neural networks (or any estimator really)
    """
    def __init__(self, base_model=None, p=None, P=None, seasonality=None, repeats=20,
                 scale_inputs=True, scale_output=True, transform_output=False):
        """
        Parameters
        ----------
        base_model : Estimator, optional
           The underlying model used for fitting and predicting. Defaults to a single layer `MLPRegressor`, where the
           number of hidden neurons are chosen automatically and also with `max_iter` being 2_000.  Any model respecting
           the sklearn's estimator `interface <https://scikit-learn.org/stable/developers/develop.html>`_ is accepted.
        p : int or None, optional
            Number of lags to look behind to predict the next instance. Leaving it None leads to automatic selection:
            "the optimal number of lags (according to the AIC) for a linear AR(p) model", following nnetar.R.
        P : int or None, optional
            Number of seasonal lags to look behind to predict the next instance. Ignored if `seasonality` is None.  If
            `seasonality` is not None and `P` is None, it is automatically selected: same procedure as for `p` except
            the data subject to selection is seasonally-adjusted with an STL decomposition, following nnetar.R.
        seasonality : int or None, optional
            The periodicity suspected in the data, e.g., 12 for monthly series. `None` signals no seasonality. No
            automatic detection of seasonality is performed.
        repeats : int, default=20
            Number of base models to fit internally. These are then averaged when making forecasts to smooth out the
            inherent randomness in the model's nature, e.g., random network weights and stochastic learning
            algorithms for neural networks or bagging in random forests.
        scale_inputs : bool or Transformer, default=True
            Whether (or how) to scale the input to the networks. The input refers to the side information variables (X),
            if any. If False, no scaling is performed; if True (default), `sklearn.preprocessing.StandardScaler()` is
            applied. You can also pass a scaler of your choice, e.g., `MinMaxScaler()`, as long as it "quacks" like an
            sklearn transformer.
        scale_output : bool or Transformer, default=True
            Whether (or how) to scale the output variable, i.e., the target series (y). If False, no scaling is
            performed; if True (default), `sklearn.preprocessing.StandardScaler()` is applied. You can also pass a
            scaler of your choice, e.g., `MinMaxScaler()`, as long as it "quacks" like an sklearn transformer.
        transform_output : bool or Transformer, default=False
            Whether (or how) to transform the output variable, i.e., the target series (y). If False, no transformation
            is performed; if True, `sklearn.preprocessing.PowerTransformer("box-cox", standardize=False)`, i.e., a
            BoxCox transformation where lambda is automatically chosen, is applied. You can also pass a transformer of
            your choice, e.g., `FunctionTransformer(np.sqrt, inverse_func=np.square)`, as long as it "quacks" like an
            sklearn transformer. Note that if `scale_output` is on, this transformation is performed before that.

        Notes
        -----
        The default MLPRegressor has two different default parameters than that of sklearn: `hidden_layer_sizes` and
        `max_iter`, where the former is `None` to signal it's automatically determined (as in nnetar.R) and the latter
        is 2_000 as lower values tend to lead convergence issues on various datasets we tried. Of course, both can be
        changed on will, e.g., you can pass `base_model=MLPRegressor()` to have the sklearn's defaults (100 and 200,
        respectively).
        """
        if base_model is None:
            base_model = MLPRegressor(**_DIFF_MLP_DEFAULTS)
        self.base_model = base_model
        self._networks = [clone(base_model) for _ in range(repeats)]

        # ARNet-specific parameters
        self.p = p
        self.P = P
        self.seasonality = seasonality
        self.repeats = repeats
        self.scale_inputs = scale_inputs
        self.scale_output = scale_output
        self.transform_output = transform_output

    def __getattr__(self, attr):
        if (value := self._networks[0].get_params().get(attr, _SENTINEL)) is not _SENTINEL:
            return value
        raise AttributeError(attr)

    def fit(self, y, X=None):
        """
        Train the model on `y` with possibly side information `X`. Note that `fit` can be called either as fit(X, y) or
        fit(y, X?).

        If left unspecified, `p` and `P` are determined first. Then prepares the design matrix where the lagged values
        reside on the left and the side information, if any, on the right. Lastly, carries out the X and y scalings and
        transformations, if any, and fits `repeats` number of base models on them.

        Sets two public post-attributes: `fitted_values_all_` and `fitted_values_`, which are the in-sample predictions
        (the fitted values) of all the networks and the mean of all the networks, respectively.

        Parameters
        ----------
        y : pd.Series or np.ndarray
          The endogenous variable, i.e., the target time series
        X : pd.DataFrame or np.ndarray, optional
          The exogenous variables, i.e., the side information matrix

        Returns
        -------
        self
          The fitted instance

        Notes
        -----
        The primary way of calling this function is `fit(y, X?)` where ? signals optionality.  However, to comply with
        the sklearn's "normal" estimators' way of passing X first y next, the function also accepts `fit(X, y)` and
        arranges internally which are which.
        """
        check_scalar(self.repeats, "repeats", int, min_val=1)
        if X is None and y is None:
            raise ValueError("Both `X` and `y` are None, don't know what to fit over")
        # Set attributes with the assumption that `.fit(y, X?)` is how we are called
        has_X = X is not None and y is not None
        self._is_y_pandas = isinstance(y, pd.Series)
        self._X = X
        self._y = y
        self._has_X = has_X
        if has_X:
            # Allow calling `.fit(X, y)` or `.fit(y, X?)`
            try:
                # Given y, X
                X, y_as_np = check_X_y(X, y, force_all_finite="allow-nan")
            except ValueError:
                # Given X, y; refine the post-attributes' values
                self._is_y_pandas = isinstance(X, pd.Series)
                self._X = y
                self._y = X
                X, y_as_np = check_X_y(y, X, force_all_finite="allow-nan", estimator=self)
            # Cannot tolerate missing data in X
            if np.isnan(X).any():
                raise ValueError("The exogenous regressors contain NaN, cannot fit")
        else:
            # Only a single time series supplied, no X; figure out which is the series
            series = X if X is not None else y
            self._X = None
            self._y = series
            self._is_y_pandas = isinstance(series, pd.Series)
            y_as_np = np.asarray(series)
            if y_as_np.ndim > 1:
                raise ValueError(f"y should be a 1D array-like, got shape {y_as_np.shape}")

        # Cannot proceed if missing data
        if np.isnan(y_as_np).any():
            raise ValueError("The target series contains NaN, cannot fit")

        # Validate ARNet-specific parameters
        check_scalar(self.p, "p", (NoneType, int))
        check_scalar(self.P, "P", (NoneType, int))
        check_scalar(self.seasonality, "seasonality", (NoneType, int))

        # X/y-scalers and y-transformer should be either booleans or sklearnesque transformers
        def _check_io_modifier(mod, default_modifier_cls):
            if isinstance(mod, bool):
                return default_modifier_cls() if mod else None
            # Don't check if it really is a TransformerMixin; if it quacks like one, it is one...
            return mod
        # Defaults for scalers are StandardScaler() and for the y-transformer, it's BoxCox with an automatic lambda
        self._X_scaler = _check_io_modifier(self.scale_inputs, default_modifier_cls=StandardScaler)
        self._y_scaler = _check_io_modifier(self.scale_output, default_modifier_cls=StandardScaler)
        boxcox_transformer = partial(PowerTransformer, method="box-cox", standardize=False)
        self._y_transformer = _check_io_modifier(self.transform_output, default_modifier_cls=boxcox_transformer)

        # Figure out "automatic" values for p, P and number of layers (if not supplied)
        is_seasonal = not (self.seasonality is None or self.seasonality == 1)
        if self.p is None:
            # Automatic `p` is via the "best" AR(p) linear model's `p`
            data_to_find_p_over = y_as_np
            if is_seasonal:
                data_to_find_p_over = y_as_np - STL(y_as_np, period=self.seasonality).fit().seasonal
            self.p = self._select_ar_order(data_to_find_p_over)
        elif self.p < 0:
            raise ValueError(f"Number of lags `p` should be nonnegative, got {self.p}")
        elif self.p >= self._y.size:
            raise ValueError("Number of lags `p` is too large compared to the size of the series: "
                            f"{self.p} >= {self._y.size}; cannot form a lag matrix to fit")

        if self.P is None:
            self.P = 1 if is_seasonal else 0
        elif self.P < 0:
            raise ValueError(f"Number of seasonal lags `P` should be nonnegative, got {self.P}")
        elif not is_seasonal and self.P > 0:
            raise ValueError(f"Seasonal lags `P` given but seasonality is not provided")
        if is_seasonal and self.P * self.seasonality >= self._y.size:
            raise ValueError("Number of seasonal lags `P` is too large compared to the size of the series: "
                            f"{self.P} * {self.seasonality} >= {self._y.size}; cannot form a lag matrix to fit")
        self._is_seasonal = is_seasonal

        # Default hidden layer configuration: 1 layer with round((p + P + X.shape[1]) / 2) neurons
        if isinstance(self.base_model, MLPRegressor) and self.base_model.hidden_layer_sizes is None:
            hls = round((self.p + self.P + (X.shape[1] if has_X else 0) + 1) / 2)
            for network in self._networks:
                network.hidden_layer_sizes = hls

        # Figure out the lags for the lag matrix
        lags = [*range(1, self.p+1)]
        if self.P > 0:
            lags += range(self.seasonality, self.seasonality * (self.P + 1), self.seasonality)
        self._lags = lags

        # If the target is to be transformed/scaled, past values of the target going into X
        # should be, too, so we first deal with y. We also precalculate the lag erasion's effect
        # to avoid a small leak.
        lags_offset = max(lags, default=0)
        if self._y_transformer is not None:
            self._y_transformer.fit(y_as_np[lags_offset:, None])
            y_as_np = self._y_transformer.transform(y_as_np[:, None]).ravel()
        if self._y_scaler is not None:
            self._y_scaler.fit(y_as_np[lags_offset:, None])
            y_as_np = self._y_scaler.transform(y_as_np[:, None]).ravel()

        # Building the design matrix
        # | y_{t-1}, y_{t-2} ... | x_1, x_2 ... |
        if has_X:
            side_X = X[lags_offset:]
            if self._X_scaler is not None:
                side_X = self._X_scaler.fit_transform(side_X)
        if lags:
            # Lag matrix easier with pandas
            y_as_pd = pd.Series(y_as_np)
            the_X = pd.concat((y_as_pd.shift(lag) for lag in lags), axis=1).iloc[lags_offset:]
            if has_X:
                # Append X to the lags if exists
                the_X = np.hstack((the_X, side_X))
        elif not has_X:
            # No lags nor side information: error
            raise RuntimeError("There is no lag component (i.e., p = 0, P = 0) and no side information (X) "
                               "either; cannot fit (maybe series is too short?)")
        else:
            # No lags, only side information
            the_X = side_X
        the_y = y_as_np[lags_offset:]

        # These are the possibly scaled, transformed X, y values used for fitting
        self._X_fit = the_X
        self._y_fit = the_y

        insample_preds = []  # Gather in-sample predictions here upon fitting
        for network in self._networks:
            network.fit(the_X, the_y)
            insample_preds.append(self._predict(network, X=None, n_steps=None))

        # Register 2 post-fit attributes: fitted_values_ and fitted_values_all_
        # Former is the mean of all networks' in-sample predictions, second is, well, all of them, not aggregated
        self.fitted_values_all_ = pd.DataFrame(insample_preds).T if self._is_y_pandas else np.asarray(insample_preds).T
        self.fitted_values_ = self.fitted_values_all_.mean(axis=1)

        return self

    def predict(self, X=None, n_steps=None, return_intervals=False,
                n_paths=1_000, alphas=95, bootstrap=False, return_paths=False):
        """
        Perform multi-step recursive forecasting and optionally also calculate prediction intervals.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The side information to use for prediction horizon. If the training data had no `X`, it's an error to pass
            one here. Conversely, if training data did have `X`, it's an error to not pass one here.
        n_steps : int
            Number of steps to forecast ahead. If `X` is supplied, it can be inferred from `X.shape[0]`. It is an error
            to pass inconsistent `X` and `n_steps`, i.e., `n_steps` must be equal to `X.shape[0]` when both are passed.
        return_intervals : bool, default=False
            Whether to also calculate and return the prediction intervals. The procedure in
            https://otexts.com/fpp3/nnetar.html#prediction-intervals-5 is followed. If there is a single alpha provided
            in `alphas`, then a single-index column pd.DataFrame is returned with column labels "lower" and "upper". If
            there is more than one alpha, a multi-index column frame is returned, e.g., with levels [80, 95] outside and
            ["lower", "upper"] inside. The parameters below are ignored if `return_intervals` is False.
        n_paths : int, default=1_000
            Number of paths to generate from the model for the forecast horizon for the prediction intervals.  Ignored
            if `return_intervals` is False.
        alphas : int, list-like of int, default=95
            Percentage values in (0, 100) for the prediction intervals' quantile calculation. Ignored if
            `return_intervals` is False.
        bootstrap : bool, default=False
            If True, the noises are sampled from the in-sample errors; otherwise, from a normal distribution following
            the mean & std of the in-sample errors. Ignored if `return_intervals` is False.
        return_paths : bool, default=False
            Whether also return the simulated `n_paths` paths into the future. Ignored if `return_intervals` is False.

        Returns
        -------
        fores : pd.Series or np.ndarray
            Forecasts into the future. If the training `y` was a pandas Series, this is a pandas Series with an
            (hopefully) appropriate index; otherwise, it's a NumPy array.
        intervals : pd.DataFrame, optional
            Prediction intervals as either a single-index column frame with labels "lower" and "upper" if one alpha,
            where the `.name` of the columns is {alpha}%; or a multi-index one if many alphas, where the levels are the
            alphas outside and "lower", "upper" inside. If the training `y` was a pandas Series, the index of the frame
            is a continuation of that one (if possible); otherwise it has the default RangeIndex.  Only provided if
            `return_intervals` is True.
        paths : pd.DataFrame, optional
            Generated future paths of shape `n_steps` x `n_paths`. Only provided if `return_intervals` is True.

        Notes
        -----
        If the in-sample predictions are wanted, i.e., the fitted values during training, it can be obtained via the
        post-fit attributes `.fitted_values_` and `fitted_values_all_`, where the latter collects all of the internal
        `repeats` number of networks' fitted values and the former is the mean of them.
        """
        if not hasattr(self, "fitted_values_"):
            raise NotFittedError("This ARNet model has not been fitted yet, so cannot predict")
        if X is None and n_steps is None:
            if return_intervals:
                raise ValueError("Prediction intervals for in-sample predictions not available, sorry. "
                                 "Maybe your model had no X and you forgot to provide `n_steps` to `.predict`?")
            warnings.warn("No `n_steps` provided (nor an `X` if exogenous regressor were used in fitting) "
                          "for `predict`, which implies in-sample prediction. "
                          "Note that this is already available as a post-fit attribute `.fitted_values_`, "
                          "which is returned now")
            return self.fitted_values_
        predictions = [self._predict(network, X, n_steps) for network in self._networks]
        mean_preds = np.mean(predictions, axis=0)  # a NumPy array at this point
        # Form a prediction index if pandas
        if self._is_y_pandas:
            # not all pd.Indexes have "freq", e.g., Int64Index
            freq = getattr(self._y.index, "freq", None)
            if freq is None:
                try:
                    freq = pd.infer_freq(self._y.index)
                except TypeError:
                    pass
            if freq is not None:
                new_index = pd.date_range(self._y.index[-1], periods=mean_preds.size+1, freq=freq, inclusive="right")
            elif hasattr(X, "index"):
                new_index = X.index
            elif pd.api.types.is_integer_dtype(self._y.index) and self._y.index.is_monotonic_increasing:
                # Integer index, e.g., years
                new_index = pd.RangeIndex(self._y.index[-1] + 1, self._y.index[-1] + 1 + mean_preds.size)
            else:
                # Couldn't make something out of the index, so give it len(y_train) + 1 .. *
                warnings.warn("Couldn't infer a frequency for the series, "
                              "defaulting to RangeIndex (0..N-1) for the future")
                new_index = None
            mean_preds = pd.Series(mean_preds, new_index)

        if not return_intervals:
            return mean_preds
        intervals, paths = self._prediction_intervals(X, n_steps, n_paths, alphas, bootstrap,
                                                      mean_preds.index if self._is_y_pandas else None)
        if not return_paths:
            return mean_preds, intervals
        return mean_preds, intervals, paths

    def _predict(self, network, X, n_steps):
        """
        Perform the actual multi-step prediction recursively. Error checking and undoing scaling/transformation take
        place here.
        """
        # Check if X and n_steps are incompatible
        if X is not None and n_steps is not None and n_steps != X.shape[0]:
            raise ValueError(f"Provided both n_steps and X but they differ in length, {n_steps} != {X.shape[0]}")

        is_in_sample = X is None and n_steps is None
        if is_in_sample:
            # If both X & n_steps are missing, predict in-sample
            predictions = network.predict(self._X_fit)
        elif self._has_X and X is None:
            # Fitted with X but no X given for prediction
            raise ValueError(f"{self.__class__.__name__} was fitted with exogenous regressors "
                             "but you didn't supply a future X for prediction")
        elif not self._has_X and X is not None:
            # Fitted without X but X given for predictions
            extra_msg = ""
            if isinstance(X, int):
                extra_msg = f" (maybe you meant to pass `n_steps={X}`)?"
            raise ValueError(f"{self.__class__.__name__} was fitted without exogenous regressors "
                             "but you supplied a future X for prediction" + extra_msg)
        # Otherwise, forecast into the future recursively
        else:
            # Here, either [no X was fitted and no future X] or [X was fitted and future X]
            if n_steps is None:
                n_steps = X.shape[0]
            if self._has_X:
                # Forecast with future X
                if np.isnan(X).any(axis=None):
                    raise ValueError("The exogenous regressors contain NaN, cannot perform prediction")
                if X.shape[1] != self._X.shape[1]:
                    raise ValueError("Number of features in future X doesn't match that of in training: "
                                     f"{X.shape[1]} != {self._X.shape[1]}")
                X_as_np = np.asarray(X)
                required_past_lags = self._lags
                predictions = []
                if required_past_lags:
                    predictions = self._y_fit[-self._lags[-1]:].tolist()
                for h in range(n_steps):
                    X_h = X_as_np[[h]]
                    if self._X_scaler is not None:
                        X_h = self._X_scaler.transform(X_h)
                    past_ys = [predictions[-lag] for lag in required_past_lags]
                    the_X_h = np.hstack(([past_ys], X_h))
                    predictions.append(network.predict(the_X_h).item())
            else:
                # Model had no `X`, i.e., lags only
                required_past_lags = self._lags
                predictions = self._y_fit[-self._lags[-1]:].tolist()
                for h in range(n_steps):
                    past_ys = [predictions[-lag] for lag in required_past_lags]
                    the_X_h = np.array(past_ys)[None]
                    predictions.append(network.predict(the_X_h).item())
            predictions = predictions[-n_steps:]

        predictions = np.asarray(predictions)

        # Un-scale and un-transform y if corresponding actions were made in training
        if self._y_scaler is not None:
            predictions = self._y_scaler.inverse_transform(predictions[:, None]).ravel()

        if self._y_transformer is not None:
            predictions = self._y_transformer.inverse_transform(predictions[:, None]).ravel()

        # If in sample predictions are made ready, wrap and index if pandas here
        if is_in_sample and self._is_y_pandas:
            predictions = pd.Series(predictions, self._y.index[-predictions.size:])

        return predictions

    def _prediction_intervals(self, X, n_steps, n_paths, alphas, bootstrap, y_index):
        """
        Prediction intervals' calculation following https://otexts.com/fpp3/nnetar.html#prediction-intervals-5
        """
        raw_fitted_values = np.mean([network.predict(self._X_fit) for network in self._networks], axis=0)
        residuals = self._y_fit - raw_fitted_values
        if bootstrap:
            noise = lambda: np.random.choice(residuals, size=1).item()
        else:
            loc, scale = np.mean(residuals), np.std(residuals)
            noise = lambda: np.random.normal(loc, scale)

        # In-sample prediction intervals should raise an error before here in `predict` already
        assert not self._has_X or (X is not None or n_steps is not None),\
            "In sample prediction intervals?! Should have errored already in `predict`"
        lags_only = False
        if X is None:
            # Only past lags
            lags_only = True
        elif n_steps is None:
            # Side information + past_lags
            n_steps = X.shape[0]
        else:
            assert X is not None or n_steps is not None, "possimpible happened?!"
            assert n_steps == X.shape[0], "X - n_steps length mismatch"
        n_lags = len(self._lags)

        # Prepare the first X for prediction
        if X is not None:
            X = np.asarray(X)
        y_train_raw = self._y_fit  # Take from the possibly scaled & transformed `y`
        first_step_x = y_train_raw[-np.array(self._lags)][None]
        if not lags_only:
            if self._X_scaler is not None:
                X = self._X_scaler.transform(X)
            first_step_x = np.hstack((first_step_x, X[[0]]))

        all_path_preds = []
        max_lag = max(self._lags)
        for _ in range(n_paths):
            current_x = first_step_x
            path_preds = y_train_raw[-max_lag:].tolist()
            for step in range(n_steps):
                next_pred = np.mean([network.predict(current_x).item() for network in self._networks])
                next_pred += noise()
                path_preds.append(next_pred)
                current_x[0, :n_lags] = [path_preds[-lag] for lag in self._lags]
                if not lags_only and step + 1 != n_steps:
                    current_x[0, n_lags:] = X[step + 1]
            # Strip off the history lags that helped in sliding window
            path_preds = np.asarray(path_preds[max_lag:])
            assert len(path_preds) == n_steps
            # Unscale and untransform the predictions, if any were done
            if self._y_scaler is not None:
                path_preds = self._y_scaler.inverse_transform(path_preds[:, None]).ravel()
            if self._y_transformer is not None:
                path_preds = self._y_transformer.inverse_transform(path_preds[:, None]).ravel()
            all_path_preds.append(path_preds)

        # Quantilify the preds to really produce intervals
        is_single_alpha = not pd.api.types.is_list_like(alphas)
        if not is_single_alpha:
            alphas = np.asarray(alphas)
        lower = np.quantile(all_path_preds, 0.5 - alphas / 200, axis=0, method="median_unbiased")
        upper = np.quantile(all_path_preds, 0.5 + alphas / 200, axis=0, method="median_unbiased")

        # Make a frame out of paths too; this has the shape n_steps x n_paths
        all_path_preds = pd.DataFrame(all_path_preds).T.set_axis(y_index, axis=0)

        if is_single_alpha:
            return (pd.DataFrame({"lower": lower, "upper": upper},
                                 index=y_index).rename_axis(columns=f"{alphas}%"),
                    all_path_preds)
        # More than one level is requested; intersperse the lower and upper values of them
        intervals = np.vstack((lower, upper))[np.arange(2*len(alphas)).reshape(-1, 2, order="F").ravel()]
        return (pd.DataFrame(intervals.T,
                             columns=pd.MultiIndex.from_product([alphas, ("lower", "upper")]),
                             index=y_index),
                all_path_preds)

    @staticmethod
    def _select_ar_order(endog, max_lag=None):
        """
        Select the "best" AR order for `endog` where the lag chosen is in range 0 to `max_lag` inclusive and the metric
        for selection is AIC. `statsmodels.tsa.ar_model.ar_select_order` doesn't use Yule-Walker's method in producing
        the coefficients, which the R code nnetar.R uses does,
        (https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/ar), so we wrote our own here. If `max_lag`
        is None, it's `min(n-1, 10*log(n))`, where n is the number of observations in `endog`.

        Notes
        -----
        If `endog` has `.size` 0, a ValueError is raised. If it has size 1, 0 is returned with a warning. If no lag can
        produce a finite AIC, 0 is returned with a warning.
        """
        endog = np.asarray(endog)
        n = endog.size
        if n == 0:
            raise ValueError("Series should have at least one observation to select an AR order")
        if n == 1:
            warnings.warn("Series has only 1 observation, 0 is returned as the optimal AR lag")
            return 0
        if max_lag is None:
            max_lag = min(n - 1, int(10 * np.log10(n)))
        best_lag = None
        best_aic = float("inf")
        for lag in range(max_lag+1):
            _, sigma = yule_walker(endog, lag, method="mle")
            # This isn't exactly AIC, but all we need is comparisons, so constant factors are taken out
            aic = lag + n*np.log(sigma)
            if aic < best_aic:
                best_aic = aic
                best_lag = lag
        if best_lag is None:
            warnings.warn("Couldn't find a suitable AR order to give finite AIC, returning 0")
            return 0
        return best_lag

    @classmethod
    def validate(cls, y, param_grid, X=None, base_model_cls=None, n_iter=-1, metric=mse,
                 cv=5, refit=True, random_state=None):
        """
        Perform time series cross validation with randomized (or full) grid search.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            The training target series to perform the validation over.
        param_grid : dict or a list of dicts
            Parameter combinations to try as either a dictionary, e.g., `{"p": [1, 2, 3], "estimator__solver": ["adam",
            "sgd"]}` or a list of dictionaries, e.g., `[{"p": 1, "estimator__solver": "adam"}, {...}]`. Note that the
            base model's parameters are passed with an "estimator\__" prefix.
        X : pd.DataFrame, np.ndarray or None, optional
            The accompanying exogenous variables for validation, if any.
        base_model_cls : Estimator class, optional
            Uninstantiated model class, e.g., `MLPRegressor`. Defaults to a single-layer MLP where the number of hidden
            neurons are chosen automatically, i.e., the default base model when `ARNet()` is called.
        n_iter : int or None, default=-1
            Either a positive integer to indicate the number of samples to randomly draw from `param_grid`, or one of
            `{-1, None}` to imply a full grid search.
        metric : callable, default=sklearn.metrics.mean_squared_error
            The validation metric to determine the fold scores. It should adapt "lesser is better".
        cv : int or sklearn.model_selection.TimeSeriesSplit, default=5
            Determines how the folds are constructed. If an integer, `TimeSeriesSplit(n_splits=cv)` is constructed. For
            a finer control, e.g., the test folds' sizes, one can pass a custom TimeSeriesSplit here.
        refit : bool, default=True
            Whether to make available a "best" model out of the parameter combinations that performed the best.  The
            model will be trained over the entire `y` (and `X` if available), and will be returned with the key
            "best_estimator\_".
        random_state : int or None, optional
            The seed used for sampling configurations from `param_grid` if randomized search is opted for with a
            nonnegative `n_iter`. Seeds the stdlib's `random`. By default, no reproducibility is aimed.

        Returns
        -------
        search_results : dict
            A dictionary with the keys
             * `scores_`: a dictionary of parameter-config (tuple) -> fold scores;
             * `best_params_`: a dictionary of parameter (str) -> value, which performed the best ("validated"
               parameters);
             * `best_estimator_`: an ARNet model that has `best_params_` and trained over the entire `y` (and `X`). Only
               provided if `refit` is True.
        """
        if n_iter is not None and not pd.api.types.is_integer(n_iter):
            raise ValueError(f"`n_iter` needs to be an integer (or None), got type {type(n_iter).__name__}")

        if n_iter in (-1, None):
            # Full grid search
            grid = ParameterGrid(param_grid)
        elif n_iter <= 0:
            raise ValueError("If integer, `n_iter` needs to be either -1 (grid search) "
                             "or positive integer (randomized search), "
                             f"got {n_iter}")
        else:
            # Randomized grid search
            random.seed(random_state)
            # ParameterSampler of sklearn might hit overflow, so we use this
            # https://stackoverflow.com/a/77511713
            grid = [{param_name: random.choice(param_possibilities)
                     for param_name, param_possibilities in param_grid.items()}
                    for _ in range(n_iter)]

        if cv is None:
            cv = TimeSeriesSplit(5)
        elif pd.api.types.is_integer(cv):
            cv = TimeSeriesSplit(cv)

        y_as_np = np.asarray(y)
        if X is not None:
            X = np.asarray(X)

        def _separate_config(conf, prefix="estimator__"):
            base_model_conf, arnet_specific_conf = {}, {}
            len_pref = len(prefix)
            for param, val in conf.items():
                if param.startswith(prefix):
                    base_model_conf[param[len_pref:]] = val
                else:
                    arnet_specific_conf[param] = val
            return base_model_conf, arnet_specific_conf

        if base_model_cls is None:
            base_model_cls = partial(MLPRegressor, **_DIFF_MLP_DEFAULTS)

        all_scores = {}
        for config in grid:
            config_scores = []
            base_model_config, arnet_specific_config = _separate_config(config)
            for train_inds, test_inds in cv.split(y_as_np):
                model = cls(base_model_cls(**base_model_config), **arnet_specific_config)
                model.fit(y_as_np[train_inds], X[train_inds] if X is not None else None)
                preds = model.predict(X=X[test_inds] if X is not None else None, n_steps=len(test_inds))
                score = metric(y_as_np[test_inds], preds)
                config_scores.append(score)
            # Hidden layer sizes is the only potentially list-like parameter, so we tuplify it if it's so,
            # so that it's hashable when put into the scores dictionary
            if pd.api.types.is_list_like(hls := config.get("estimator__hidden_layer_sizes"), allow_sets=False):
                config["estimator__hidden_layer_sizes"] = tuple(config["estimator__hidden_layer_sizes"])
            all_scores[tuple(config.items())] = config_scores

        best_params = dict(min(all_scores, key=lambda p: np.mean(all_scores[p])))
        if not refit:
            return {"scores_": all_scores, "best_params_": best_params}

        base_model_params, arnet_specific_params = _separate_config(best_params)
        best_model = ARNet(base_model_cls(**base_model_params), **arnet_specific_params)
        best_model.fit(y, X)
        return {"scores_": all_scores, "best_params_": best_params, "best_estimator_": best_model}

    @staticmethod
    def plot(lines, labels=None, true_indexes=0, intervals=None, paths=None, figure_kwargs=None, figure=None,
             show=True):
        """
        Plot `lines` with either a plotly or a matplotlib backend. Might be useful for plotting true values with
        predictions as well as the prediction intervals and simulated paths.

        Parameters
        ----------
        lines : list-like of list-likes
            List of series to plot.
        labels : list-like of str, optional
            Corresponding labels (names) of the lines to appear on the legend.
        true_indexes : int or list-like of ints or None, default=0
            Indexes of the "true" values in the list `lines`, i.e., the location(s) of the target series. Used for
            marking the non-true series with dashes to distinguish them better from true values. Can pass `None` to make
            all lines appear sans dashes. Default is to assume 0th element of `lines` has a true series.
        intervals : pd.DataFrame, optional
            The prediction intervals obtained through `.predict(..., return_intervals=True)`. If supplied, the intervals
            with the corresponding confidence levels are plotted as shaded areas with differing transparency.
        paths : pd.DataFrame, optional
            The simulated paths obtained through `.predict(..., return_paths=True)`. If supplied, the generated future
            paths are plotted out. Expected to be of shape `(n_steps, n_paths)`.
        figure_kwargs : dict, optional
            Configuration options for the figure displayed. If using plotly, passed to `fig.update_layout`; if
            matplotlib, passed to `plt.setp`.
        figure : plotly.graph_objects.Figure or matplotlib.pyplot.figure, optional
            Pre-prepared figure to perform the plotting on.
        show : bool, default=True
            Whether to show the painted figure immediately.

        Returns
        -------
        fig : plotly.graph_objects.Figure or matplotlib.pyplot.figure, optional
            The figure plotting happened on.

        Notes
        -----
        This function needs either plotly or matplotlib to be installed, and the libraries are tried in that
        order. If none of them is available, a RuntimeError is raised. If `show` is True (default), the figure is
        immediately displayed in addition to being returned. If using the matplotlib backend, `plt.setp` is not as
        encompassing as plotly's `.update_layout`, so it might be better to set `show=False`, capture the return value
        (the figure), apply figure updates over plt/fig/fig.gca() and show manually with `fig.show`.
        """
        use_matplotlib = False
        try:
            import plotly.graph_objects as go
        except ImportError:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise RuntimeError("To plot, either \"plotly\" or \"matplotlib\" should be installed")
            else:
                use_matplotlib = True

        if not pd.api.types.is_list_like(true_indexes):
            true_indexes = [true_indexes]

        fig = figure or (plt.figure() if use_matplotlib else go.Figure())
        index = next((t.index for t in lines if hasattr(t, "index")), [*range(lines[0].size)])
        for idx, (line, label) in enumerate(zip(lines, labels if labels else [None] * len(lines))):
            line_style = None if idx in true_indexes else ("--" if use_matplotlib else {"dash": "dash"})
            x = index if not hasattr(line, "index") else line.index
            y = line
            if use_matplotlib:
                plt.plot(x, y, ls=line_style, label=label)
            else:
                fig.add_trace(go.Scatter(x=x, y=y, line=line_style, name=label))
        if labels and use_matplotlib:
            plt.legend()

        if figure_kwargs:
            if use_matplotlib:
                plt.setp(fig, **figure_kwargs)
            else:
                fig.update_layout(**figure_kwargs)

        if intervals is None and paths is None:
            if not show:
                return fig
            if use_matplotlib:
                plt.show()
            else:
                fig.show()
            return fig

        # Plot the confidence intervals
        if intervals is not None:
            if not isinstance(intervals, pd.DataFrame):
                raise ValueError("`intervals` is expected to be a pandas DataFrame (e.g., returned by `predict` "
                                 f"with `return_intervals=True`) but it has type {type(intervals)}")
            n_conf_levels = intervals.columns.nlevels
            if n_conf_levels == 1:
                intervals = intervals.set_axis(pd.MultiIndex.from_product([[intervals.columns.name.rstrip("%")],
                                                                           intervals.columns]), axis=1)
            fores_index = intervals.index
            transparency = intervals.shape[1] // 2 * 0.1
            for conf_level, interval_df in intervals.T.groupby(level=0):
                interval_df = interval_df.droplevel(0)
                lower, upper = interval_df.loc["lower"], interval_df.loc["upper"]
                if use_matplotlib:
                    plt.fill_between(fores_index, lower, upper,
                                     color="b", alpha=transparency, label=f"{conf_level}% PI")
                else:
                    fig.add_traces([go.Scatter(x=fores_index, y=upper,
                                               hovertemplate=f"{conf_level}% " "Upper: %{y}<extra></extra>",
                                               mode="lines", line_color="rgba(0, 0, 0, 0)",
                                               showlegend=False),
                                    go.Scatter(x=fores_index, y=lower,
                                               hovertemplate=f"{conf_level}% " "Lower: %{y}<extra></extra>",
                                               mode="lines", line_color="rgba(0, 0, 0, 0)",
                                               name=f"{conf_level}% PI",
                                               fill="tonexty", fillcolor=f"rgba(0, 0, 255, {transparency})")])
                transparency -= 0.1

        # Plot the generated future paths
        if paths is not None:
            if not isinstance(paths, pd.DataFrame):
                raise ValueError("`paths` is expected to be a pandas DataFrame (e.g., returned by `predict` "
                                 f"with `return_paths=True`) but it has type {type(paths)}")
            fores_index = paths.index
            for path_no in paths:
                path = paths[path_no]
                if use_matplotlib:
                    plt.plot(fores_index, path)
                else:
                    fig.add_trace(go.Scatter(x=fores_index, y=path, showlegend=False))
        if use_matplotlib:
            plt.legend()
        if not show:
            return fig

        if use_matplotlib:
            plt.show()
        else:
            fig.show()
        return fig

    def __repr__(self):
        """
        Provide a back-evalable representation for the model. First comes the ARNet specific parameters, e.g., p, P,
        and then comes MLPRegressor's representation.
        """
        _INIT_PARAMS = ("base_model", "p", "P", "seasonality",
                        "repeats", "scale_inputs", "scale_output", "transform_output")
        defaults_for_specifics = dict(zip(_INIT_PARAMS, self.__init__.__defaults__))
        params_to_repr = {}
        for arnet_param in _INIT_PARAMS:
            if arnet_param == "base_model":
                # special handling afterwards
                continue
            # Check if the real (possibly updated) value differs from the original (default) one
            if (arnet_real_value := getattr(self, arnet_param)) != defaults_for_specifics[arnet_param]:
                params_to_repr[arnet_param] = arnet_real_value
        # If there was no seasonality, P=0 is determined, which doesn't look good on repr, so we filter that
        if not getattr(self, "_is_seasonal", True) and "P" in params_to_repr:
            assert params_to_repr["P"] == 0, "no seasonal should give 0 P"
            del params_to_repr["P"]
        # Put in front the base model's representation; base model is possibly updated, e.g., hls of MLPRegressor
        params_to_repr = {"base_model": repr(self._networks[0]), **params_to_repr}
        return "ARNet(" + ", ".join(f"{par}={val}" for par, val in params_to_repr.items()) + ")"
