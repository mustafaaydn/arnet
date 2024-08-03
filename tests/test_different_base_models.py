import unittest
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.arnet import ARNet


class TestDifferentBaseModels(unittest.TestCase):
    def setUp(self):
        # Yearly sunspots dataset as the y-only, and daily wind speed as the X-y one
        sunspots = (pd.read_csv("tests/datasets/sunspots.csv")
                      .pipe(lambda fr: fr.set_axis(pd.PeriodIndex(fr.pop("Year"), freq="Y").to_timestamp()))
                      .squeeze())  # https://rdrr.io/r/datasets/sunspot.year.html
        wind_speed_X = pd.read_csv("tests/datasets/wind_speed.csv", index_col="date", parse_dates=["date"],
                                   nrows=1_000)  # https://www.kaggle.com/fedesoriano/wind-speed-prediction-dataset
        wind_speed_y = wind_speed_X.pop("wind_speed")
        self.y_only_dataset = {"y_train": sunspots.iloc[:-12], "y_test": sunspots.iloc[-12:]}
        self.y_and_X_dataset = {"X_train": wind_speed_X.iloc[:-200], "X_test": wind_speed_X.iloc[-200:],
                                "y_train": wind_speed_y.iloc[:-200], "y_test": wind_speed_y.iloc[-200:]}

    def test_different_base_models(self):
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            LGBMRegressor = None
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        X_train, y_train, X_test = itemgetter("X_train", "y_train", "X_test")(self.y_and_X_dataset)

        base_models = [RandomForestRegressor(), LinearRegression()]
        if LGBMRegressor is not None:
            base_models.append(LGBMRegressor())

        # Fit-predict flow
        for base_model in base_models:
            # y-only dataset
            try:
                model = ARNet(base_model, repeats=1).fit(y_train.to_numpy())
                fores = model.predict(n_steps=12)
            except Exception as e:
                self.fail(f"(y-only data) Couldn't fit-predict with {base_model.__class__.__name__} "
                          f"as the base model: {e}")
            else:
                self.assertIsNot(base_model, model._networks[0], "passed base model is cloned")
                self.assertEqual(fores.size, 12)
                self.assertIsInstance(fores, np.ndarray)
                self.assertEqual(fores.ndim, 1)

            # X-y dataset
            try:
                model = ARNet(base_model, repeats=1).fit(X_train, y_train)
                fores = model.predict(X_test)
            except Exception as e:
                self.fail(f"(X-y data) Couldn't fit-predict with {base_model.__class__.__name__} "
                          f"as the base model: {e}")
            else:
                self.assertIsNot(base_model, model._networks[0], "passed base model is cloned")
                self.assertEqual(fores.size, len(X_test))
                self.assertTrue(fores.index.equals(X_test.index))

        # Validation with a RandomForestRegressor
        p, max_features = 5, 0.7
        search = ARNet.validate(y_train,
                                base_model_cls=RandomForestRegressor,
                                param_grid={"p": [p], "estimator__max_features": [max_features]},
                                cv=TimeSeriesSplit(3, test_size=10))
        self.assertIsInstance(search, dict)
        self.assertEqual(sorted(search.keys()), ["best_estimator_", "best_params_", "scores_"])
        best_params = search["best_params_"]
        self.assertIsInstance(best_params, dict)
        self.assertEqual(sorted(best_params.keys()), ["estimator__max_features", "p"])
        self.assertEqual(best_params["p"], p)
        self.assertEqual(best_params["estimator__max_features"], max_features)

        scores = search["scores_"]
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 1)
        self.assertEqual(sorted(next(iter(scores.keys()))), [("estimator__max_features", max_features), ("p", p)])

        best_estimator = search["best_estimator_"]
        self.assertIsInstance(best_estimator, ARNet)
        self.assertTrue(hasattr(best_estimator, "fitted_values_"))

        # Repr with a different base model
        model = ARNet(RandomForestRegressor())
        self.assertEqual(repr(model), "ARNet(base_model=RandomForestRegressor())")

        model = ARNet(RandomForestRegressor(min_samples_leaf=20), scale_inputs=False)
        self.assertEqual(repr(model),
                         "ARNet(base_model=RandomForestRegressor(min_samples_leaf=20), scale_inputs=False)")

        model = ARNet(RandomForestRegressor(n_estimators=50, max_features=0.9), p=3)
        self.assertEqual(repr(model), "ARNet(base_model=RandomForestRegressor(max_features=0.9, n_estimators=50), p=3)")

        ## Prediction intervals with a different base model
        # Multiple alphas
        alphas = 95, 80
        n_steps = 5
        rv = (ARNet(LinearRegression())
                .fit(y_train)
                .predict(n_steps=n_steps, alphas=alphas, return_intervals=True, n_paths=10))
        self.assertEqual(len(rv), 2)
        self.assertIsInstance(rv[1], pd.DataFrame)
        self.assertEqual(rv[1].shape, (n_steps, 2*len(alphas)))
        self.assertEqual(rv[1].columns.nlevels, 2)
        self.assertEqual(rv[1].columns.size, 2*len(alphas))
        self.assertEqual(rv[1].columns.tolist(),
                         [(alpha, side) for alpha in alphas for side in ("lower", "upper")])
        # Single alpha and also paths
        alpha = 80
        n_steps, n_paths = 5, 10
        rv = (ARNet(RandomForestRegressor(n_estimators=50, max_features=0.9), p=3)
                 .fit(y_train)
                 .predict(n_steps=n_steps, alphas=alpha,
                          return_intervals=True, return_paths=True, n_paths=n_paths))
        self.assertEqual(len(rv), 3)
        self.assertIsInstance(rv[2], pd.DataFrame)
        self.assertEqual(rv[2].shape, (n_steps, n_paths))
        self.assertEqual(rv[2].columns.tolist(), [*range(n_paths)])
