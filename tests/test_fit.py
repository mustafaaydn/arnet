import unittest
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from src.arnet import ARNet


class TestFit(unittest.TestCase):
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

    def test_fit_no_X_y(self):
        with self.assertRaisesRegex(ValueError, "Both `X` and `y` are None"):
            ARNet().fit(None, None)

    def test_fit_bad_y(self):
        good_y = self.y_only_dataset["y_train"]
        corrupt_y = good_y.copy()
        corrupt_y.iloc[0] = np.nan
        with self.assertRaisesRegex(ValueError, "The target series contains NaN, cannot fit"):
            ARNet().fit(corrupt_y)
        with self.assertRaisesRegex(ValueError, "y should be a 1D array-like, got shape"):
            ARNet().fit(good_y.to_numpy()[None])
        with self.assertRaisesRegex(ValueError, "y should be a 1D array-like, got shape"):
            ARNet().fit(good_y.to_numpy()[:, None])

    def test_fit_bad_X(self):
        good_X, good_y = itemgetter("X_train", "y_train")(self.y_and_X_dataset)
        corrupt_X = good_X.copy()
        corrupt_X.iat[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "The exogenous regressors contain NaN, cannot fit"):
            ARNet().fit(corrupt_X, good_y)
        with self.assertRaisesRegex(ValueError, "Found array with dim 3. ARNet expected <= 2."):
            ARNet().fit(good_X.to_numpy()[None], good_y)
        with self.assertRaisesRegex(ValueError, "Found array with dim 3. ARNet expected <= 2."):
            ARNet().fit(good_X.to_numpy()[:, None], good_y)

    def test_fit_too_short_y(self):
        with self.assertRaisesRegex(ValueError, "Number of lags `p` is too large compared to the size of the series"):
            ARNet(p=10).fit(self.y_only_dataset["y_train"].iloc[:5])
        with self.assertRaisesRegex(RuntimeError, r"cannot fit \(maybe series is too short\?\)"):
            ARNet().fit(self.y_only_dataset["y_train"].iloc[:1])

    def test_fit_bad_pP(self):
        y_train = self.y_only_dataset["y_train"]
        # Negative p, P
        with self.assertRaisesRegex(ValueError, "Number of lags `p` should be nonnegative"):
            ARNet(p=-2).fit(y_train)
        with self.assertRaisesRegex(ValueError, "Number of seasonal lags `P` should be nonnegative"):
            ARNet(P=-2).fit(y_train)
        # P without seasonality
        with self.assertRaisesRegex(ValueError, "Seasonal lags `P` given but seasonality is not provided"):
            ARNet(seasonality=None, P=3).fit(y_train)
        # Too large p, P
        with self.assertRaisesRegex(ValueError, "Number of lags `p` is too large"):
            ARNet(seasonality=None, p=y_train.size + 20).fit(y_train)
        with self.assertRaisesRegex(ValueError, "Number of seasonal lags `P` is too large"):
            ARNet(seasonality=12, P=y_train.size).fit(y_train)
        # Noninteger p, P
        with self.assertRaisesRegex(TypeError, "p must be an instance of {NoneType, int}, not float"):
            ARNet(p=7.48).fit(y_train)
        with self.assertRaisesRegex(TypeError, "P must be an instance of {NoneType, int}, not float."):
            ARNet(P=365.24, seasonality=7).fit(y_train)

    def test_fit_automatic_p(self):
        y = pd.concat(itemgetter("y_train", "y_test")(self.y_only_dataset))
        model = ARNet().fit(y)
        self.assertEqual(model.p, 9)

    def test_fit_automatic_hls(self):
        y = pd.concat(itemgetter("y_train", "y_test")(self.y_only_dataset))
        model = ARNet().fit(y)
        self.assertEqual(model.hidden_layer_sizes, 5)

    def test_fit_non_automatic_pPhls(self):
        y_train = self.y_only_dataset["y_train"]
        p, P, hls = 77, 4, (54, 32, 8)
        model = ARNet(p=p).fit(y_train)
        self.assertEqual(model.p, p)
        model = ARNet(seasonality=7, P=P).fit(y_train)
        self.assertEqual(model.P, P)
        model = ARNet(MLPRegressor(hidden_layer_sizes=hls)).fit(y_train)
        self.assertEqual(model.hidden_layer_sizes, hls)

    def test_fit_lags(self):
        y_train = self.y_only_dataset["y_train"]
        model = ARNet(p=6).fit(y_train)
        self.assertEqual(model._lags, [1, 2, 3, 4, 5, 6])
        model = ARNet(p=0, P=5, seasonality=7).fit(y_train)
        self.assertEqual(model._lags, [7, 14, 21, 28, 35])
        model = ARNet(p=2, P=3, seasonality=12).fit(y_train)
        self.assertEqual(model._lags, [1, 2, 12, 24, 36])

    def test_fit_return(self):
        model = ARNet()
        self.assertIs(model, model.fit(self.y_only_dataset["y_train"]))

    def test_fit_post_attributes(self):
        model = ARNet().fit(self.y_only_dataset["y_train"])
        self.assertTrue(hasattr(model, "fitted_values_"))
        self.assertTrue(hasattr(model, "fitted_values_all_"))

    def test_fit_fit(self):
        y_train = self.y_only_dataset["y_train"]
        model = ARNet().fit(y_train)
        try:
            model.fit(y_train)
        except Exception as e:
            self.fail(f"Couldn't refit over the same data the second time: {e}")
