import unittest
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from src.arnet import ARNet


class TestPredict(unittest.TestCase):
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

    def test_predict_without_fitting(self):
        with self.assertRaisesRegex(NotFittedError, "This ARNet model has not been fitted yet, so cannot predict"):
            ARNet().predict(n_steps=5)

    def test_predict_no_X_y_but_intervals(self):
        X_train, y_train = itemgetter("X_train", "y_train")(self.y_and_X_dataset)
        model_wX = ARNet().fit(X_train, y_train)
        with self.assertRaisesRegex(ValueError, "Prediction intervals for in-sample predictions not available, sorry"):
            model_wX.predict(X=None, n_steps=None, return_intervals=True)
        model_woX = ARNet().fit(y_train)
        with self.assertRaisesRegex(ValueError, "Prediction intervals for in-sample predictions not available, sorry"):
            model_woX.predict(n_steps=None, return_intervals=True)

    def test_predict_in_sample(self):
        with self.assertWarnsRegex(UserWarning, "implies in-sample prediction"):
            ARNet().fit(self.y_only_dataset["y_train"]).predict()

    def test_predict_bad_X(self):
        X_train, y_train, X_test = itemgetter("X_train", "y_train", "X_test")(self.y_and_X_dataset)
        bad_X_test = X_test.copy()
        bad_X_test.iat[0, 0] = np.nan
        model = ARNet().fit(X_train, y_train)
        with self.assertRaisesRegex(ValueError, "The exogenous regressors contain NaN, cannot perform prediction"):
            model.predict(bad_X_test)

    def test_predict_incompatible_X_nsteps(self):
        X_train, y_train, X_test = itemgetter("X_train", "y_train", "X_test")(self.y_and_X_dataset)
        model = ARNet().fit(X_train, y_train)
        with self.assertRaisesRegex(ValueError, "Provided both n_steps and X but they differ in length"):
            model.predict(X_test, n_steps=X_test.shape[0] + 40)

    def test_predict_X_fit_but_no_predict(self):
        X_train, y_train = itemgetter("X_train", "y_train")(self.y_and_X_dataset)
        model = ARNet().fit(X_train, y_train)
        with self.assertRaisesRegex(ValueError,
                                    "fitted with exogenous regressors but you didn't supply a future X for prediction"):
            model.predict(n_steps=5)

    def test_predict_no_X_fit_but_yes_predict(self):
        X_test, y_train = itemgetter("X_test", "y_train")(self.y_and_X_dataset)
        model = ARNet().fit(y_train)
        with self.assertRaisesRegex(ValueError,
                                    "fitted without exogenous regressors but you supplied a future X for prediction"):
            model.predict(X_test)

    def test_predict_X_fit_but_incompatible_n_features(self):
        X_train, y_train, X_test = itemgetter("X_train", "y_train", "X_test")(self.y_and_X_dataset)
        model = ARNet().fit(X_train, y_train)
        with self.assertRaisesRegex(ValueError, "Number of features in future X doesn't match that of in training"):
            model.predict(X_test.iloc[:, 1:])

    def test_predict_return(self):
        y_train = self.y_only_dataset["y_train"]
        preds = ARNet().fit(y_train).predict(n_steps=5)
        self.assertIsInstance(preds, pd.Series)
        preds = ARNet().fit(y_train.to_numpy()).predict(n_steps=5)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.ndim, 1)

    def test_prediction_index(self):
        y_train = self.y_only_dataset["y_train"]
        n_steps = 30
        model = ARNet().fit(y_train)

        # In sample
        self.assertIsInstance(model.fitted_values_, pd.Series)
        self.assertIsInstance(model.fitted_values_all_, pd.DataFrame)
        self.assertTrue(model.fitted_values_.index.equals(y_train.index[-model.fitted_values_.size:]))
        self.assertEqual(model.fitted_values_.size, y_train.size - model.p)
        self.assertEqual(model.fitted_values_all_.shape, (y_train.size - model.p, model.repeats))

        model_np = ARNet().fit(y_train.to_numpy())
        self.assertIsInstance(model_np.fitted_values_, np.ndarray)
        self.assertIsInstance(model_np.fitted_values_all_, np.ndarray)
        self.assertEqual(model_np.fitted_values_.size, y_train.size - model_np.p)
        self.assertEqual(model_np.fitted_values_all_.shape, (y_train.size - model_np.p, model_np.repeats))

        # Out sample
        fores = model.predict(n_steps=n_steps)
        self.assertTrue(fores.index.equals(pd.DatetimeIndex(
            ['1977-01-01', '1978-01-01', '1979-01-01', '1980-01-01',
             '1981-01-01', '1982-01-01', '1983-01-01', '1984-01-01',
             '1985-01-01', '1986-01-01', '1987-01-01', '1988-01-01',
             '1989-01-01', '1990-01-01', '1991-01-01', '1992-01-01',
             '1993-01-01', '1994-01-01', '1995-01-01', '1996-01-01',
             '1997-01-01', '1998-01-01', '1999-01-01', '2000-01-01',
             '2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01',
             '2005-01-01', '2006-01-01'],  dtype='datetime64[ns]', freq='YS-JAN')))

        fores = ARNet().fit(y_train.set_axis(range(y_train.size))).predict(n_steps=n_steps)
        self.assertTrue(fores.index.equals(pd.RangeIndex(start=y_train.size, stop=y_train.size + n_steps)))

        with self.assertWarnsRegex(UserWarning, "Couldn't infer a frequency for the series"):
            fores = ARNet().fit(y_train.set_axis(reversed(range(y_train.size)))).predict(n_steps=n_steps)
        self.assertTrue(fores.index.equals(pd.RangeIndex(n_steps)))
