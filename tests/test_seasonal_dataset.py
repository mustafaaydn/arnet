import unittest

import pandas as pd
from src.arnet import ARNet


class TestSeasonal(unittest.TestCase):
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

    def test_seasonal_dataset(self):
        y = (pd.read_csv("tests/datasets/airline_passengers.csv")
               .pipe(lambda fr: fr.set_axis(pd.PeriodIndex(fr.pop("Month"), freq="M").to_timestamp()))
               .squeeze())
        try:
            model = ARNet(seasonality=12)
            model.fit(y)
            fores = model.predict(n_steps=12)
        except Exception as e:
            self.fail(f"(seasonal data) Couldn't fit-predict: {e}")

        self.assertTrue(fores.index.equals(pd.DatetimeIndex(
            ["1961-01-01", "1961-02-01", "1961-03-01", "1961-04-01",
             "1961-05-01", "1961-06-01", "1961-07-01", "1961-08-01",
             "1961-09-01", "1961-10-01", "1961-11-01", "1961-12-01"], freq="MS")))
        self.assertNotEqual(model.P, 0)
        self.assertEqual(repr(model),
                         "ARNet(base_model=MLPRegressor(hidden_layer_sizes=2, max_iter=2000), "
                         "p=1, P=1, seasonality=12)")

        self.assertIsInstance(model.fitted_values_, pd.Series)
        self.assertIsInstance(model.fitted_values_all_, pd.DataFrame)
        self.assertTrue(model.fitted_values_.index.equals(y.index[-model.fitted_values_.size:]))
        self.assertEqual(model.fitted_values_.size, y.size - max(model.p, model.P * model.seasonality))
        self.assertEqual(model.fitted_values_all_.shape,
                         (y.size - max(model.p, model.P * model.seasonality), model.repeats))
