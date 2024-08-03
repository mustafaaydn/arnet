import unittest
from operator import itemgetter

import pandas as pd
from sklearn.neural_network import MLPRegressor
from src.arnet import ARNet


class TestRepr(unittest.TestCase):
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

    def test_repr(self):
        model = ARNet()
        self.assertEqual(repr(model), "ARNet(base_model=MLPRegressor(hidden_layer_sizes=None, max_iter=2000))")

        model = ARNet(MLPRegressor(hidden_layer_sizes=44), p=2, P=3, seasonality=7)
        self.assertEqual(repr(model),
                         "ARNet(base_model=MLPRegressor(hidden_layer_sizes=44), p=2, P=3, seasonality=7)")

        model = ARNet(MLPRegressor(solver="sgd"), p=2)
        self.assertEqual(repr(model), "ARNet(base_model=MLPRegressor(solver='sgd'), p=2)")

        y = pd.concat(itemgetter("y_train", "y_test")(self.y_only_dataset))
        model = ARNet().fit(y)
        self.assertEqual(repr(model), "ARNet(base_model=MLPRegressor(hidden_layer_sizes=5, max_iter=2000), p=9)")
