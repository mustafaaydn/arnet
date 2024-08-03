import unittest

import pandas as pd
from src.arnet import ARNet


class TestPI(unittest.TestCase):
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

    def test_prediction_intervals_return_single_alpha(self):
        y_train = self.y_only_dataset["y_train"]
        alpha = 8
        n_steps = 5
        rv = ARNet().fit(y_train).predict(n_steps=n_steps, alphas=alpha, return_intervals=True, n_paths=10)
        self.assertEqual(len(rv), 2)
        self.assertIsInstance(rv[1], pd.DataFrame)
        self.assertEqual(rv[1].shape, (n_steps, 2))
        self.assertEqual(rv[1].columns.nlevels, 1)
        self.assertEqual(rv[1].columns.size, 2)
        self.assertEqual(rv[1].columns.tolist(), ["lower", "upper"])
        self.assertEqual(rv[1].columns.name, f"{alpha}%")

    def test_prediction_intervals_return_multiple_alphas(self):
        y_train = self.y_only_dataset["y_train"]
        alphas = 95, 80
        n_steps = 5
        rv = ARNet().fit(y_train).predict(n_steps=n_steps, alphas=alphas, return_intervals=True, n_paths=10)
        self.assertEqual(len(rv), 2)
        self.assertIsInstance(rv[1], pd.DataFrame)
        self.assertEqual(rv[1].shape, (n_steps, 2*len(alphas)))
        self.assertEqual(rv[1].columns.nlevels, 2)
        self.assertEqual(rv[1].columns.size, 2*len(alphas))
        self.assertEqual(rv[1].columns.tolist(),
                         [(alpha, side) for alpha in alphas for side in ("lower", "upper")])

    def test_prediction_intervals_return_with_paths(self):
        y_train = self.y_only_dataset["y_train"]
        alpha = 80
        n_steps, n_paths = 5, 10
        rv = (ARNet().fit(y_train).predict(n_steps=n_steps, alphas=alpha,
                                           return_intervals=True, return_paths=True, n_paths=n_paths))
        self.assertEqual(len(rv), 3)
        self.assertIsInstance(rv[2], pd.DataFrame)
        self.assertEqual(rv[2].shape, (n_steps, n_paths))
        self.assertEqual(rv[2].columns.tolist(), [*range(n_paths)])
