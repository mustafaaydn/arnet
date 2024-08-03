import unittest

import pandas as pd
from src.arnet import ARNet


class TestFitPredict(unittest.TestCase):
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

    # Fit-predict flow altogether
    def test_fit_predict_y_only(self):
        try:
            model = ARNet()
            model.fit(self.y_only_dataset["y_train"])
            model.predict(n_steps=len(self.y_only_dataset["y_test"]))
        except Exception as e:
            self.fail(f"Couldn't fit-predict with a y-only data: {e}")

    def test_fit_predict_y_and_X(self):
        try:
            model = ARNet()
            model.fit(self.y_and_X_dataset["y_train"], self.y_and_X_dataset["X_train"])
            model.predict(self.y_and_X_dataset["X_test"], n_steps=len(self.y_and_X_dataset["y_test"]))
        except Exception as e:
            self.fail(f"Couldn't fit-predict with an y-X data: {e}")

    def test_fit_predict_X_and_y(self):
        try:
            model = ARNet()
            model.fit(self.y_and_X_dataset["X_train"], self.y_and_X_dataset["y_train"])
            model.predict(self.y_and_X_dataset["X_test"], n_steps=len(self.y_and_X_dataset["y_test"]))
        except Exception as e:
            self.fail(f"Couldn't fit-predict with an X-y data: {e}")
