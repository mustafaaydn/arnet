import unittest
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from src.arnet import ARNet


class TestAR(unittest.TestCase):
    def setUp(self):
        sunspots = (pd.read_csv("tests/datasets/sunspots.csv")
                      .pipe(lambda fr: fr.set_axis(pd.PeriodIndex(fr.pop("Year"), freq="Y").to_timestamp()))
                      .squeeze())  # https://rdrr.io/r/datasets/sunspot.year.html
        self.y_only_dataset = {"y_train": sunspots.iloc[:-12], "y_test": sunspots.iloc[-12:]}

    def test_ar_order_selection(self):
        with self.assertRaisesRegex(ValueError, "Series should have at least one observation"):
            ARNet._select_ar_order([])
        with self.assertWarnsRegex(UserWarning, "Series has only 1 observation"):
            order = ARNet._select_ar_order([5])
        self.assertEqual(order, 0)

        y = pd.concat(itemgetter("y_train", "y_test")(self.y_only_dataset))
        self.assertEqual(ARNet._select_ar_order(y), 9)

    def test_linreg_same_as_ar(self):
        y = pd.concat(itemgetter("y_train", "y_test")(self.y_only_dataset))
        model = ARNet(LinearRegression(), repeats=1)
        linreg_fores = model.fit(y).predict(n_steps=20)
        ar_fores = AutoReg(y, lags=model.p).fit().forecast(steps=linreg_fores.size).set_axis(linreg_fores.index)
        self.assertTrue(np.allclose(linreg_fores, ar_fores), msg="LinearRegression forecasts ~~ AR(p) forecasts")
