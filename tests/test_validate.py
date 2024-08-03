import unittest
from operator import itemgetter

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.arnet import ARNet


class TestValidate(unittest.TestCase):
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

    def test_validate_one_config_yes_refit(self):
        y_train = self.y_only_dataset["y_train"]
        p, max_iter = 3, 5_000
        search = ARNet.validate(y_train,
                                param_grid={"p": [p], "estimator__max_iter": [max_iter]},
                                cv=TimeSeriesSplit(3, test_size=10))
        self.assertIsInstance(search, dict)
        self.assertEqual(sorted(search.keys()), ["best_estimator_", "best_params_", "scores_"])
        best_params = search["best_params_"]
        self.assertIsInstance(best_params, dict)
        self.assertEqual(sorted(best_params.keys()), ["estimator__max_iter", "p"])
        self.assertEqual(best_params["p"], p)
        self.assertEqual(best_params["estimator__max_iter"], max_iter)

        scores = search["scores_"]
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 1)
        self.assertEqual(sorted(next(iter(scores.keys()))), [("estimator__max_iter", max_iter), ("p", p)])

        best_estimator = search["best_estimator_"]
        self.assertIsInstance(best_estimator, ARNet)
        self.assertTrue(hasattr(best_estimator, "fitted_values_"))
        self.assertIsInstance(best_estimator.predict(n_steps=1), pd.Series)

    def test_validate_one_config_no_refit(self):
        y_train = self.y_only_dataset["y_train"]
        p, solver, hls = 2, "sgd", [24, 18]
        search = ARNet.validate(y_train,
                                param_grid={"p": [p],
                                            "estimator__solver": [solver],
                                            "estimator__hidden_layer_sizes": [hls]},
                                cv=TimeSeriesSplit(4, test_size=8),
                                refit=False)
        self.assertIsInstance(search, dict)
        self.assertEqual(sorted(search.keys()), ["best_params_", "scores_"])
        best_params = search["best_params_"]
        self.assertIsInstance(best_params, dict)
        self.assertEqual(sorted(best_params.keys()), ["estimator__hidden_layer_sizes", "estimator__solver", "p"])
        self.assertEqual(best_params["p"], p)
        self.assertEqual(best_params["estimator__solver"], solver)
        self.assertEqual(best_params["estimator__hidden_layer_sizes"], tuple(hls))

        scores = search["scores_"]
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 1)
        self.assertEqual(sorted(next(iter(scores.keys()))), [("estimator__hidden_layer_sizes", tuple(hls)),
                                                             ("estimator__solver", solver), ("p", p)])

        def test_validate_one_config_yes_refit_with_X(self):
            X_train, y_train = self.y_and_X_dataset["X_train"], self.y_and_X_dataset["y_train"]
            p, max_iter = 3, 5_000
            search = ARNet.validate(y_train,
                                    X=X_train,
                                    param_grid={"p": [p], "estimator__max_iter": [max_iter]},
                                    cv=TimeSeriesSplit(3, test_size=10))
            self.assertIsInstance(search, dict)
            self.assertEqual(sorted(search.keys()), ["best_estimator_", "best_params_", "scores_"])
            best_params = search["best_params_"]
            self.assertIsInstance(best_params, dict)
            self.assertEqual(sorted(best_params.keys()), ["estimator__max_iter", "p"])
            self.assertEqual(best_params["p"], p)
            self.assertEqual(best_params["estimator__max_iter"], max_iter)

            scores = search["scores_"]
            self.assertIsInstance(scores, dict)
            self.assertEqual(len(scores), 1)
            self.assertEqual(sorted(next(iter(scores.keys()))), [("estimator__max_iter", max_iter), ("p", p)])

            best_estimator = search["best_estimator_"]
            self.assertIsInstance(best_estimator, ARNet)
            self.assertTrue(hasattr(best_estimator, "fitted_values_"))
            self.assertIsInstance(best_estimator.predict(n_steps=1), pd.Series)

    def test_validate_randomized_search(self):
        X_train, y_train, X_test = itemgetter("X_train", "y_train", "X_test")(self.y_and_X_dataset)
        p, max_iter = 3, 5_000
        search = ARNet.validate(y_train,
                                X=X_train,
                                param_grid={"p": [p], "estimator__max_iter": [max_iter]},
                                cv=TimeSeriesSplit(3, test_size=10),
                                n_iter=1)
        self.assertIsInstance(search, dict)
        self.assertEqual(sorted(search.keys()), ["best_estimator_", "best_params_", "scores_"])
        best_params = search["best_params_"]
        self.assertIsInstance(best_params, dict)
        self.assertEqual(sorted(best_params.keys()), ["estimator__max_iter", "p"])
        self.assertEqual(best_params["p"], p)
        self.assertEqual(best_params["estimator__max_iter"], max_iter)

        scores = search["scores_"]
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 1)
        self.assertEqual(sorted(next(iter(scores.keys()))), [("estimator__max_iter", max_iter), ("p", p)])

        best_estimator = search["best_estimator_"]
        self.assertIsInstance(best_estimator, ARNet)
        self.assertTrue(hasattr(best_estimator, "fitted_values_"))
        self.assertIsInstance(best_estimator.predict(X_test), pd.Series)

    def test_validate_randomized_bad_n_iter(self):
        y_train = self.y_only_dataset["y_train"]
        grid = {"p": [2], "estimator__max_iter": [50]}
        with self.assertRaisesRegex(ValueError, r"`n_iter` needs to be an integer \(or None\)"):
            ARNet.validate(y_train, grid, n_iter=7.84)

        with self.assertRaisesRegex(ValueError, r"If integer, `n_iter` needs to be either -1 \(grid search\) "
                                                "or positive integer \(randomized search\)"):
            ARNet.validate(y_train, grid, n_iter=0)

        with self.assertRaisesRegex(ValueError, r"If integer, `n_iter` needs to be either -1 \(grid search\) "
                                                "or positive integer \(randomized search\)"):
            ARNet.validate(y_train, grid, n_iter=-12)
