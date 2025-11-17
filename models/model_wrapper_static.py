import pandas as pd
from utils import DistanceUtil


class ModelWrapperStatic:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X

        # normalize y once
        self.y = (y - y.min()) / (y.max() - y.min())

        # ⚡ Precompute dictionary for O(1) hyperparameter lookup
        # Key = tuple of hyperparameter values in column order
        # Val = row index
        self.lookup = {
            tuple(float(v) for v in row.values): idx
            for idx, row in self.X.iterrows()
        }

    def _find_row_fast(self, hyperparams):
        if not hyperparams:
            raise ValueError("No hyperparameters provided.")

        # Build key respecting X column order (fast!)
        key = tuple(float(hyperparams[col]) for col in self.X.columns)

        try:
            return self.lookup[key]
        except KeyError:
            return None

    def _score_tuple(self, idx):
        row = self.y.loc[idx]
        return tuple((1 - v) if col.endswith("-") else v
                     for col, v in row.items())

    def _avg_d2h(self, scores):
        d2h = DistanceUtil.d2h([1] * len(scores), scores)
        return scores, d2h

    def get_score(self, hyperparams):
        idx = self._find_row_fast(hyperparams)
        print("found idx", idx)
        return self._score_tuple(idx)

    def run_model(self, hyperparams=None, budget=None):
        scores = self.get_score(hyperparams)
        _, d2h = self._avg_d2h(scores)
        return 1 - d2h

    def evaluate(self, hyperparameters=None):
        scores = self.get_score(hyperparameters)
        return self._avg_d2h(scores)

    def test(self, hyperparameters=None):
        return self.evaluate(hyperparameters)
