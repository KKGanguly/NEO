import pandas as pd
from utils import DistanceUtil


class ModelWrapperStatic:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X

        # Normalize y once
        self.y = (y - y.min()) / (y.max() - y.min())

        # -----------------------------------------------------
        # Detect categorical vs numerical (same rule as ConfigSpace)
        # -----------------------------------------------------
        self.categorical_cols = [
            col for col in X.columns
            if not pd.api.types.is_numeric_dtype(X[col])
        ]
        self.numeric_cols = [
            col for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        ]

        # -----------------------------------------------------
        # Build lookup table using SAME ENCODING as ConfigSpace
        # -----------------------------------------------------
        def encode_value(col, v):
            if col in self.numeric_cols:
                return float(v)
            else:
                return str(v)   # categorical: keep raw string

        self.lookup = {
            tuple(encode_value(col, row[col]) for col in self.X.columns): idx
            for idx, row in self.X.iterrows()
        }

    # ----------------------------------------------------------
    # Encode SMAC config using same rules as dataset encoding
    # ----------------------------------------------------------
    def _encode_hp(self, col, v):
        if col in self.numeric_cols:
            return float(v)
        else:
            return str(v)

    # ----------------------------------------------------------
    def _find_row_fast(self, hyperparams):
        if not hyperparams:
            raise ValueError("No hyperparameters provided.")

        key = tuple(self._encode_hp(col, hyperparams[col])
                    for col in self.X.columns)

        return self.lookup.get(key, None)

    # ----------------------------------------------------------
    def _score_tuple(self, idx):
        row = self.y.loc[idx]
        return tuple(
            (1 - v) if col.endswith("-") else v
            for col, v in row.items()
        )

    # ----------------------------------------------------------
    def _avg_d2h(self, scores):
        d2h = DistanceUtil.d2h([1] * len(scores), scores)
        return scores, d2h

    # ----------------------------------------------------------
    def get_score(self, hyperparams):
        idx = self._find_row_fast(hyperparams)
        print("found idx", idx)
        return self._score_tuple(idx)

    # ----------------------------------------------------------
    def run_model(self, hyperparams=None, budget=None):
        scores = self.get_score(hyperparams)
        _, d2h = self._avg_d2h(scores)
        return 1 - d2h

    # ----------------------------------------------------------
    def evaluate(self, hyperparameters=None):
        scores = self.get_score(hyperparameters)
        return self._avg_d2h(scores)

    # ----------------------------------------------------------
    def test(self, hyperparameters=None):
        return self.evaluate(hyperparameters)
