import tempfile
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import active_learning.src.bl as bl
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data


class ActLearnOptimizer(BaseOptimizer):

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        # Encoded table for nearest-row matching
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # Build KD-tree on encoded data
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

        # Pre-build BL CSV dataframe
        self._bl_csv_path = self._make_bl_csv()


    # -------------------------------------------------------
    # Helper: convert BL row → dict of encoded hyperparameters
    # -------------------------------------------------------
    def _clean(self, v):
        import numpy as np
        if isinstance(v, (np.integer, np.int64)): return int(v)
        if isinstance(v, (np.floating, np.float64)): return float(v)
        return v

    def _row_to_dict(self, row):
        return {col: self._clean(v) for col, v in zip(self.columns, row)}


    # -------------------------------------------------------
    # Build BL-compatible CSV (X cols get X; Y cols keep +/-)
    # -------------------------------------------------------
    def _make_bl_csv(self):
        X = self.model_wrapper.X.copy()
        Y = self.model_wrapper.y.copy()

        # Check Y suffixes correctness
        for col in Y.columns:
            if not (col.endswith("+") or col.endswith("-")):
                raise ValueError(
                    f"Y column '{col}' must end with + or - for BL objective direction."
                )

        df = pd.concat([X, Y], axis=1)

        new_cols = []
        for col in df.columns:
            if col in Y.columns:
                new_cols.append(col)   # keep + or -
            else:
                # X column — add X suffix if missing
                new_cols.append(col if col.endswith("X") else col + "X")

        df.columns = new_cols

        # Write to temporary file
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix="_bl.csv",
            mode="w"
        )
        df.to_csv(tmp.name, index=False)
        return tmp.name


    # -------------------------------------------------------
    # OPTIMIZE via BL.actLearn
    # -------------------------------------------------------
    def optimize(self):
        n_trials = self.config["n_trials"]
        print(f"=== Running ACTLEARN optimizer (budget={n_trials}) ===")

        # Load BL dataset from temporary CSV
        bl_data = bl.Data(bl.csv(self._bl_csv_path))

        if len(bl_data.cols.y) == 0:
            raise ValueError("BL did not detect any Y objective columns.")

        bl.the.Stop = n_trials

        # Run BL Active Learning
        learner = bl.actLearn(bl_data, shuffle=True)

        best_row_raw = bl.first(learner.best.rows)

        # Strip BL Y columns before KD-tree mapping
        clean_row = best_row_raw[:len(self.columns)]

        encoded_row = self.nn.nearestRow(clean_row)
        hp = self._row_to_dict(encoded_row)

        # Evaluate with the TRUE model
        self.logging_util.start_logging()
        score = self.model_wrapper.run_model(hp)
        fitness = 1 - score
        self.logging_util.log(hp, fitness, 1)
        self.logging_util.stop_logging()

        self.best_config = hp
        self.best_value = fitness

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")

        return self.best_config, self.best_value
