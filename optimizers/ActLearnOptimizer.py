from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import active_learning.src.bl as bl
from optimizers.base_optimizer import BaseOptimizer


class ActLearnOptimizer(BaseOptimizer):
    """
    Pure BL Active Learning optimizer.
    Bypasses ModelWrapper entirely—no encoding, no lookup, no NN.
    Uses BL's own ydist as the objective score.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
        self._bl_data = None


    def _load_bl_data(self):
        """Load BL-compatible CSV file from model_config."""
        csv_path = self.model_config.get_dataset_file()
        data = bl.Data(bl.csv(csv_path))

        if len(data.cols.y) == 0:
            raise ValueError("BL could not detect Y columns in dataset CSV.")

        return data


    def optimize(self):
        n_trials = self.config["n_trials"]
        print(f"=== Running BL Active Learning (budget={n_trials}) ===")

        # 1. Load BL dataset (raw rows)
        data = self._load_bl_data()

        # 2. Set BL stopping condition
        bl.the.Stop = n_trials

        # 3. Run BL ActiveLearner
        result = bl.actLearn(data, shuffle=True)

        # 4. BL gives best rows directly
        best_row = bl.first(result.best.rows)

        # 5. Extract X-components only
        x_len = len(data.cols.x)
        best_hp = dict(zip(data.cols.names[:x_len], best_row[:x_len]))

        # 6. Compute BL's own score = ydist
        best_d2h = bl.ydist(best_row, data)

        # Normalize to SEOptBench scoring convention:
        #   model_wrapper returns "1 - d2h", so do the same
        best_value = best_d2h

        # 7. Log result
        self.logging_util.start_logging()
        self.logging_util.log(best_hp, best_value, 1)
        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value = best_value

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")

        return self.best_config, self.best_value
