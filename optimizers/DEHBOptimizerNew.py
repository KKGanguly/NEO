import time
import uuid
import random
from dehb import DEHB
from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.configuration import Configuration
from ConfigSpace import ConfigurationSpace
import numpy as np
from models.Data import Data

class DEHBOptimizer(BaseOptimizer):
    """
    ✔ Updated to follow SMACOptimizer logic
    ✔ Samples configs → nearest row lookup → model evaluation
    ✔ Uses caching
    ✔ Logs every evaluation
    ✔ Returns best_config, best_value
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        # Tabular encoded dataset (same as SMAC)
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.cache = {}

        # KD-tree–style nearest neighbor
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

        # Configspace with correct hyperparameters
        self.config_space, _, _ = self.model_config.get_configspace()
        self.best_config = None
        self.best_value = None

    # -------------------- Helpers --------------------

    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)): return int(v)
        if isinstance(v, (np.floating, np.float64)): return float(v)
        return v

    def _config_to_dict(self, config: Configuration):
        return {p: self._clean(config[p]) for p in self.model_config.param_names}

    def _row_tuple(self, d):
        return tuple(d[p] for p in self.model_config.param_names)

    def _nearest_row(self, hp_dict):
        query = [hp_dict[col] for col in self.columns]
        nn_row = self.nn.nearestRow(query)
        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}

    # -------------------- Optimization --------------------

    def optimize(self):

        if not self.logging_util:
            raise ValueError("Logging util not set")

        n_trials = self.config["n_trials"]
        self.logging_util.start_logging()

        # ------------------------------------------------#
        # Objective identical to SMAC
        # ------------------------------------------------#
        def objective(config: Configuration, fidelity: float, **kwargs):

            raw_hp = self._config_to_dict(config)
            valid_hp = self._nearest_row(raw_hp)

            key = self._row_tuple(valid_hp)
            if key in self.cache:
                return {"fitness": self.cache[key], "cost": 1}

            score = self.model_wrapper.run_model(valid_hp)
            fitness = 1 - score

            # Log the evaluation
            self.logging_util.log(valid_hp, fitness, 1)

            self.cache[key] = fitness
            return {"fitness": fitness, "cost": 1}

        # ------------------------------------------------#
        print("Starting DEHB optimization")
        output_directory = (
            f"{self.config['output_directory']}/"
            f"dehb_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )

        random.seed(self.seed)

        # ------------------------------------------------#
        # Create DEHB instance
        # ------------------------------------------------#
        dehb = DEHB(
            f=objective,
            cs=self.config_space,
            min_fidelity=1,
            max_fidelity=10,        # <-- Tabular dataset has no fidelity
            n_workers=1,
            seed=self.seed,
            output_path=output_directory
        )

        # ------------------------------------------------#
        # Main optimization loop
        # ------------------------------------------------#
        for _ in range(n_trials):
            job = dehb.ask()
            config = job["config"]
            result = objective(config, fidelity=1)
            dehb.tell(job, result)

        # ------------------------------------------------#
        # Retrieve best result via DEHB internals
        # ------------------------------------------------#
        inc_config = dehb.vector_to_configspace(dehb.inc_config)
        final_raw_hp = self._config_to_dict(inc_config)
        final_valid_hp = self._nearest_row(final_raw_hp)

        final_score = 1 - self.model_wrapper.run_model(final_valid_hp)

        self.best_config = final_valid_hp
        self.best_value = final_score

        print(f"Best config: {self.best_config}, score = {self.best_value}")

        self.logging_util.stop_logging()

        return self.best_config, self.best_value
