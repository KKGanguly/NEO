import numpy as np
import time
import uuid
from sklearn.tree import DecisionTreeRegressor
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant
)
from collections import OrderedDict


class FLASHOptimizer(BaseOptimizer):
    """
    FLASH rewritten to match the internal design of TPEOptimizer + SMACOptimizer:

    ✔ Uses tabular encoded dataset from model_wrapper.X
    ✔ Samples real configurations based on discrete ConfigSpace
    ✔ Maps each configuration → nearest row via Data KD-tree
    ✔ Evaluates using model_wrapper.run_model()
    ✔ Tracks cached evaluations
    ✔ Performs FLASH surrogate-model search
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        # Tabular encoded dataset
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.y = self.model_wrapper.y  # optional: depends on your wrapper
        self.cache = {}

        # KD-Tree for nearest row lookup
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

        # Extract ConfigSpace
        self.config_space, self.param_names, _ = self.model_config.get_configspace()

        self.best_config = None
        self.best_value = None

    # ------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------ #
    def _clean(self, v):
        """Convert numpy types to Python primitives."""
        if hasattr(v, "item"):
            return v.item()
        return v

    def _nearest_row(self, hp_dict):
        """Map hyperparameter dict → nearest valid table row."""
        query = [hp_dict[col] for col in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _sample_random_config(self):
        """Sample from ConfigSpace (FLASH needs random initialization)."""
        sample = {}
        for hp in self.config_space.get_hyperparameters():

            if isinstance(hp, Constant):
                sample[hp.name] = hp.value

            elif isinstance(hp, OrdinalHyperparameter):
                vals = list(hp.sequence)
                sample[hp.name] = np.random.choice(vals)

            elif isinstance(hp, CategoricalHyperparameter):
                vals = list(hp.choices)
                sample[hp.name] = np.random.choice(vals)

            else:
                raise ValueError(f"Unsupported HP type {type(hp)}")

        return sample

    # ------------------------------------------------ #
    # FLASH objective wrapper
    # ------------------------------------------------ #
    def _evaluate(self, hp_dict):
        """Evaluate configuration after nearest-row projection."""
        valid_hp = self._nearest_row(hp_dict)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            return self.cache[key]

        score = self.model_wrapper.run_model(valid_hp)
        fitness = 1 - score

        self.cache[key] = fitness
        self.logging_util.log(valid_hp, fitness, 1)

        return fitness

    # ------------------------------------------------ #
    # Main Optimization Loop
    # ------------------------------------------------ #
    def optimize(self):

        if not self.logging_util:
            raise ValueError("Logging util not provided.")

        self.logging_util.start_logging()
        n_trials = self.config["n_trials"]

        # FLASH defaults (using 4, init samples did as good as 4)
        init_samples = 4  
        surrogate = DecisionTreeRegressor()

        # ----------------------------------------------- #
        # Initial population
        # ----------------------------------------------- #
        X = []
        y = []

        for _ in range(init_samples):
            cfg = self._sample_random_config()
            X.append([cfg[p] for p in self.columns])
            y.append(self._evaluate(cfg))

        X = np.array(X)
        y = np.array(y)

        # ----------------------------------------------- #
        # Main FLASH loop
        # ----------------------------------------------- #
        for t in range(n_trials - init_samples):

            # Train surrogate
            surrogate.fit(X, y)

            # Generate candidate pool (1000 samples)
            candidates = []
            for _ in range(1000):
                cfg = self._sample_random_config()
                candidates.append([cfg[p] for p in self.columns])
            candidates = np.array(candidates)

            # Predict fitness
            preds = surrogate.predict(candidates)
            best_idx = np.argmin(preds)
            best_candidate_vec = candidates[best_idx]

            # Convert back to dict
            candidate_dict = {col: best_candidate_vec[i] for i, col in enumerate(self.columns)}

            # Evaluate
            fitness = self._evaluate(candidate_dict)

            # Add to training set
            X = np.vstack([X, best_candidate_vec.reshape(1, -1)])
            y = np.append(y, fitness)

        # ------------------------------------------------ #
        # Final best config
        # ------------------------------------------------ #
        best_idx = np.argmin(y)
        best_vec = X[best_idx]
        final_hp = {c: best_vec[i] for i, c in enumerate(self.columns)}
        final_score = 1 - y[best_idx]

        self.best_config = final_hp
        self.best_value = final_score

        self.logging_util.stop_logging()
        return self.best_config, self.best_value
