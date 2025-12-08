from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
import numpy as np
import time
from optimizers.base_optimizer import BaseOptimizer
import random
import uuid
from sklearn.model_selection import train_test_split
from models.Data import Data
from ConfigSpace.configuration import Configuration
from smac import Scenario, HyperparameterOptimizationFacade as HPOFacade
from smac.utils.configspace import convert_configurations_to_array


class SMACOptimizer(BaseOptimizer):
    """
    ✔ Correct version for tabular HPO
    ✔ SMAC samples from the configspace only
    ✔ Each sample is mapped to the nearest row in the tabular dataset
    ✔ All evaluations come from the table
    ✔ No train/test split
    ✔ No surrogate prediction on held-out data
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        # Tabular encoded dataset
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.cache = {}

        # KD-tree for nearest neighbor lookup across ALL rows
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

        # SMAC configspace
        self.config_space, _, _ = self.model_config.get_configspace()

        self.best_config = None
        self.best_value = None

    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)): return int(v)
        if isinstance(v, (np.floating, np.float64)): return float(v)
        return v

    def _config_to_dict(self, config: Configuration):
        return {p: self._clean(config[p]) for p in self.model_config.param_names}

    def _row_tuple(self, hyperparams):
        return tuple(hyperparams[p] for p in self.model_config.param_names)

    def _nearest_row(self, hyperparams_dict):
        """Return nearest valid configuration row from the table."""
        query = [hyperparams_dict[col] for col in self.columns]
        nn_row = self.nn.nearestRow(query)

        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}

    def optimize(self):

        if not self.logging_util:
            raise ValueError("Logging util not set")

        n_trials = self.config["n_trials"]
        total_budget = self.config["n_trials"]
        init_design_size = 4 
        self.logging_util.start_logging()

        # ------------------------------------------------#
        # Define SMAC Objective
        # ------------------------------------------------#
        def objective(config: Configuration, seed=0):
            raw_hp = self._config_to_dict(config)
            valid_hp = self._nearest_row(raw_hp)

            key = self._row_tuple(valid_hp)
            if key in self.cache:
                return self.cache[key]

            score = self.model_wrapper.run_model(valid_hp)
            fitness = 1 - score

            self.logging_util.log(valid_hp, fitness, 1)
            self.cache[key] = fitness

            return fitness

        output_directory = (
            f"{self.config['output_directory']}/"
            f"smac_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )

        # Scenario
        scenario = Scenario(
            configspace=self.config_space,
            n_trials=n_trials,
            deterministic=True,
            output_directory=output_directory,
            seed=self.seed,
        )

        # SMAC driver
        initial_design = HPOFacade.get_initial_design(scenario,
            n_configs=init_design_size)

        smac = HPOFacade(
            scenario=scenario,
            target_function=objective,
            initial_design=initial_design,
            overwrite=True,
        )

        try:
            incumbent = smac.optimize()
        except Exception:
            incumbent = smac.optimizer.intensifier.get_incumbent()

        final_dict = self._nearest_row(self._config_to_dict(incumbent))
        final_score = 1 - self.model_wrapper.run_model(final_dict)

        self.best_config = final_dict
        self.best_value = final_score

        self.logging_util.stop_logging()
        return self.best_config, self.best_value
