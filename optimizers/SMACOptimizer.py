import time
import uuid
import numpy as np

from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.configuration import Configuration
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from models.Data import Data   # adjust path


class SMACOptimizer(BaseOptimizer):

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.cache = {}

        # NN structure for fallback
        rows = self.X_df.values.tolist()
        self.data_nn = Data(
            rows
        )

    # ----------------------------------------
    # Utility helpers
    # ----------------------------------------

    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        return v

    def _config_to_dict(self, config: Configuration):
        """Convert SMAC config → Python dict using model_config param order."""
        return {p: self._clean(config[p]) for p in self.model_config.param_names}

    def _row_tuple(self, hyperparams: dict):
        """Hashable row representation."""
        return tuple(hyperparams[p] for p in self.model_config.param_names)

    def _valid_row(self, hyperparams: dict):
        """
        Ensure the hyperparams represent a real row:
        - try direct lookup
        - fall back to nearest neighbor
        """

        key = self._row_tuple(hyperparams)

        # Try exact match via ModelWrapperStatic.lookup
        if key in self.model_wrapper.lookup:
            idx = self.model_wrapper.lookup[key]
            row_series = self.X_df.iloc[idx]
            return {col: self._clean(v) for col, v in row_series.items()}

        # NN fallback (continuous or unseen config)
        q = [hyperparams[col] for col in self.columns]
        nn_row = self.data_nn.nearestRow(q)
        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}

    # ----------------------------------------

    def optimize(self):

        if not self.logging_util:
            raise ValueError("Logging utils not set!!")

        print("Starting SMAC optimization with ModelConfigurationStatic")

        total_budget = self.config["n_trials"]
        init_ratio = self.config.get("initial_ratio", 0.1)
        init_design_size = max(1, int(total_budget * init_ratio))

        random_seed = self.seed

        # Get ConfigSpace from ModelConfigurationStatic
        self.config_space, _, _ = self.model_config.get_configspace()

        # ----------------------------------------
        # SMAC objective
        # ----------------------------------------
        def objective(config: Configuration, seed: int = 0):
            raw_hp = self._config_to_dict(config)
            eval_hp = self._valid_row(raw_hp)

            key = self._row_tuple(eval_hp)

            if key in self.cache:
                return self.cache[key]

            score = self.model_wrapper.run_model(eval_hp)
            fitness = 1 - score  # SMAC minimizes
            self.logging_util.log(eval_hp, fitness, 1)
            self.cache[key] = fitness

            return fitness

        # ----------------------------------------
        # SMAC Scenario
        # ----------------------------------------

        output_dir = (
            f"{self.config['output_directory']}/"
            f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )

        scenario = Scenario(
            configspace=self.config_space,
            n_trials=total_budget,
            deterministic=True,
            output_directory=output_dir,
            seed=random_seed,
        )

        initial_design = HPOFacade.get_initial_design(
            scenario,
            n_configs=init_design_size
        )

        smac = HPOFacade(
            scenario=scenario,
            target_function=objective,
            initial_design=initial_design
        )

        self.logging_util.start_logging()

        try:
            incumbent = smac.optimize()
        except Exception as e:
            print("SMAC failed / early stopped:", e)
            incumbent = smac.optimizer.intensifier.get_incumbent()

        # ----------------------------------------
        # Best Solution
        # ----------------------------------------

        raw_hp = self._config_to_dict(incumbent)
        best_hp = self._valid_row(raw_hp)

        key = self._row_tuple(best_hp)
        best_fitness = self.cache.get(key)

        if best_fitness is None:
            score = self.model_wrapper.run_model(best_hp)
            best_fitness = 1 - score

        self.best_config = best_hp
        self.best_value = best_fitness

        self.logging_util.stop_logging()

        print("Best config:", best_hp)
        print("Best fitness:", best_fitness)
