import time
import uuid
import numpy as np

from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from ConfigSpace.configuration import Configuration
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario


class SMACOptimizer(BaseOptimizer):

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
        self.config_space = None
        self.X_df = self.model_wrapper.X

        self.cache = {}


    def optimize(self):

        if not self.logging_util:
            raise ValueError("Logging utils not set!!")

        print("Starting SMAC optimization (row-based, cached)")

        total_budget = self.config["n_trials"]
        initial_ratio = self.config.get("initial_ratio", 0.1)  # default: 10%

        initial_design_size = max(1, int(total_budget * initial_ratio))
        bo_trials = max(1, total_budget - initial_design_size)


        random_seed = self.seed

        n_rows = len(self.X_df)
        row_ids = [str(i) for i in range(n_rows)]  # SMAC likes strings

        cs = ConfigurationSpace(seed=random_seed)
        row_hp = CategoricalHyperparameter("row_id", choices=row_ids)
        cs.add_hyperparameter(row_hp)
        self.config_space = cs

        def objective(config: Configuration, seed: int = 0):
            # SMAC gives us a config with a single key: "row_id"
            row_id_str = config["row_id"]
            row_id = int(row_id_str)

            # Cache lookup: if we already evaluated this row, return immediately
            if row_id in self.cache:
                return self.cache[row_id]

            # Get hyperparameters for this row as a dict
            row_series = self.X_df.iloc[row_id]
            hyperparams = row_series.to_dict()

            score = self.model_wrapper.run_model(hyperparams)

            fitness = 1 - score

            self.logging_util.log(hyperparams, fitness, 1)
            self.cache[row_id] = fitness
            return fitness

        # ------------------------------------------------------------------
        # SMAC Scenario
        # ------------------------------------------------------------------
        output_dir = (
            f"{self.config['output_directory']}/"
            f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )

        scenario = Scenario(
            configspace=self.config_space,
            n_trials=bo_trials,      # BO trials (initial handled via initial_design)
            output_directory=output_dir,
            deterministic=True,
            seed=random_seed,
        )

        # initial = exact percentage of total budget (rows)
        initial_design = HPOFacade.get_initial_design(
            scenario,
            n_configs=initial_design_size
        )

        smac = HPOFacade(
            scenario=scenario,
            target_function=objective,
            initial_design=initial_design
        )

        self.logging_util.start_logging()

        print(f"SMAC optimization started (rows={len(self.X_df)})...")
        print(f"[Initial={initial_design_size}] + [BO={bo_trials}] = {total_budget} total labels")

        try:
            incumbent = smac.optimize()
        except Exception as e:
            print(f"SMAC stopped early: {e}")
            incumbent = smac.optimizer.intensifier.get_incumbent()

        # ------------------------------------------------------------------
        # Best solution
        # ------------------------------------------------------------------
        best_row_id = int(incumbent["row_id"])
        best_row_series = self.X_df.iloc[best_row_id]
        best_hyperparams = best_row_series.to_dict()

        self.best_config = best_hyperparams
        # get fitness from cache (or recompute safely)
        if best_row_id in self.cache:
            best_fitness = self.cache[best_row_id]
        else:
            score = self.model_wrapper.run_model(best_hyperparams)
            best_fitness = 1 - score

        self.best_value = best_fitness

        print(f"Best row id: {best_row_id}")
        print(f"Best config: {self.best_config}")
        print(f"Best value (fitness): {self.best_value}")

        self.logging_util.stop_logging()
