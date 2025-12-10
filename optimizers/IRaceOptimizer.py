import time
import uuid
import numpy as np
import pandas as pd
from irace import irace

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)


class IraceOptimizer(BaseOptimizer):
    """
    irace optimizer integrated into the discrete-table hyperparameter framework.

    ✔ Uses discrete ConfigSpace
    ✔ Converts irace proposed config → nearest row via KD-tree
    ✔ Evaluates model_wrapper.run_model(valid_hp)
    ✔ Maintains internal cache
    ✔ Logs using logging_util
    ✔ Returns best_config, best_value
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        # Encoded dataset (design matrix)
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # KD-tree nearest row finder
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        # ConfigSpace
        self.config_space, self.param_names, _ = self.model_config.get_configspace()

        # Cache for evaluated configs
        self.cache = {}

        # Best solution tracking
        self.best_config = None
        self.best_value = float("inf")

    # -------------------------- Helpers -------------------------- #

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        """Project a configuration onto the nearest valid encoded table row."""
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    # ---------------------- Parameter Table ---------------------- #

    def _generate_parameters_table(self):
        """
        Convert ConfigSpace into an irace parameter table formatted string.

        irace format:
        <param>  ""   c   ("v1","v2","v3")
        """
        lines = [""]

        for hp in self.config_space.get_hyperparameters():

            if isinstance(hp, Constant):
                # irace constants can be modeled as ("value")
                choices = [hp.value]

            elif isinstance(hp, OrdinalHyperparameter):
                choices = list(hp.sequence)

            elif isinstance(hp, CategoricalHyperparameter):
                choices = list(hp.choices)

            else:
                raise ValueError("Unsupported hyperparameter type for irace")

            rng = '("' + '", "'.join(map(str, choices)) + '")'
            line = f"{hp.name}    \"\"    c    {rng}"
            lines.append(line)

        return "\n".join(lines)

    # ------------------------- Target Runner ------------------------ #

    def _make_runner(self):
        """
        Build the irace-compatible target_runner() function.
        """

        def target_runner(experiment, scenario):
            # Convert experiment config dict → usable Python dict
            raw_hp = experiment["configuration"]

            # Convert strings → native numeric types where necessary
            cleaned_hp = {}
            for hp in self.config_space.get_hyperparameters():

                name = hp.name
                val = raw_hp[name]

                if isinstance(hp, Constant):
                    cleaned_hp[name] = hp.value

                elif isinstance(hp, OrdinalHyperparameter):
                    cleaned_hp[name] = val  # categorical string, already encoded

                elif isinstance(hp, CategoricalHyperparameter):
                    cleaned_hp[name] = val

            # Project into nearest encoded row
            valid_hp = self._nearest_row(cleaned_hp)
            key = self._row_tuple(valid_hp)

            if key in self.cache:
                fitness = self.cache[key]
            else:
                score = self.model_wrapper.run_model(valid_hp)
                fitness = 1 - score
                self.cache[key] = fitness

            # track best
            if fitness < self.best_value:
                self.best_value = fitness
                self.best_config = valid_hp

            # log
            self.logging_util.log(valid_hp, fitness, 1)
            return dict(cost=fitness)

        return target_runner

    # --------------------------- Optimize --------------------------- #

    def optimize(self):
        if not self.logging_util:
            raise ValueError("Logging util not provided.")

        n_trials = self.config["n_trials"]
        self.logging_util.start_logging()

        # irace uses "maxExperiments" as its evaluation budget
        scenario = dict(
            instances=[0],
            maxExperiments=n_trials,
            debugLevel=0,
            parallel=1,
            logFile=""
        )

        parameters_table = self._generate_parameters_table()
        target_runner = self._make_runner()

        try:
            tuner = irace(scenario, parameters_table, target_runner)
            tuner.run()

        except Exception as e:
            print(f"irace terminated early: {e}")

        # Convert best_value back to actual score
        final_score = 1 - self.best_value

        self.logging_util.stop_logging()
        return self.best_config, final_score
