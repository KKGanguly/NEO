import time
import uuid
import random
from dehb import DEHB
from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.configuration import Configuration
from ConfigSpace import ConfigurationSpace
import numpy as np
from models.Data import Data
import copy

# -------------------------------------------------------------------
# NEW: Custom configuration space that samples only predefined configs
# -------------------------------------------------------------------
class CustomConfigurationSpace(ConfigurationSpace):
    def __init__(self, predefined_configs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #safe copy to ensure no side effects
        self.predefined_configs = copy.deepcopy(predefined_configs)

    def sample_configuration(self, size=1):
        #to ensure immutability
        frozen = tuple(self.predefined_configs)
        if size == 1:
            cfg = random.choice(self.predefined_configs)
            return Configuration(self, values=cfg)
        else:
            return [Configuration(self, values=c)
                    for c in random.sample(self.predefined_configs, size)]


# -------------------------------------------------------------------
# DEHB Optimizer (your original, minimally modified)
# -------------------------------------------------------------------
class DEHBOptimizer(BaseOptimizer):

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # Data class from your model for NN snapping
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

        # ORIGINAL configspace (we replace it below)
        self.orig_config_space, self.param_names, self.space = self.model_config.get_configspace()
        
        self.cache = {}
        self.best_config = None
        self.best_value = None

    # -------------------------------------------------------------------
    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)): return int(v)
        if isinstance(v, (np.floating, np.float64)): return float(v)
        return v

    def _config_to_dict(self, config):
        return {p: self._clean(config[p]) for p in self.param_names}

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        nn_row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, nn_row)}

    def _row_tuple(self, d):
        return tuple(d[p] for p in self.param_names)

    def create_configspace(self):
        combined_space = list(zip(*self.model_config.get_hyperparam_dict().values()))
        config_dict = [dict(zip(self.param_names, values)) for values in combined_space]
        cs = CustomConfigurationSpace(config_dict)
        for hyperparameter in self.orig_config_space.get_hyperparameters():
            cs.add_hyperparameter(hyperparameter)       
        return cs

    # -------------------------------------------------------------------
    # Validation wrapper
    # -------------------------------------------------------------------
    def make_validated_config(self, hp_dict):
        c = Configuration(self.config_space, hp_dict, allow_inactive_with_values=False)
        self.config_space.check_configuration(c)
        return c

    # -------------------------------------------------------------------
    def optimize(self):

        if not self.logging_util:
            raise ValueError("Logging util not set.")

        n_trials = self.config["n_trials"]
        self.logging_util.start_logging()

        # -------------------------------------------------------------------
        # Objective wrapper (unchanged)
        # -------------------------------------------------------------------
        def objective(config, fidelity, **kwargs):
            raw_hp = copy.deepcopy(self._config_to_dict(config))
            snapped_hp = self._nearest_row(raw_hp)

            key = self._row_tuple(snapped_hp)
            if key in self.cache:
                return {"fitness": self.cache[key], "cost": 1}

            score = self.model_wrapper.run_model(snapped_hp)
            fitness = 1 - score

            self.logging_util.log(snapped_hp, fitness, 1)
            self.cache[key] = fitness

            return {"fitness": fitness, "cost": 1}

        print("Starting DEHB optimization")
        #discretize tabular space so that initial selection is within search space
        self.config_space = self.create_configspace()

        output_directory = (
            f"{self.config['output_directory']}/"
            f"dehb_run_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )

        random.seed(self.seed)

        # -------------------------------------------------------------------
        # Use our *discrete config space* here
        # -------------------------------------------------------------------
        dehb = DEHB(
            f=objective,
            cs=self.config_space,
            min_fidelity=1,
            max_fidelity=10,
            n_workers=1,
            seed=self.seed,
            output_path=output_directory
        )
        dehb.vectorized = False

        # -------------------------------------------------------------------
        # MAIN LOOP (unchanged except configspace behavior)
        # -------------------------------------------------------------------
        for _ in range(n_trials):

            job = dehb.ask()
            raw_cfg = job["config"]

            raw_hp = self._config_to_dict(raw_cfg)
            snapped_hp = self._nearest_row(raw_hp)

            cfg_obj = self.make_validated_config(snapped_hp)
            cfg_obj = Configuration(self.config_space,
                                   copy.deepcopy(cfg_obj.get_dictionary()))

            result = objective(cfg_obj, fidelity=job["fidelity"])
            dehb.tell(job, result)

        # -------------------------------------------------------------------
        # FINAL RESULT (same as before)
        # -------------------------------------------------------------------
        inc = dehb.vector_to_configspace(dehb.inc_config)
        raw_inc = self._config_to_dict(inc)
        final_hp = self._nearest_row(raw_inc)

        final_score = 1 - self.model_wrapper.run_model(final_hp)

        self.best_config = final_hp
        self.best_value = final_score

        print(f"Final best config = {self.best_config}")
        print(f"Final best score = {self.best_value}")

        self.logging_util.stop_logging()

        return self.best_config, self.best_value
