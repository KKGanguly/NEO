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
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.cache = {}

        # ------------------------------------------------------
        # 1. Train/Test split (STATIC FOR THE WHOLE RUN)
        # ------------------------------------------------------
        self.X_train, self.X_test = train_test_split(
            self.X_df,
            test_size=0.5,
            random_state=self.seed,  # ensures reproducibility
        )

        # ------------------------------------------------------
        # 2. KD-trees (BUILT ONCE)
        # ------------------------------------------------------
        self.nn_train = Data(self.X_train.values.tolist())
        self.nn_test = Data(self.X_test.values.tolist())

        # ------------------------------------------------------
        # 3. ConfigSpace - build from full dataset (delayed if needed)
        # ------------------------------------------------------
        # Build ConfigSpace immediately if we have model_config,
        # otherwise delay until set_model_config() is called
        self.config_space, _, _ = self.model_config.get_configspace()

        # optimize() will ONLY run SMAC
    
   
    
    def set_model_config(self, model_config):
        """Override to initialize ConfigSpace when model_config is set."""
        super().set_model_config(model_config)
        if self.config_space is None:
            self._initialize_configspace()
    
    # ----------------------------------------------------------
    # ConfigSpace builder (fallback if model_config unavailable)
    # ----------------------------------------------------------
    def _build_configspace_from_data(self):
        """
        Build a ConfigSpace from the full dataset as a fallback.
        """
        cs = ConfigurationSpace(seed=self.seed)

        for col in self.columns:
            values = self.X_df[col].tolist()
            uniq = sorted(list(set(values)))

            # Constant hyperparameter
            if len(uniq) == 1:
                cs.add_hyperparameter(Constant(col, uniq[0]))
                continue

            # Numeric
            all_numeric = all(isinstance(v, (int, float, np.number)) for v in uniq)

            if all_numeric:
                lo = float(min(uniq))
                hi = float(max(uniq))

                # Collapse extremely narrow ranges
                if abs(hi - lo) < 1e-12:
                    cs.add_hyperparameter(Constant(col, lo))
                else:
                    hp = UniformFloatHyperparameter(
                        name=col,
                        lower=lo,
                        upper=hi,
                        default_value=np.median(uniq),
                    )
                    cs.add_hyperparameter(hp)
                continue

            # Categorical
            cs.add_hyperparameter(
                CategoricalHyperparameter(col, uniq)
            )

        return cs

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        return v

    def _config_to_dict(self, config: Configuration):
        return {p: self._clean(config[p]) for p in self.model_config.param_names}

    def _row_tuple(self, hyperparams: dict):
        return tuple(hyperparams[p] for p in self.model_config.param_names)

    def _valid_row_train(self, hyperparams: dict):
        """NN fallback using TRAIN KD-tree only."""
        q = [hyperparams[col] for col in self.columns]
        nn_row = self.nn_train.nearestRow(q)
        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}

    def _valid_row_test(self, hyperparams: dict):
        """NN fallback using TEST KD-tree only."""
        q = [hyperparams[col] for col in self.columns]
        nn_row = self.nn_test.nearestRow(q)
        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}

    # ----------------------------------------------------------
    # Main optimize
    # ----------------------------------------------------------
    def optimize(self):
        """Run SMAC using the already-prepared TRAIN/TEST/KD trees/configspace."""
        if not self.logging_util:
            raise ValueError("Logging utility not set!")

        print("Starting SMAC optimization (train/test already prepared in __init__)")

        total_budget = self.config["n_trials"]

      
        # Objective uses TRAIN nearest neighbor
        def objective(config: Configuration, seed: int = 0):
            raw_hp = self._config_to_dict(config)
            eval_hp = self._valid_row_train(raw_hp)

            key = self._row_tuple(eval_hp)
            if key in self.cache:
                return self.cache[key]

            score = self.model_wrapper.run_model(eval_hp)
            fitness = 1 - score
            self.logging_util.log(eval_hp, fitness, 1)

            self.cache[key] = fitness
            return fitness

        # Unique output dir per SMAC run
        output_directory = (
            f"{self.config['output_directory']}/"
            f"smac_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )

        scenario = Scenario(
            configspace=self.config_space,
            n_trials=total_budget,
            deterministic=True,
            output_directory=output_directory,
            seed=self.seed,
        )

        initial_design = HPOFacade.get_initial_design(
            scenario,
            n_configs=max(1, int(total_budget * 0.1)),
        )

        smac = HPOFacade(
            scenario=scenario,
            target_function=objective,
            initial_design=initial_design,
            overwrite=True,  # ensure fresh runs
        )

        self.logging_util.start_logging()
        try:
            incumbent = smac.optimize()
        except Exception:
            incumbent = smac.optimizer.intensifier.get_incumbent()

        # -----------------------
        # Evaluate TEST surrogate - FIXED VERSION
        # -----------------------

        model = smac._model
        
        # Get all test configurations as Configuration objects
        test_configs = []
        test_dicts = []

        for _, row in self.X_test.iterrows():
            hp = {c: self._clean(row[c]) for c in self.columns}
            test_dicts.append(hp)
            try:
                config_obj = Configuration(self.config_space, values=hp)
                test_configs.append(config_obj)
            except Exception as e:
                print(f"Warning: Could not create config object for {hp}: {e}")
                test_configs.append(None)

        # Filter out None configurations
        valid_test_configs = [c for c in test_configs if c is not None]
        valid_test_dicts = [d for c, d in zip(test_configs, test_dicts) if c is not None]

        if not valid_test_configs:
            print("Warning: No valid test configurations found, using incumbent")
            return None, None

        # Convert configurations to array format for the model
        test_vectors = convert_configurations_to_array(valid_test_configs)

        # Predict w/ RF surrogate
        try:
            mu, _ = model.predict(test_vectors)
        except Exception as e:
            print(f"Warning: Model prediction failed: {e}")
            # Fallback: just use the incumbent
            return None, None

        best_idx = int(np.argmin(mu))
        best_hp = valid_test_dicts[best_idx]

        # Optionally project using test KD-tree
        best_hp = self._valid_row_test(best_hp)

        self.best_config = best_hp
        self.best_value = objective(Configuration(self.config_space, values=best_hp))

        self.logging_util.stop_logging()

        return self.best_config, self.best_value