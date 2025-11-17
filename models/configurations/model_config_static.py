import numpy as np
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant
)
from ConfigSpace import ConfigurationSpace


class ModelConfigurationStatic:
    def __init__(self, config, dataset_file, seed=42):
        self.config = config
        self.dataset_file = dataset_file
        self.seed = seed

        self.configspace = None
        self.param_names = None
        self.hyperparam_space = None
        self.hyperparam_dict = None

        self.get_configspace()

    def set_seed(self, seed):
        self.seed = seed
        if self.configspace:
            self.configspace.seed(seed)

    def get_dataset_file(self):
        return self.dataset_file

    def get_hyperparam_dict(self):
        return self.hyperparam_dict

    def get_hyperconfig_distribution(self):
        cs = ConfigurationSpace(seed=self.seed)

        for param_name, param_values in self.config.items():
            # deduplicate values
            values = sorted(list(set(param_values)))

            # -------------------------------------------------
            # CASE 1: Only one unique value → Constant
            # -------------------------------------------------
            if len(values) == 1:
                single_val = values[0]
                hp = Constant(param_name, single_val)
                cs.add_hyperparameter(hp)
                continue

            # -------------------------------------------------
            # CASE 2: Numeric (Ordinal Float)
            # -------------------------------------------------
            all_numeric = all(
                isinstance(v, (int, float, np.number))
                for v in values
            )

            if all_numeric:
                vmin, vmax = float(min(values)), float(max(values))

                # if range is too small, also force Constant
                if abs(vmax - vmin) < 1e-9:
                    hp = Constant(param_name, float(values[0]))
                else:
                    hp = UniformFloatHyperparameter(
                        name=param_name,
                        lower=vmin,
                        upper=vmax,
                        default_value=np.median(values),
                    )

            # -------------------------------------------------
            # CASE 3: Non-numeric → categorical
            # -------------------------------------------------
            else:
                hp = CategoricalHyperparameter(param_name, values)

            cs.add_hyperparameter(hp)

        return cs

    def get_configspace(self, recompute=False):
        if recompute or self.configspace is None:
            self.configspace = self.get_hyperconfig_distribution()
            self.param_names = list(self.config.keys())
            self.hyperparam_space = [list(set(v)) for v in self.config.values()]
            self.hyperparam_dict = {
                name: list(set(vals)) for name, vals in self.config.items()
            }
        return self.configspace, self.param_names, self.hyperparam_space

    def cs_to_dict(self, config):
        return config.get_dictionary()
