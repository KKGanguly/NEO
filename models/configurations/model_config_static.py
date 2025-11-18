import numpy as np
from datetime import datetime
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant
)
from ConfigSpace import ConfigurationSpace
from utils.EncodingUtils import EncodingUtils
class ModelConfigurationStatic:
    def __init__(self, config, dataset_file, seed=42):
        self.config = config
        self.dataset_file = dataset_file
        self.seed = seed
        self.configspace = None
        self.param_names = None
        self.hyperparam_space = None
        self.hyperparam_dict = None
        # Store column type info for consistent encoding
        self.column_types = {}
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
            values = list(set(param_values))
            
            # Infer column type using shared utility
            col_type, processed_values = EncodingUtils.infer_column_type(values)
            self.column_types[param_name] = col_type
            
            # CASE 1 – constant
            if len(processed_values) == 1:
                cs.add_hyperparameter(Constant(param_name, processed_values[0]))
                continue
            
            # CASE 2 – date (converted to numeric)
            if col_type == 'date':
                vmin, vmax = min(processed_values), max(processed_values)
                cs.add_hyperparameter(
                    UniformFloatHyperparameter(
                        param_name,
                        lower=float(vmin),
                        upper=float(vmax),
                        default_value=(vmin + vmax) / 2
                    )
                )
                continue
            
            # CASE 3 – numeric
            if col_type == 'numeric':
                vmin, vmax = min(processed_values), max(processed_values)
                if abs(vmax - vmin) < 1e-12:
                    hp = Constant(param_name, float(processed_values[0]))
                else:
                    hp = UniformFloatHyperparameter(
                        name=param_name,
                        lower=vmin,
                        upper=vmax,
                        default_value=np.median(processed_values),
                    )
                cs.add_hyperparameter(hp)
                continue
            
            # CASE 4 – categorical with high cardinality
            if len(processed_values) > 40:
                numeric_vals = [float(abs(hash(v)) % 100000) for v in processed_values]
                vmin, vmax = min(numeric_vals), max(numeric_vals)
                cs.add_hyperparameter(
                    UniformFloatHyperparameter(
                        name=param_name,
                        lower=vmin,
                        upper=vmax,
                        default_value=np.median(numeric_vals),
                    )
                )
                continue
            
            # CASE 5 – normal categorical
            cs.add_hyperparameter(CategoricalHyperparameter(param_name, processed_values))
        
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