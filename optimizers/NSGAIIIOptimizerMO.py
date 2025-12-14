import optuna
from optuna.samplers import NSGAIIISampler

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from utils import DistanceUtil


class NSGAIIIOptimizer(BaseOptimizer):

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}

        # Number of objectives = length of score tuple
        self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.X_df.iloc[0][c] for c in self.columns}
            )
        )

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _objective(self, trial):
        raw_hp = {}

        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                raw_hp[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
            else:
                raise ValueError("Unsupported hyperparameter")

        valid_hp = self._nearest_row(raw_hp)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            return self.cache[key]

        try:
            scores = self.model_wrapper.get_score(valid_hp)
        except Exception:
            return tuple(0.0 for _ in range(self.num_objectives))

        self.cache[key] = scores
        return scores  # tuple → multi-objective

    def optimize(self):
        n_trials = self.config["n_trials"]

        sampler = NSGAIIISampler(seed=self.seed)
        study = optuna.create_study(
            directions=["maximize"] * self.num_objectives,
            sampler=sampler,
        )

        study.optimize(self._objective, n_trials=n_trials, catch=(Exception,))

        # Final population
        # Pareto-optimal trials only
        trials = study.best_trials

        best_trial = None
        best_d2h = float("inf")

        # Ideal point is implicitly [1, 1, ..., 1]
        ideal = [1] * self.num_objectives

        for t in trials:
            d2h = DistanceUtil.d2h(ideal, t.values)
            if d2h < best_d2h:
                best_d2h = d2h
                best_trial = t

        best_raw = dict(best_trial.params)
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                best_raw[hp.name] = hp.value

        final_hp = self._nearest_row(best_raw)

        self.best_config = final_hp
        self.best_value = best_d2h  # same scale as SMAC/TPE

        return final_hp, self.best_value

