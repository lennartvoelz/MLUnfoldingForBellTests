import argparse
import copy
import logging
import os
from typing import Any, Dict, Tuple

import optuna

from src.reconstruction.diffusion.unified_config import (
    load_unified_config,
    UnifiedDiffusionConfig,
)
from train_diffusion import train_diffusion_model_from_config

logger = logging.getLogger(__name__)


def _using_precomputed_datasets(config: UnifiedDiffusionConfig) -> bool:
    """Return True if coordinated precomputed datasets are available.

    We use the same logic as ``load_training_data`` in ``train_diffusion``:
    if both ``detector_dataset_path`` and ``truth_dataset_path`` exist on disk
    we consider the run to be using precomputed, coordinated CSV files.
    """

    file_paths = config.get_file_paths()
    det_path = file_paths.get("detector_dataset_path")
    truth_path = file_paths.get("truth_dataset_path")

    if not det_path or not truth_path:
        return False

    return os.path.exists(det_path) and os.path.exists(truth_path)


def build_trial_config(
    base_config: UnifiedDiffusionConfig,
    trial: optuna.Trial
) -> UnifiedDiffusionConfig:
    """Create a per-trial copy of the config with sampled hyperparameters.

    This function mutates a deep copy of ``base_config`` so the original
    configuration loaded from YAML remains unchanged.
    """

    # Deep copy to avoid side effects across trials
    config: UnifiedDiffusionConfig = copy.deepcopy(base_config)

    # --- Sample hyperparameters ---
    # Learning rate (log-uniform over a typical range)
    # lr = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)

    # L2 regularization strength for Adam (weight_decay). We sample on a
    # log scale over a typical small range; values near the lower end behave
    # similarly to having no weight decay.
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-6, log=True)

    # Hidden dimension for the diffusion network
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])

    # Number of epochs (kept relatively small to make search tractable)
    # epochs = trial.suggest_int("epochs", 30, 50, step=10)

    # Normalisation ranges for pT and E (in GeV) - separate hyperparameters
    # pT_range = trial.suggest_int("pT_range", 250, 500, step=50)
    # E_range = trial.suggest_int("E_range", 400, 600, step=50)

    # Dropout rate
    dropout = trial.suggest_float("dropout", 0.0, 0.15, step=0.01)

    # Diffusion schedule parameters around current defaults
    # beta_1 = trial.suggest_float("beta_1", 1e-5, 1e-3, log=True)
    # beta_T = trial.suggest_float("beta_T", 5e-3, 5e-2, log=True)
    # T = trial.suggest_int("T", 1000, 1600, step=100)

    # Model depth
    num_layers = trial.suggest_int("num_layers", 3, 8, step=1)

    # --- Apply to config dicts ---
    # Training hyperparameters
    # config.training["learning_rate"] = float(lr)
    config.training["epochs"] = 35
    config.training["weight_decay"] = float(weight_decay)

    # Model hyperparameters
    config.model["hidden_dim"] = int(hidden_dim)
    # config.model["beta_1"] = float(beta_1)
    # config.model["beta_T"] = float(beta_T)
    # config.model["T"] = int(T)
    config.model["num_layers"] = int(num_layers)
    config.model["dropout"] = float(dropout)

    # Normalisation ranges (separate for pT and E)
    # config.data_processing["pT_range"] = int(pT_range)
    # config.data_processing["E_range"] = int(E_range)

    # Give each trial a unique train_type so checkpoints and history files
    # do not overwrite each other.
    base_train_type = config.training.get("train_type", "default")
    config.training["train_type"] = f"{base_train_type}_trial{trial.number}"

    # Recompute all derived values (input_dim, norm_vec, state_name, etc.)
    # so that they are consistent with the sampled hyperparameters.
    config._compute_derived_values()

    return config


def objective(
    trial: optuna.Trial,
    base_config: UnifiedDiffusionConfig,
) -> float:
    """Optuna objective that trains the diffusion model and returns best val loss."""

    config = build_trial_config(base_config, trial)

    # Run training; this will create its own checkpoints and history files
    _, (_, val_loss_list, epoch_list) = train_diffusion_model_from_config(config)

    if not val_loss_list:
        raise RuntimeError("No validation losses recorded; check save_interval settings.")

    best_val = min(val_loss_list)

    # Report the best val loss to Optuna
    trial.set_user_attr("best_epoch", int(epoch_list[val_loss_list.index(best_val)]))
    return best_val


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for diffusion model.")
    parser.add_argument("--config", default="diffusion_config.yaml", help="Path to diffusion config YAML")
    parser.add_argument("--n-trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument(
        "--study-name",
        default="diffusion_hparam_search",
        help="Optuna study name (for logging and storage)",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL (e.g. sqlite:///diffusion_optuna.db). "
            "If omitted, an in-memory study is used."
        ),
    )
    args = parser.parse_args()

    # Load base configuration once; each trial will deep-copy and modify it
    base_config = load_unified_config(args.config)

    # Set up Optuna study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
        )

    study.optimize(
        lambda trial: objective(
            trial,
            base_config
        ),
        n_trials=args.n_trials,
    )

    logger.info("Best trial:")
    best = study.best_trial
    logger.info(f"  Value (best validation loss): {best.value}")
    logger.info("  Hyperparameters:")
    for k, v in best.params.items():
        logger.info(f"    {k}: {v}")
    if "best_epoch" in best.user_attrs:
        logger.info(f"  Best epoch: {best.user_attrs['best_epoch']}")


if __name__ == "__main__":
    main()
