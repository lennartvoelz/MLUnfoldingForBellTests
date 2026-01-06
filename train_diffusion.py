import os
import logging
import tqdm
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR

from src.reconstruction.diffusion.model import Model
from src.reconstruction.diffusion.unified_config import load_unified_config
from src.reconstruction.diffusion.data_utils import DiffusionDataPreprocessor
from src.data_preproc.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Simple validation-based early stopping callback.

    Tracks the best validation loss seen so far and stops training when the
    loss has not improved for a configured number of epochs. Optionally
    restores the best model weights when stopping.
    """

    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)

        self.best_loss = float("inf")
        self.num_bad_epochs = 0
        self.best_state_dict = None
        self.stop = False

    def step(self, val_loss, model):
        """Update internal state given the latest validation loss."""

        if val_loss is None:
            return

        improved = val_loss < self.best_loss - self.min_delta

        if self.best_loss == float("inf") or improved:
            self.best_loss = val_loss
            self.num_bad_epochs = 0

            if self.restore_best_weights:
                # Store a copy of the best model weights on CPU to minimize
                # GPU memory usage.
                state_dict = model.state_dict()
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in state_dict.items()
                }
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stop = True

    def restore(self, model, device=None):
        """Restore the best model weights if available."""

        if not self.restore_best_weights or self.best_state_dict is None:
            return

        model.load_state_dict(self.best_state_dict)
        if device is not None:
            model.to(device)


def pick_device(requested=None):
    if requested and "cuda" in str(requested).lower() and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


dev = pick_device(os.environ.get("MODEL_DEVICE"))


def load_training_data(config):
    logger.info("Loading coordinated training data...")

    file_paths = config.get_file_paths()
    detector_dataset_path = file_paths.get(
        "detector_dataset_path", "diffusion_training_dataset.csv"
    )
    truth_dataset_path = file_paths.get(
        "truth_dataset_path", "diffusion_training_targets.csv"
    )

    if os.path.exists(detector_dataset_path) and os.path.exists(truth_dataset_path):
        logger.info(f"Loading coordinated datasets: {detector_dataset_path}, {truth_dataset_path}")

        detector_data = pd.read_csv(detector_dataset_path)
        truth_data = pd.read_csv(truth_dataset_path)

        if len(detector_data) != len(truth_data):
            raise ValueError(
                f"Row count mismatch: detector {len(detector_data)} vs truth {len(truth_data)}"
            )

        logger.info(f"Loaded coordinated datasets: {len(detector_data)} rows each")

        if "process" in detector_data.columns:
            types = detector_data["process"].unique()
        else:
            types = ["unknown"]

        return detector_data, truth_data, types

    else:
        logger.warning("Coordinated datasets not found, falling back to legacy data loading...")

        data_preprocessor = DataPreprocessor(
            data_path=file_paths.get("data_path"),
            raw_data_path=file_paths.get("raw_data_path"),
            truth_path=file_paths.get("truth_path"),
            detector_sim_path=file_paths.get("detector_sim_path"),
            processed_features_path=file_paths.get("processed_features_path", ""),
            processed_targets_path=file_paths.get("processed_targets_path", ""),
            cuts=True,
            splits=False,
            drop_zeroes=True,
        )

        _, _, types = data_preprocessor.run_preprocessing()

        detector_data = pd.read_csv(file_paths.get("detector_sim_path") + "_cuts.csv")
        truth_data = pd.read_csv(file_paths.get("truth_path") + "_cuts.csv")

        return detector_data, truth_data, types


def load_validation_data(config):
    """Load coordinated validation data if separate datasets are configured.

    Returns ``(detector_val_data, truth_val_data)`` when a separate validation
    dataset is available, or ``(None, None)`` when no separate validation source
    is configured or the files are missing.
    """

    file_paths = config.get_file_paths()
    det_val_path = file_paths.get("detector_val_dataset_path")
    truth_val_path = file_paths.get("truth_val_dataset_path")

    if not det_val_path or not truth_val_path:
        logger.info("No separate validation dataset paths configured; using in-dataset split.")
        return None, None

    if not (os.path.exists(det_val_path) and os.path.exists(truth_val_path)):
        logger.warning(f"Validation datasets not found: {det_val_path}, {truth_val_path}")
        return None, None

    logger.info(f"Loading coordinated validation datasets: {det_val_path}, {truth_val_path}")

    detector_val = pd.read_csv(det_val_path)
    truth_val = pd.read_csv(truth_val_path)

    if len(detector_val) != len(truth_val):
        raise ValueError(
            "Row count mismatch in validation data: "
            f"detector {len(detector_val)} vs truth {len(truth_val)}"
        )

    logger.info(f"Loaded validation datasets: {len(detector_val)} rows each")

    return detector_val, truth_val


def prepare_diffusion_data(detector_data, truth_data, diffusion_config):
    diff_preprocessor = DiffusionDataPreprocessor(diffusion_config)

    conditioning_columns = [
        col
        for col in detector_data.columns
        if col.startswith(
            ("mom_pT_", "eta_moment_", "phi_moment_", "mom_mt_", "mom_px_", "mom_py_")
        )
    ]

    if conditioning_columns:
        logger.info(f"Found {len(conditioning_columns)} precomputed conditioning features")

        mpx = detector_data["mpx"].values
        mpy = detector_data["mpy"].values
        mpt = np.sqrt(mpx**2 + mpy**2)
        missing_4vec = np.column_stack([mpt, mpx, mpy, np.zeros_like(mpx)])

        detector_4vecs = np.concatenate(
            [
                detector_data[["p_l_1_E", "p_l_1_x", "p_l_1_y", "p_l_1_z"]].values,
                detector_data[["p_l_2_E", "p_l_2_x", "p_l_2_y", "p_l_2_z"]].values,
                missing_4vec,
            ],
            axis=1,
        )

        if conditioning_columns and diffusion_config.moment_conditioning_enabled:
            conditioning_features = detector_data[conditioning_columns].values
            X = np.concatenate([detector_4vecs, conditioning_features], axis=1)
        else:
            conditioning_features = np.zeros((detector_4vecs.shape[0], 0))
            X = detector_4vecs

        truth_columns = [
            "p_v_1_E_truth",
            "p_v_1_x_truth",
            "p_v_1_y_truth",
            "p_v_1_z_truth",
            "p_v_2_E_truth",
            "p_v_2_x_truth",
            "p_v_2_y_truth",
            "p_v_2_z_truth",
        ]

        missing_truth_cols = [
            col for col in truth_columns if col not in truth_data.columns
        ]
        if missing_truth_cols:
            logger.warning(f"Missing truth columns: {missing_truth_cols}")
            alt_truth_columns = [
                "p_v_1_E",
                "p_v_1_x",
                "p_v_1_y",
                "p_v_1_z",
                "p_v_2_E",
                "p_v_2_x",
                "p_v_2_y",
                "p_v_2_z",
            ]
            if all(col in truth_data.columns for col in alt_truth_columns):
                truth_columns = alt_truth_columns
            else:
                raise ValueError("Cannot find required truth columns in truth_data")

        y = truth_data[truth_columns].values

        if getattr(diffusion_config, "predict_conditioning_features", False):
            y = np.concatenate([y, conditioning_features], axis=1)

    else:
        X, y = diff_preprocessor.prepare_training_data(detector_data, truth_data)

    X_norm, y_norm = diff_preprocessor.normalize_data(X, y)

    return X_norm, y_norm


def create_data_loaders(X, y, batch_size, train_split=0.9):
    """Create training and validation data loaders."""
    n_samples = X.shape[0]
    n_train = int(train_split * n_samples)

    # Split data
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    # Create batches
    n_train_batches = len(X_train) // batch_size
    n_val_batches = len(X_val) // batch_size

    train_batches = []
    for i in range(n_train_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        train_batches.append(
            (X_train_tensor[start_idx:end_idx], y_train_tensor[start_idx:end_idx])
        )

    val_batches = []
    for i in range(n_val_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        val_batches.append(
            (X_val_tensor[start_idx:end_idx], y_val_tensor[start_idx:end_idx])
        )

    return train_batches, val_batches


def create_data_loaders_from_splits(X_train, y_train, X_val, y_val, batch_size):
    """Create data loaders from explicit training and validation arrays.

    This helper mirrors :func:`create_data_loaders` but assumes that ``X_train``
    and ``X_val`` already represent disjoint datasets (e.g. when using a
    separate validation CSV).
    """

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    # Create batches
    n_train_batches = len(X_train) // batch_size
    n_val_batches = len(X_val) // batch_size

    train_batches = []
    for i in range(n_train_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        train_batches.append(
            (X_train_tensor[start_idx:end_idx], y_train_tensor[start_idx:end_idx])
        )

    val_batches = []
    for i in range(n_val_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        val_batches.append(
            (X_val_tensor[start_idx:end_idx], y_val_tensor[start_idx:end_idx])
        )

    return train_batches, val_batches


def train_diffusion_model_from_config(config):
    """Train diffusion model given a pre-loaded UnifiedDiffusionConfig.

    This is the core training routine used both by the CLI entry point and by
    external scripts (e.g. hyperparameter search) that want to construct and
    modify the config in Python before training.
    """

    config.set_seed()
    config.create_directories()

    logger.info("Starting diffusion model training")
    config.print_summary()

    detector_data, truth_data, _ = load_training_data(config)
    detector_val_data, truth_val_data = load_validation_data(config)

    if detector_val_data is None or truth_val_data is None:
        X, y = prepare_diffusion_data(detector_data, truth_data, config)
        logger.info(f"Training/validation data shape: X={X.shape}, y={y.shape}")

        expected_input_dim = int(config.input_dim)
        actual_input_dim = int(X.shape[1])
        if expected_input_dim != actual_input_dim:
            raise RuntimeError(
                "Input dimension mismatch between config and training data: "
                f"config.input_dim={expected_input_dim}, X.shape[1]={actual_input_dim}. "
                "This usually indicates that the moment-conditioning "
                "configuration (pt/eta/phi/mt/px/py moments) used for the "
                "current config or Optuna trial does not match the conditioning "
                "features present in the precomputed datasets. When using "
                "precomputed CSVs, either fix the moment counts in the base "
                "config to match the dataset layout, or restrict the "
                "hyperparameter search so that trials do not change the "
                "conditioning configuration."
            )

        train_batches, val_batches = create_data_loaders(
            X, y, config.batch_size, config.training.get("train_split", 0.9)
        )
    else:
        logger.info("Using separate datasets for training and validation.")

        X_train, y_train = prepare_diffusion_data(detector_data, truth_data, config)
        X_val, y_val = prepare_diffusion_data(
            detector_val_data, truth_val_data, config
        )

        logger.info(
            f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}; "
            f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}"
        )

        expected_input_dim = int(config.input_dim)
        train_input_dim = int(X_train.shape[1])
        val_input_dim = int(X_val.shape[1])

        if train_input_dim != expected_input_dim or val_input_dim != expected_input_dim:
            raise RuntimeError(
                "Input dimension mismatch between config and training/validation data: "
                f"config.input_dim={expected_input_dim}, "
                f"X_train.shape[1]={train_input_dim}, X_val.shape[1]={val_input_dim}. "
                "This usually indicates that the conditioning configuration "
                "(pt/eta/phi/mt/px/py moments) used for the current config or "
                "Optuna trial does not match the conditioning features present "
                "in the precomputed datasets. When using precomputed CSVs, "
                "either fix the moment counts in the base config to match the "
                "dataset layout, or restrict the hyperparameter search so that "
                "trials do not change the conditioning configuration."
            )

        train_batches, val_batches = create_data_loaders_from_splits(
            X_train, y_train, X_val, y_val, config.batch_size
        )

    logger.info(f"Training batches: {len(train_batches)}, Validation batches: {len(val_batches)}")

    model = torch.compile(
        Model(
            device=config.device,
            beta_1=config.beta_1,
            beta_T=config.beta_T,
            T=config.T,
            input_dim=config.input_dim,
            output_dim=config.output_dim,
        )
    )

    train_batches = [
        (x.to(config.device), y.to(config.device)) for x, y in train_batches
    ]
    val_batches = [(x.to(config.device), y.to(config.device)) for x, y in val_batches]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=getattr(config, "weight_decay", 0.0)
    )
    scheduler = LinearLR(
        optimizer,
        1,
        1e-2,
        total_iters=config.epochs * len(train_batches),
    )

    train_loss_list = []
    val_loss_list = []
    epoch_list = []

    early_stopping = None
    patience = getattr(config, "early_stopping_patience", 0)
    if len(val_batches) > 0 and patience > 0:
        early_stopping = EarlyStopping(
            patience=patience,
            restore_best_weights=getattr(
                config, "early_stopping_restore_best", True
            ),
        )

    pbar = tqdm.tqdm(total=config.epochs)

    for epoch in range(1, config.epochs + 1):
        model.train()
        cum_train_loss = 0

        for batch_x, batch_y in train_batches:
            optimizer.zero_grad()
            loss = model.loss_fn(batch_y, batch_x)
            loss.backward()
            optimizer.step()
            scheduler.step()

            cum_train_loss += loss.item()

        train_loss = cum_train_loss / len(train_batches)

        run_validation = False
        if len(val_batches) > 0:
            if early_stopping is not None:
                run_validation = True
            elif epoch % config.save_int == 0 or epoch == 5:
                run_validation = True

        val_loss = None
        if run_validation:
            model.eval()
            cum_val_loss = 0

            with torch.no_grad():
                for batch_x, batch_y in val_batches:
                    batch_val_loss = model.loss_fn(batch_y, batch_x)
                    cum_val_loss += batch_val_loss.item()

            val_loss = cum_val_loss / len(val_batches)

            if early_stopping is not None:
                early_stopping.step(val_loss, model)

                if early_stopping.stop:
                    if early_stopping.restore_best_weights:
                        early_stopping.restore(model, config.device)

                    pbar.set_description(
                        f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f} (early stop)"
                    )
                    pbar.update()
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} with best val loss {early_stopping.best_loss:.6f}"
                    )
                    checkpoint_path = os.path.join(
                        config.ckpt_path,
                        f"diffusion_{config.train_type}_b{config.batch_size}_it{epoch}_final.pth",
                    )
                    torch.save(model.state_dict(), checkpoint_path)
                    break

            if epoch % config.save_int == 0 or epoch == 5:
                checkpoint_path = os.path.join(
                    config.ckpt_path,
                    f"diffusion_{config.train_type}_b{config.batch_size}_it{epoch}.pth",
                )
                torch.save(model.state_dict(), checkpoint_path)

                epoch_list.append(epoch)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

            pbar.set_description(
                f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}"
            )
        else:
            pbar.set_description(f"train loss: {train_loss:.6f}")

        pbar.update()

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_loss_list, label="Training Loss")
    plt.plot(epoch_list, val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Diffusion Model Training Loss")
    plt.savefig(
        os.path.join(
            config.plots_path,
            f"loss_{config.train_type}_it{config.epochs}.png",
        )
    )
    plt.close()

    history_path = os.path.join(
        config.output_path, f"training_history_{config.state_name}.npz"
    )
    np.savez(
        history_path,
        epochs=epoch_list,
        train_loss=train_loss_list,
        val_loss=val_loss_list,
    )

    logger.info(f"Training completed! Model saved to: {config.ckpt_path}")
    logger.info(f"Training history saved: {history_path}")

    return model, (train_loss_list, val_loss_list, epoch_list)


def train_diffusion_model(config_path="diffusion_config.yaml"):
    """Main training function for diffusion model.

    This preserves the existing CLI behaviour (load config from YAML) while
    delegating the actual work to :func:`train_diffusion_model_from_config`.
    """

    config = load_unified_config(config_path)
    return train_diffusion_model_from_config(config)


if __name__ == "__main__":
    train_diffusion_model()
