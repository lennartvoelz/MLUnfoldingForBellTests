import os
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


def pick_device(requested=None):
    if requested and "cuda" in str(requested).lower() and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


dev = pick_device(os.environ.get("MODEL_DEVICE"))


def load_training_data(config):
    print("Loading coordinated training data...")

    # Get file paths from unified config
    file_paths = config.get_file_paths()
    detector_dataset_path = file_paths.get(
        "detector_dataset_path", "diffusion_training_dataset.csv"
    )
    truth_dataset_path = file_paths.get(
        "truth_dataset_path", "diffusion_training_targets.csv"
    )

    if os.path.exists(detector_dataset_path) and os.path.exists(truth_dataset_path):
        print("Loading coordinated datasets:")
        print(f"  Detector features: {detector_dataset_path}")
        print(f"  Truth targets: {truth_dataset_path}")

        # Load the coordinated datasets
        detector_data = pd.read_csv(detector_dataset_path)
        truth_data = pd.read_csv(truth_dataset_path)

        # Verify row correspondence
        if len(detector_data) != len(truth_data):
            raise ValueError(
                f"Row count mismatch: detector {len(detector_data)} vs truth {len(truth_data)}"
            )

        print(f"Loaded coordinated datasets: {len(detector_data)} rows each")
        print(
            f"Detector columns: {len(detector_data.columns)}, Truth columns: {len(truth_data.columns)}"
        )

        # Extract process types for compatibility
        if "process" in detector_data.columns:
            types = detector_data["process"].unique()
            print(f"Process types: {list(types)}")
        else:
            types = ["unknown"]

        return detector_data, truth_data, types

    else:
        print("Coordinated datasets not found, falling back to legacy data loading...")
        print(
            "Please run preprocess_diffusion_data.py first to generate coordinated datasets."
        )

        # Fallback to original data loading method using unified config paths
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

        # Get preprocessed data
        _, _, types = data_preprocessor.run_preprocessing()

        # Load detector simulation and truth data separately
        detector_data = pd.read_csv(file_paths.get("detector_sim_path") + "_cuts.csv")
        truth_data = pd.read_csv(file_paths.get("truth_path") + "_cuts.csv")
        print(
            f"Detector data shape: {detector_data.shape}, Truth data shape: {truth_data.shape}"
        )

        return detector_data, truth_data, types


def prepare_diffusion_data(detector_data, truth_data, diffusion_config):
    print("Preparing diffusion training data...")

    # Initialize diffusion data preprocessor
    diff_preprocessor = DiffusionDataPreprocessor(diffusion_config)

    # Check if data is already preprocessed with conditioning features
    conditioning_columns = [
        col
        for col in detector_data.columns
        if col.startswith(("mom_pT_", "eta_moment_", "phi_moment_"))
    ]

    if conditioning_columns:
        print(f"Found {len(conditioning_columns)} precomputed conditioning features")
        print("Using preprocessed data with existing conditioning features...")

        # Extract four-vector components (detector measurements)
        # Note: Four-vector columns are handled directly in the processing below

        # Convert missing momentum to 4-vector format
        mpx = detector_data["mpx"].values
        mpy = detector_data["mpy"].values
        mpt = np.sqrt(mpx**2 + mpy**2)
        missing_4vec = np.column_stack([mpt, mpx, mpy, np.zeros_like(mpx)])

        # Combine detector four-vectors
        detector_4vecs = np.concatenate(
            [
                detector_data[["p_l_1_E", "p_l_1_x", "p_l_1_y", "p_l_1_z"]].values,
                detector_data[["p_l_2_E", "p_l_2_x", "p_l_2_y", "p_l_2_z"]].values,
                missing_4vec,
            ],
            axis=1,
        )

        # Extract conditioning features (if any)
        if conditioning_columns and diffusion_config.moment_conditioning_enabled:
            conditioning_features = detector_data[conditioning_columns].values
            # Combine detector measurements with conditioning features
            X = np.concatenate([detector_4vecs, conditioning_features], axis=1)
        else:
            # No conditioning features - use only detector measurements
            X = detector_4vecs
            print("No moment conditioning - using only detector measurements")

        # Extract truth-level neutrino four-vectors
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

        # Check if truth columns exist
        missing_truth_cols = [
            col for col in truth_columns if col not in truth_data.columns
        ]
        if missing_truth_cols:
            print(f"Warning: Missing truth columns: {missing_truth_cols}")
            print("Available truth columns:", list(truth_data.columns))
            # Try alternative column names
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
                print("Using alternative truth column names (without '_truth' suffix)")
                truth_columns = alt_truth_columns
            else:
                raise ValueError("Cannot find required truth columns in truth_data")

        y = truth_data[truth_columns].values

        print(f"Preprocessed data shapes: X={X.shape}, y={y.shape}")

    else:
        print("No precomputed conditioning features found, computing them now...")
        # Use original preprocessing method
        X, y = diff_preprocessor.prepare_training_data(detector_data, truth_data)

    # Normalize data
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


def train_diffusion_model(config_path="diffusion_config.yaml"):
    """Main training function for diffusion model."""

    # Load unified configuration
    config = load_unified_config(config_path)

    # Set seed
    config.set_seed()

    # Create directories
    config.create_directories()

    print("DIFFUSION MODEL TRAINING")
    print("Using unified configuration system")
    config.print_summary()

    # Load and prepare data
    detector_data, truth_data, _ = load_training_data(config)
    X, y = prepare_diffusion_data(detector_data, truth_data, config)

    print(f"Training data shape: X={X.shape}, y={y.shape}")

    # Create data loaders
    train_batches, val_batches = create_data_loaders(
        X, y, config.batch_size, config.training.get("train_split", 0.9)
    )

    print(
        f"Training batches: {len(train_batches)}, Validation batches: {len(val_batches)}"
    )

    # Initialize model using unified config
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

    # Move batches to device
    train_batches = [
        (x.to(config.device), y.to(config.device)) for x, y in train_batches
    ]
    val_batches = [(x.to(config.device), y.to(config.device)) for x, y in val_batches]

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = LinearLR(
        optimizer,
        1,
        1e-2,
        total_iters=config.epochs * len(train_batches),
    )

    # Training tracking
    train_loss_list = []
    val_loss_list = []
    epoch_list = []

    # Training loop
    pbar = tqdm.tqdm(total=config.epochs)

    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        cum_train_loss = 0

        for batch_x, batch_y in train_batches:
            optimizer.zero_grad()
            loss = model.loss_fn(batch_y, batch_x)  # y=truth, x=conditioning
            loss.backward()
            optimizer.step()
            scheduler.step()

            cum_train_loss += loss.item()

        train_loss = cum_train_loss / len(train_batches)

        # Validation
        if epoch % config.save_int == 0 or epoch == 5:
            model.eval()
            cum_val_loss = 0

            with torch.no_grad():
                for batch_x, batch_y in val_batches:
                    val_loss = model.loss_fn(batch_y, batch_x)
                    cum_val_loss += val_loss.item()

            val_loss = cum_val_loss / len(val_batches)

            # Save checkpoint
            checkpoint_path = os.path.join(
                config.ckpt_path,
                f"diffusion_{config.train_type}_b{config.batch_size}_it{epoch}.pth",
            )
            torch.save(model.state_dict(), checkpoint_path)

            # Save loss tracking
            epoch_list.append(epoch)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            pbar.set_description(
                f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}"
            )
        else:
            pbar.set_description(f"train loss: {train_loss:.6f}")

        pbar.update()

    # Plot and save loss curves
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

    # Save training history
    history_path = os.path.join(
        config.output_path, f"training_history_{config.state_name}.npz"
    )
    np.savez(
        history_path,
        epochs=epoch_list,
        train_loss=train_loss_list,
        val_loss=val_loss_list,
    )

    print("Training completed!")
    print(f"Final model saved to: {config.ckpt_path}")
    print(f"Training history saved: {history_path}")

    return model, (train_loss_list, val_loss_list, epoch_list)


if __name__ == "__main__":
    train_diffusion_model()
