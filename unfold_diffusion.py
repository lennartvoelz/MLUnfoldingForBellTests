import torch
import tqdm
import os
import numpy as np
import pandas as pd

from src.reconstruction.diffusion import (
    DiffusionProcess,
    Model,
    DiffusionDataPreprocessor,
)
from src.reconstruction.diffusion.unified_config import load_unified_config
from src.data_preproc.preprocessing import DataPreprocessor


def load_test_data(config):
    """Load test data for unfolding."""
    print("Loading test data...")

    # Get file paths from unified config
    file_paths = config.get_file_paths()
    detector_sim_path = file_paths.get("detector_sim_path")

    if not detector_sim_path:
        raise ValueError("detector_sim_path not found in configuration")

    # Load detector simulation data for unfolding
    detector_data = pd.read_csv(detector_sim_path + "_cuts.csv")

    return detector_data


def unfold_with_diffusion(config_path="diffusion_config.yaml", model_path=None):
    """Unfold events using trained diffusion model."""

    # Load unified configuration
    config = load_unified_config(config_path)

    # Set seed
    config.set_seed()

    print("DIFFUSION MODEL UNFOLDING")
    print("Using unified configuration system")
    config.print_summary()

    # Load test data
    detector_data = load_test_data(config)

    # Limit data size for unfolding
    if len(detector_data) > config.unfold_size:
        detector_data = detector_data.head(config.unfold_size)

    print(f"Unfolding {len(detector_data)} events")

    # Initialize data preprocessor
    data_preprocessor = DiffusionDataPreprocessor(config)

    # Prepare detector data
    lep1_4vec, lep2_4vec, missing_4vec = data_preprocessor.convert_to_four_vectors(
        detector_data
    )
    conditioning_features = data_preprocessor.calculate_conditioning_features(
        lep1_4vec, lep2_4vec, missing_4vec
    )

    # Combine features
    detector_features = np.concatenate([lep1_4vec, lep2_4vec, missing_4vec], axis=1)

    # Handle conditioning features (might be empty if moment conditioning disabled)
    if config.moment_conditioning_enabled and conditioning_features.shape[1] > 0:
        X = np.concatenate([detector_features, conditioning_features], axis=1)
        print(
            f"Using moment conditioning with {conditioning_features.shape[1]} features"
        )
    else:
        X = detector_features
        print("No moment conditioning - using only detector measurements")

    # Normalize
    X_norm, _ = data_preprocessor.normalize_data(X, np.zeros((X.shape[0], 8)))
    X_tensor = torch.from_numpy(X_norm).float().to(config.device)

    # Load trained model
    if model_path is None:
        model_path = os.path.join(
            config.ckpt_path,
            f"diffusion_{config.train_type}_b{config.batch_size}_it{config.epochs}.pth",
        )

    print(f"Loading model from: {model_path}")

    # Initialize model using unified config
    model_config = config.get_model_config()
    model = Model(
        device=config.device,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        T=config.T,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
    )

    # Load state dict with cleanup for compiled models
    state_dict = torch.load(model_path, weights_only=True, map_location=config.device)
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    # Initialize diffusion process
    diffusion_process = DiffusionProcess(
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        T=config.T,
        diffusion_fn=model,
        device=config.device,
        shape=(config.output_dim,),
    )

    print("Unfolding events...")

    # Unfold in batches
    unfolded_results = []
    n_batches = len(X_tensor) // config.sample_size

    pbar = tqdm.tqdm(total=n_batches)

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * config.sample_size
            end_idx = (i + 1) * config.sample_size

            batch_conditioning = X_tensor[start_idx:end_idx]

            # Generate samples
            unfolded_batch = diffusion_process.sampling(
                config.sample_size, batch_conditioning
            )

            unfolded_results.append(unfolded_batch.cpu().numpy())
            pbar.update()

    # Handle remaining events
    if len(X_tensor) % config.sample_size != 0:
        remaining_start = n_batches * config.sample_size
        remaining_conditioning = X_tensor[remaining_start:]

        remaining_unfolded = diffusion_process.sampling(
            len(remaining_conditioning), remaining_conditioning
        )

        unfolded_results.append(remaining_unfolded.cpu().numpy())

    # Combine results
    unfolded = np.concatenate(unfolded_results, axis=0)

    # Denormalize
    unfolded_denorm = data_preprocessor.denormalize_output(unfolded)

    # Prepare output format (add event numbers)
    event_numbers = np.arange(len(unfolded_denorm))
    output_data = np.column_stack([event_numbers, unfolded_denorm])

    # Save results
    output_path = os.path.join(
        config.output_path,
        f"unfold_diffusion_{config.unf_type}.npy",
    )

    os.makedirs(config.output_path, exist_ok=True)
    np.save(output_path, output_data)

    print(f"Unfolding completed! Results saved to: {output_path}")
    print(f"Unfolded {len(unfolded_denorm)} events")

    return output_data


def evaluate_unfolding_quality(unfolded_path, truth_path):
    """Evaluate quality of unfolded results against truth."""
    print("Evaluating unfolding quality...")

    # Load unfolded results
    unfolded_data = np.load(unfolded_path)
    unfolded_neutrinos = unfolded_data[:, 1:]  # Remove event numbers

    # Load truth data
    truth_data = pd.read_csv(truth_path + "_cuts.csv")
    truth_neutrinos = truth_data[
        [
            "p_v_1_E_truth",
            "p_v_1_x_truth",
            "p_v_1_y_truth",
            "p_v_1_z_truth",
            "p_v_2_E_truth",
            "p_v_2_x_truth",
            "p_v_2_y_truth",
            "p_v_2_z_truth",
        ]
    ].values

    # Ensure same number of events
    min_events = min(len(unfolded_neutrinos), len(truth_neutrinos))
    unfolded_neutrinos = unfolded_neutrinos[:min_events]
    truth_neutrinos = truth_neutrinos[:min_events]

    # Calculate metrics
    diff = unfolded_neutrinos - truth_neutrinos
    mae = np.mean(np.abs(diff), axis=0)
    mse = np.mean(diff**2, axis=0)
    rmse = np.sqrt(mse)

    print("Evaluation Results:")
    print(f"Mean Absolute Error (per component): {mae}")
    print(f"Root Mean Square Error (per component): {rmse}")
    print(f"Overall MAE: {np.mean(mae):.4f}")
    print(f"Overall RMSE: {np.mean(rmse):.4f}")

    return {
        "mae_per_component": mae,
        "rmse_per_component": rmse,
        "mae_total": np.mean(mae),
        "rmse_total": np.mean(rmse),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unfold events using diffusion model")
    parser.add_argument(
        "--config", default="diffusion_config.yaml", help="Configuration file path"
    )
    parser.add_argument("--model", default=None, help="Model checkpoint path")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate against truth data"
    )

    args = parser.parse_args()

    # Perform unfolding
    output_data = unfold_with_diffusion(args.config, args.model)

    # Evaluate if requested
    if args.evaluate:
        config = load_unified_config(args.config)
        file_paths = config.get_file_paths()

        unfolded_path = os.path.join(
            config.output_path,
            f"unfold_diffusion_{config.unf_type}.npy",
        )

        evaluate_unfolding_quality(unfolded_path, file_paths.get("truth_path"))
