# Getting Started
I just dumped all my dependencies to a `req.txt` file. You can install them with pip:

```bash
pip install -r req.txt
```

# Create structured datasets from the raw data:
The raw data path expects the inputs that you (Vince) provided. 
Set the raw data path and the output paths in the `config.yaml` file. Then run the `main.py` script to generate the structured datasets.

# Preprocess the structured datasets:
Select the datasets you want to train on in the `diffusion_config.yaml` file. Also specify the modes, if any, you want to use for conditional generation. Then run the `preprocess_diffusion_data.py` script to preprocess the datasets for training.

# Train the diffusion model:
Set the training parameters in the `diffusion_config.yaml` file, then run the `train_diffusion.py` script to start training the diffusion model.

# Unfold with the trained model:
Use the `unfold_diffusion.py` script to unfold the structured datasets with the trained diffusion model. The model will unfold the dataset with the key `detector_sim_path`. 

# Evaluate the unfolded datasets:
Inspect the unfolded dataset with the diff.ipynb notebook. This will also convert the unfolded dataset from .npy to .csv format. Afterwards you can run `diffusion_eval.py` to create the Bell hist plot and Gellmann coefficient plot for the unfolded dataset.