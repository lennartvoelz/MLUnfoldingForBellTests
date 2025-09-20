from src.data_preproc.preprocessing import DataPreprocessor
import yaml

config = yaml.safe_load(open('config.yaml'))

data = DataPreprocessor(data_path=config['data_path'], raw_data_path=config['raw_data_path'], truth_path=config['truth_path'], detector_sim_path=config['detector_sim_path'],
                        processed_features_path=config['processed_features_path'], processed_targets_path=config['processed_targets_path'],
                        cuts=True, splits=False, drop_zeroes=True)

X, y, types = data.run_preprocessing()