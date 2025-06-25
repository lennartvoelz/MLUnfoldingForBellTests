from src.data_preproc.preprocessing import DataPreprocessor
import yaml

config = yaml.safe_load(open('config.yaml'))

data = DataPreprocessor(data_path=config['data_path'], raw_data_path=config['raw_data_path'], truth_path=config['truth_path'],
                        cuts=False, splits=False, drop_zeroes=True)

X, y, types = data.run_preprocessing()