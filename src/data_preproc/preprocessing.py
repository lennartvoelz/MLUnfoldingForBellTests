import pandas as pd
import logging

class DataPreprocessor:
    def __init__(self, data_path, processed_features_path, processed_targets_path):
        self.data_path = data_path
        self.processed_features_path = processed_features_path
        self.processed_targets_path = processed_targets_path
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def load_data(self):
        raw_data_path = self.data_path
        self.logger.info(f"Loading data from {raw_data_path}")
        self.data = pd.read_csv(raw_data_path)
        self.logger.info("Data loaded successfully.")

    def process_features(self):
        self.logger.info("Processing features.")

        lep0 = self.data[['p_l_1_E', 'p_l_1_x', 'p_l_1_y', 'p_l_1_z']]
        lep1 = self.data[['p_l_2_E', 'p_l_2_x', 'p_l_2_y', 'p_l_2_z']]
        mpx = (self.data['p_v_1_x'] + self.data['p_v_2_x']).to_frame(name='mpx')
        mpy = (self.data['p_v_1_y'] + self.data['p_v_2_y']).to_frame(name='mpy')

        self.X = pd.concat([lep0, lep1, mpx, mpy], axis=1)
        self.logger.info("Features processed successfully.")

    def process_targets(self):
        self.logger.info("Processing targets.")
        y1 = self.data[['p_v_1_E', 'p_v_1_x', 'p_v_1_y', 'p_v_1_z']]
        y2 = self.data[['p_v_2_E', 'p_v_2_x', 'p_v_2_y', 'p_v_2_z']]

        self.y = pd.concat([y1, y2], axis=1)
        self.logger.info("Targets processed successfully.")

    def scale_data(self):
        self.logger.info("Scaling data.")
        self.X /= 1e3
        self.y /= 1e3
        self.logger.info("Data scaling completed.")

    def save_data(self):
        features_path = self.config['data']['processed_features_path']
        targets_path = self.config['data']['processed_targets_path']
        self.logger.info(f"Saving features to {features_path}")
        self.X.to_csv(features_path, index=False)
        self.logger.info(f"Saving targets to {targets_path}")
        self.y.to_csv(targets_path, index=False)
        self.logger.info("Data saved successfully.")