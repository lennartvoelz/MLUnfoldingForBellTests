import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path, processed_features_path=None, processed_targets_path=None, cuts=True, splits=True):
        self.data_path = data_path
        self.processed_features_path = processed_features_path
        self.processed_targets_path = processed_targets_path
        self.cuts = cuts
        self.splits = splits

    def load_data(self):
        raw_data_path = self.data_path
        self.data = pd.read_csv(raw_data_path)
        # Only take first 100000 rows
        self.data = self.data.head(100000)

    def process_features(self):
        lep0 = self.data[['p_l_1_E', 'p_l_1_x', 'p_l_1_y', 'p_l_1_z']]
        lep1 = self.data[['p_l_2_E', 'p_l_2_x', 'p_l_2_y', 'p_l_2_z']]
        
        if 'mpx' not in self.data.columns:
            mpx = (self.data['p_v_1_x'] + self.data['p_v_2_x']).to_frame(name='mpx')
            mpy = (self.data['p_v_1_y'] + self.data['p_v_2_y']).to_frame(name='mpy')
        else:
            mpx = self.data[['mpx']]
            mpy = self.data[['mpy']]

        self.X = pd.concat([lep0, lep1, mpx, mpy], axis=1)

    def process_targets(self):
        y1 = self.data[['p_v_1_E', 'p_v_1_x', 'p_v_1_y', 'p_v_1_z']]
        y2 = self.data[['p_v_2_E', 'p_v_2_x', 'p_v_2_y', 'p_v_2_z']]

        self.y = pd.concat([y1, y2], axis=1)

    def scale_data(self):
        self.X /= 1e3
        self.y /= 1e3

    def save_data(self):
        features_path = self.config['data']['processed_features_path']
        targets_path = self.config['data']['processed_targets_path']
        self.X.to_csv(features_path, index=False)
        self.y.to_csv(targets_path, index=False)

    def create_split(self):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()

    def p_T(self, lep_x, lep_y):
        return np.sqrt(lep_x**2 + lep_y**2)
    
    def p(self, lep_x, lep_y, lep_z):
        return np.sqrt(lep_x**2 + lep_y**2 + lep_z**2)
    
    def eta(self, p, pz):
        return np.abs(1/2 * np.log((p + pz)/(p - pz)))

    def apply_selection_cuts(self):
        self.X = self.X.assign(lep0_pT = self.p_T(self.X['p_l_1_x'], self.X['p_l_1_y']))
        self.X = self.X.assign(lep1_pT = self.p_T(self.X['p_l_2_x'], self.X['p_l_2_y']))
        self.X = self.X.assign(lep0_p = self.p(self.X['p_l_1_x'], self.X['p_l_1_y'], self.X['p_l_1_z']))
        self.X = self.X.assign(lep1_p = self.p(self.X['p_l_2_x'], self.X['p_l_2_y'], self.X['p_l_2_z']))
        self.X = self.X.assign(lep0_eta = self.eta(self.X['lep0_p'], self.X['p_l_1_z']))
        self.X = self.X.assign(lep1_eta = self.eta(self.X['lep1_p'], self.X['p_l_2_z']))

        self.X = self.X[(self.X.lep0_pT > 22.0) & (self.X.lep1_pT > 15.0) & (self.X.lep0_eta < 2.5) & (self.X.lep1_eta < 2.5)]
        self.y = self.y.loc[self.X.index]
        self.X = self.X.drop(columns=['lep0_pT', 'lep1_pT', 'lep0_p', 'lep1_p', 'lep0_eta', 'lep1_eta'])

    def run_preprocessing(self) -> tuple:
        self.load_data()
        self.process_features()
        self.process_targets()
        #self.scale_data()
        if self.cuts:
            self.apply_selection_cuts()
        if self.processed_features_path and self.processed_targets_path:
            self.save_data()
        if self.splits:
            return self.create_split()
        else:
            return self.X.to_numpy(), self.y.to_numpy()