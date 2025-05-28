# Modified with https://cernbox.cern.ch/jupyter/public/Ju7DYsj0y8sQe2j/GeorgesAnalysis.ipynb?contextRouteName=files-public-link&contextRouteParams.driveAliasAndItem=public/Ju7DYsj0y8sQe2j
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval

class DataPreprocessor:
    def __init__(self, raw_data_path, data_path, processed_features_path=None,
                 processed_targets_path=None, cuts=True, splits=True, mass_matching=False):
        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.processed_features_path = processed_features_path
        self.processed_targets_path = processed_targets_path
        self.cuts = cuts
        self.splits = splits
        self.mass_matching = mass_matching

    def event_type(self, array, ids):
        """
        Determines event type by specifying all particles in a specific array must occur exactly once.
        :param array: array containing particle ids in event.
        :param ids: list containing id numbers of required particles.
        :returns type: Bool True if all ids in array exactly once.
        """
        event_type = False
        type_count = 0
        # Require all particles to appear once only
        for id in ids:
            if np.sum(array == id) == 1:
                type_count +=1
        # Require all particles to be present
        if type_count == len(ids):
            event_type = True
        return event_type

    def get_event_type(self, array):
        # Define type-1 decays to be to electron and anti-muon, and type-2 decays to be to positron and muon
        ids_1 = [-13, 14, 11, -12]
        ids_2 = [13, -14, -11, 12]

        if self.event_type(np.array(array), ids_1):
            return 1
        elif self.event_type(np.array(array), ids_2):
            return 2
        return np.nan

    def extract_data(self):
        if not os.path.exists(self.data_path):
            # Read raw data
            df = pd.read_csv(self.raw_data_path)

            df.drop(columns=["Particle.PT", "Particle.Eta", "Particle.Phi"], inplace=True)

            # From: https://cernbox.cern.ch/files/link/public/Ju7DYsj0y8sQe2j?tiles-size=1&items-per-page=100&view-mode=resource-table
            df = pd.read_csv("data/hww_simulated.csv")
            df.drop(columns=["Particle.PT", "Particle.Eta", "Particle.Phi"], inplace=True)

            # Interpret list of values as list, save 4 last values and explode into 4 columns; replace with type
            df['Particle.PID'] = df['Particle.PID'].replace(r"  ", r", ", regex=True)
            df['Particle.PID'] = df['Particle.PID'].replace(r" -", r", -", regex=True)
            df['Particle.PID'] = df['Particle.PID'].replace(r"\[ ", r"[", regex=True)
            df['Particle.PID'] = df['Particle.PID'].apply(literal_eval)
            df['Particle.PID'] = df['Particle.PID'].apply(lambda x: x[-4:])

            # Apply literal_eval to each of the first 9 columns except PID
            for col in df.columns[1:]:
                df[col] = df[col].apply(literal_eval)
                df[col] = df[col].apply(lambda x: x[0] if len(x) > 0 else np.nan)
            df

            from src.utils.change_of_coordinates import exp_to_four_vec

            df[['Electron.E', 'Electron.px', 'Electron.py', 'Electron.pz']] = df.apply(
                lambda row: pd.Series(exp_to_four_vec(row['Electron.PT'], row['Electron.Eta'], row['Electron.Phi'], 0.000511)), axis=1)

            df[['Muon.E', 'Muon.px', 'Muon.py', 'Muon.pz']] = df.apply(
                lambda row: pd.Series(exp_to_four_vec(row['Muon.PT'], row['Muon.Eta'], row['Muon.Phi'], 0.10566)), axis=1)

            df['Event.Type'] = df['Particle.PID'].apply(self.get_event_type)
            df

            # l1 has highest p_T

            df["p_l_1_E"] = df.apply(lambda row: row["Electron.E"] if row["Electron.PT"] > row["Muon.PT"] else row["Muon.E"], axis=1)
            df["p_l_1_x"] = df.apply(lambda row: row["Electron.px"] if row["Electron.PT"] > row["Muon.PT"] else row["Muon.px"], axis=1)
            df["p_l_1_y"] = df.apply(lambda row: row["Electron.py"] if row["Electron.PT"] > row["Muon.PT"] else row["Muon.py"], axis=1)
            df["p_l_1_z"] = df.apply(lambda row: row["Electron.pz"] if row["Electron.PT"] > row["Muon.PT"] else row["Muon.pz"], axis=1)

            df["p_l_2_E"] = df.apply(lambda row: row["Muon.E"] if row["Electron.PT"] > row["Muon.PT"] else row["Electron.E"], axis=1)
            df["p_l_2_x"] = df.apply(lambda row: row["Muon.px"] if row["Electron.PT"] > row["Muon.PT"] else row["Electron.px"], axis=1)
            df["p_l_2_y"] = df.apply(lambda row: row["Muon.py"] if row["Electron.PT"] > row["Muon.PT"] else row["Electron.py"], axis=1)
            df["p_l_2_z"] = df.apply(lambda row: row["Muon.pz"] if row["Electron.PT"] > row["Muon.PT"] else row["Electron.pz"], axis=1)

            df.drop(columns=["Electron.E", "Electron.px", "Electron.py", "Electron.pz", "Muon.E", "Muon.px", "Muon.py", "Muon.pz", "Particle.PID", "Electron.PT", "Electron.Eta", "Electron.Phi", "Muon.PT", "Muon.Eta", "Muon.Phi"], inplace=True)


            df["mpx"] = df.apply(lambda row: row["MissingET.MET"] * np.cos(row["MissingET.Phi"]), axis=1)
            df["mpy"] = df.apply(lambda row: row["MissingET.MET"] * np.sin(row["MissingET.Phi"]), axis=1)

            df.drop(columns=["MissingET.MET", "MissingET.Phi", "MissingET.Eta"], inplace=True)

            # From https://cernbox.cern.ch/files/link/public/Ju7DYsj0y8sQe2j?tiles-size=1&items-per-page=100&view-mode=resource-table
            # Extracted csv from output_tlvs.root
            df_truth = pd.read_csv("data/df_final.csv")

            df["Neutrino_1.E"] = df_truth["v1_E"]
            df["Neutrino_1.x"] = df_truth["v1_p_x"]
            df["Neutrino_1.y"] = df_truth["v1_p_y"]
            df["Neutrino_1.z"] = df_truth["v1_p_z"]
            df["Neutrino_2.E"] = df_truth["v2_E"]
            df["Neutrino_2.x"] = df_truth["v2_p_x"]
            df["Neutrino_2.y"] = df_truth["v2_p_y"]
            df["Neutrino_2.z"] = df_truth["v2_p_z"]

            df = df.dropna(how='any').copy()
            # To infer which lepton does each neutrino associated with, we can consider 2 combinations
            #
            # A: $(l_1,v_1)$ and $(l_2,v_2)$
            #
            # B: $(l_1,v_2)$ and $(l_2,v_1)$
            #
            # For each variant we can compute mass of reconstruced W boson and compare with to known physical mass of 80.4. Therefore for each combination following metric is calculated:
            # $$ \varDelta = |M_{W1} - 80.4| + |M_{W2} - 80.4| ,$$
            # where $M_{Wi} = \sqrt{(E_1 + E_2)^2 + (p_{x1} + p_{x2})^2 + (p_{y1} + p_{y2})^2 + (p_{z1} + p_{z2})^2}$ . Indexes refer to the first and second particle in the lepton-neutroni pair.
            #
            # Then we select combination with lesser delta.
            if not self.mass_matching:
                def get_mass (E1, x1, y1, z1, E2, x2, y2, z2):
                    return np.sqrt((E1 + E2)**2 + (x1 + x2)**2 + (y1 + y2)**2 + (z1 + z2)**2)

                W_mass = 80.4

                # A
                A_mass1 = get_mass(df['p_l_1_E'],      df['p_l_1_x'],      df['p_l_1_y'],      df['p_l_1_z'],
                                   df['Neutrino_1.E'], df['Neutrino_1.x'], df['Neutrino_1.y'], df['Neutrino_1.z'])
                A_mass2 = get_mass(df['p_l_2_E'],      df['p_l_2_x'],      df['p_l_2_y'],      df['p_l_2_z'],
                                   df['Neutrino_2.E'], df['Neutrino_2.x'], df['Neutrino_2.y'], df['Neutrino_2.z'])
                A_mass1

                df['Delta1'] = np.abs(A_mass1 - W_mass) + np.abs(A_mass2 - W_mass)

                # B
                B_mass1 = get_mass(df['p_l_1_E'],      df['p_l_1_x'],      df['p_l_1_y'],      df['p_l_1_z'],
                                   df['Neutrino_2.E'], df['Neutrino_2.x'], df['Neutrino_2.y'], df['Neutrino_2.z'])
                B_mass2 = get_mass(df['p_l_2_E'],      df['p_l_2_x'],      df['p_l_2_y'],      df['p_l_2_z'],
                                   df['Neutrino_1.E'], df['Neutrino_1.x'], df['Neutrino_1.y'], df['Neutrino_1.z'])
                B_mass1

                df['Delta2'] = np.abs(B_mass1 - W_mass) + np.abs(B_mass2 - W_mass)

                condition = df["Delta1"] < df["Delta2"]

                df["p_v_1_E"] = df["Neutrino_1.E"].where(condition, df["Neutrino_2.E"])
                df["p_v_1_x"] = df["Neutrino_1.x"].where(condition, df["Neutrino_2.x"])
                df["p_v_1_y"] = df["Neutrino_1.y"].where(condition, df["Neutrino_2.y"])
                df["p_v_1_z"] = df["Neutrino_1.z"].where(condition, df["Neutrino_2.z"])
                df["p_v_2_E"] = df["Neutrino_2.E"].where(condition, df["Neutrino_1.E"])
                df["p_v_2_x"] = df["Neutrino_2.x"].where(condition, df["Neutrino_1.x"])
                df["p_v_2_y"] = df["Neutrino_2.y"].where(condition, df["Neutrino_1.y"])
                df["p_v_2_z"] = df["Neutrino_2.z"].where(condition, df["Neutrino_1.z"])

                df = df.drop(columns=['Neutrino_1.E', 'Neutrino_1.x', 'Neutrino_1.y', 'Neutrino_1.z', 'Neutrino_2.E', 'Neutrino_2.x', 'Neutrino_2.y', 'Neutrino_2.z', 'Delta1', 'Delta2'])
            else:
                df["p_v_1_E"] = df["Neutrino_1.E"]
                df["p_v_1_x"] = df["Neutrino_1.x"]
                df["p_v_1_y"] = df["Neutrino_1.y"]
                df["p_v_1_z"] = df["Neutrino_1.z"]
                df["p_v_2_E"] = df["Neutrino_2.E"]
                df["p_v_2_x"] = df["Neutrino_2.x"]
                df["p_v_2_y"] = df["Neutrino_2.y"]
                df["p_v_2_z"] = df["Neutrino_2.z"]

                df = df.drop(columns=['Neutrino_1.E', 'Neutrino_1.x', 'Neutrino_1.y', 'Neutrino_1.z', 'Neutrino_2.E', 'Neutrino_2.x', 'Neutrino_2.y', 'Neutrino_2.z'])

            df.to_csv(self.data_path, index=False)

    def load_data(self):
        # raw_data_path = self.data_path
        self.data = pd.read_csv(self.data_path)
        # Only take first 50000 rows
        # self.data = self.data.head(50000)

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

        self.types = self.data['Event.Type'].copy()

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

    def create_split(self, return_numpy=True):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        print("Here!")
        if return_numpy:
            return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test

    def p_T(self, lep_x, lep_y):
        return np.sqrt(lep_x**2 + lep_y**2)

    def p(self, lep_x, lep_y, lep_z):
        return np.sqrt(lep_x**2 + lep_y**2 + lep_z**2)

    def eta(self, p, pz):
        return np.abs(1/2 * np.log((p + pz)/(p - pz)))

    def apply_selection_cuts(self, use_event_type=False):
        self.X = self.X.assign(lep0_pT = self.p_T(self.X['p_l_1_x'], self.X['p_l_1_y']))
        self.X = self.X.assign(lep1_pT = self.p_T(self.X['p_l_2_x'], self.X['p_l_2_y']))
        self.X = self.X.assign(lep0_p = self.p(self.X['p_l_1_x'], self.X['p_l_1_y'], self.X['p_l_1_z']))
        self.X = self.X.assign(lep1_p = self.p(self.X['p_l_2_x'], self.X['p_l_2_y'], self.X['p_l_2_z']))
        self.X = self.X.assign(lep0_eta = self.eta(self.X['lep0_p'], self.X['p_l_1_z']))
        self.X = self.X.assign(lep1_eta = self.eta(self.X['lep1_p'], self.X['p_l_2_z']))

        # Either use Event.Type to differentiate between electron and muon
        if use_event_type:
            # Common cut
            mask = (self.X.lep0_pT > 22.0) & (self.X.lep1_pT > 10.0)

            # Conditional cuts based on Event.Type
            type1_mask = (self.X['Event.Type'] == 1) & (self.X.lep0_eta < 2.47) & (self.X.lep1_eta < 2.5)
            type2_mask = (self.X['Event.Type'] == 2) & (self.X.lep0_eta < 2.5) & (self.X.lep1_eta < 2.47)

            full_mask = mask & (type1_mask | type2_mask)

            self.X = self.X[full_mask].copy()
        # Or do not and use 2.5
        else:
            self.X = self.X[(self.X.lep0_pT > 22.0) & (self.X.lep1_pT > 10.0) & (self.X.lep0_eta < 2.5) & (self.X.lep1_eta < 2.5)]

        self.y = self.y.loc[self.X.index]

        self.X = self.X.drop(columns=['lep0_pT', 'lep1_pT', 'lep0_p', 'lep1_p', 'lep0_eta', 'lep1_eta'])

    def run_preprocessing(self,return_numpy=True) -> tuple:
        self.extract_data()
        self.load_data()
        self.process_features()
        self.process_targets()
        #self.scale_data()
        if self.cuts:
            self.apply_selection_cuts()
        if self.processed_features_path and self.processed_targets_path:
            self.save_data()
        if self.splits:
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_split(return_numpy)
            return X_train, X_val, X_test, y_train, y_val, y_test, self.types
        else:
            if return_numpy:
                return self.X.to_numpy(), self.y.to_numpy(), self.types
            else:
                return self.X, self.y, self.types