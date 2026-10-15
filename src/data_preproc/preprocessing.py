import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.change_of_coordinates import exp_to_four_vec


class DataPreprocessor:
    def __init__(
        self,
        raw_data_path,
        data_path,
        processed_features_path=None,
        processed_targets_path=None,
        detector_sim_path=None,
        truth_path=None,
        cuts=True,
        splits=True,
        drop_zeroes=False,
    ):
        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.processed_features_path = processed_features_path
        self.processed_targets_path = processed_targets_path
        self.cuts = cuts
        self.splits = splits
        self.drop_zeroes = drop_zeroes
        self.detector_sim_path = detector_sim_path
        self.truth_path = truth_path
        self.types = None

    def extract_data(self):
        if not os.path.exists(self.data_path + ".csv"):
            # From: https://cernbox.cern.ch/files/link/public/Ju7DYsj0y8sQe2j?tiles-size=1&items-per-page=100&view-mode=resource-table
            df = pd.read_csv(self.raw_data_path + ".csv")

            df["Event.Type"] = df["case"].copy()
            df.drop(columns=["case"], inplace=True)
            if "Particle.Status" in df.columns:
                df.drop(columns=["Particle.Status"], inplace=True)

            # Detector simulation
            df[["Electron.E", "Electron.px", "Electron.py", "Electron.pz"]] = df.apply(
                lambda row: pd.Series(
                    exp_to_four_vec(row["e_pt"], row["e_eta"], row["e_phi"], 0.000511)
                ),
                axis=1,
            )
            df[["Muon.E", "Muon.px", "Muon.py", "Muon.pz"]] = df.apply(
                lambda row: pd.Series(
                    exp_to_four_vec(row["mu_pt"], row["mu_eta"], row["mu_phi"], 0.10566)
                ),
                axis=1,
            )
            # Truth
            df[
                [
                    "Truth.Electron.E",
                    "Truth.Electron.px",
                    "Truth.Electron.py",
                    "Truth.Electron.pz",
                ]
            ] = df.apply(
                lambda row: pd.Series(
                    exp_to_four_vec(
                        row["truth_e_pt"],
                        row["truth_e_eta"],
                        row["truth_e_phi"],
                        0.000511,
                    )
                ),
                axis=1,
            )
            df[["Truth.Muon.E", "Truth.Muon.px", "Truth.Muon.py", "Truth.Muon.pz"]] = (
                df.apply(
                    lambda row: pd.Series(
                        exp_to_four_vec(
                            row["truth_mu_pt"],
                            row["truth_mu_eta"],
                            row["truth_mu_phi"],
                            0.10566,
                        )
                    ),
                    axis=1,
                )
            )
            # Truth neutrinos
            df[
                [
                    "Truth.Neutrino.Electron.E",
                    "Truth.Neutrino.Electron.px",
                    "Truth.Neutrino.Electron.py",
                    "Truth.Neutrino.Electron.pz",
                ]
            ] = df.apply(
                lambda row: pd.Series(
                    exp_to_four_vec(
                        row["truth_ve_pt"],
                        row["truth_ve_eta"],
                        row["truth_ve_phi"],
                        0.000511,
                    )
                ),
                axis=1,
            )
            df[
                [
                    "Truth.Neutrino.Muon.E",
                    "Truth.Neutrino.Muon.px",
                    "Truth.Neutrino.Muon.py",
                    "Truth.Neutrino.Muon.pz",
                ]
            ] = df.apply(
                lambda row: pd.Series(
                    exp_to_four_vec(
                        row["truth_vmu_pt"],
                        row["truth_vmu_eta"],
                        row["truth_vmu_phi"],
                        0.10566,
                    )
                ),
                axis=1,
            )

            # l1 has highest p_T
            # Detector simulation
            df["p_l_1_E"] = df.apply(
                lambda row: row["Electron.E"]
                if row["e_pt"] > row["mu_pt"]
                else row["Muon.E"],
                axis=1,
            )
            df["p_l_1_x"] = df.apply(
                lambda row: row["Electron.px"]
                if row["e_pt"] > row["mu_pt"]
                else row["Muon.px"],
                axis=1,
            )
            df["p_l_1_y"] = df.apply(
                lambda row: row["Electron.py"]
                if row["e_pt"] > row["mu_pt"]
                else row["Muon.py"],
                axis=1,
            )
            df["p_l_1_z"] = df.apply(
                lambda row: row["Electron.pz"]
                if row["e_pt"] > row["mu_pt"]
                else row["Muon.pz"],
                axis=1,
            )
            df["p_l_2_E"] = df.apply(
                lambda row: row["Muon.E"]
                if row["e_pt"] > row["mu_pt"]
                else row["Electron.E"],
                axis=1,
            )
            df["p_l_2_x"] = df.apply(
                lambda row: row["Muon.px"]
                if row["e_pt"] > row["mu_pt"]
                else row["Electron.px"],
                axis=1,
            )
            df["p_l_2_y"] = df.apply(
                lambda row: row["Muon.py"]
                if row["e_pt"] > row["mu_pt"]
                else row["Electron.py"],
                axis=1,
            )
            df["p_l_2_z"] = df.apply(
                lambda row: row["Muon.pz"]
                if row["e_pt"] > row["mu_pt"]
                else row["Electron.pz"],
                axis=1,
            )
            # Truth
            df["p_l_1_E_truth"] = df.apply(
                lambda row: row["Truth.Electron.E"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Muon.E"],
                axis=1,
            )
            df["p_l_1_x_truth"] = df.apply(
                lambda row: row["Truth.Electron.px"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Muon.px"],
                axis=1,
            )
            df["p_l_1_y_truth"] = df.apply(
                lambda row: row["Truth.Electron.py"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Muon.py"],
                axis=1,
            )
            df["p_l_1_z_truth"] = df.apply(
                lambda row: row["Truth.Electron.pz"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Muon.pz"],
                axis=1,
            )
            df["p_l_2_E_truth"] = df.apply(
                lambda row: row["Truth.Muon.E"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Electron.E"],
                axis=1,
            )
            df["p_l_2_x_truth"] = df.apply(
                lambda row: row["Truth.Muon.px"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Electron.px"],
                axis=1,
            )
            df["p_l_2_y_truth"] = df.apply(
                lambda row: row["Truth.Muon.py"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Electron.py"],
                axis=1,
            )
            df["p_l_2_z_truth"] = df.apply(
                lambda row: row["Truth.Muon.pz"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Electron.pz"],
                axis=1,
            )
            # Truth neutrino
            df["p_v_1_E_truth"] = df.apply(
                lambda row: row["Truth.Neutrino.Electron.E"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Neutrino.Muon.E"],
                axis=1,
            )
            df["p_v_1_x_truth"] = df.apply(
                lambda row: row["Truth.Neutrino.Electron.px"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Neutrino.Muon.px"],
                axis=1,
            )
            df["p_v_1_y_truth"] = df.apply(
                lambda row: row["Truth.Neutrino.Electron.py"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Neutrino.Muon.py"],
                axis=1,
            )
            df["p_v_1_z_truth"] = df.apply(
                lambda row: row["Truth.Neutrino.Electron.pz"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Neutrino.Muon.pz"],
                axis=1,
            )
            df["p_v_2_E_truth"] = df.apply(
                lambda row: row["Truth.Neutrino.Muon.E"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Neutrino.Electron.E"],
                axis=1,
            )
            df["p_v_2_x_truth"] = df.apply(
                lambda row: row["Truth.Neutrino.Muon.px"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Neutrino.Electron.px"],
                axis=1,
            )
            df["p_v_2_y_truth"] = df.apply(
                lambda row: row["Truth.Neutrino.Muon.py"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Neutrino.Electron.py"],
                axis=1,
            )
            df["p_v_2_z_truth"] = df.apply(
                lambda row: row["Truth.Neutrino.Muon.pz"]
                if row["truth_e_pt"] > row["truth_mu_pt"]
                else row["Truth.Neutrino.Electron.pz"],
                axis=1,
            )
            df.drop(
                columns=[
                    "Electron.E",
                    "Electron.px",
                    "Electron.py",
                    "Electron.pz",
                    "Muon.E",
                    "Muon.px",
                    "Muon.py",
                    "Muon.pz",
                    "e_pt",
                    "e_eta",
                    "e_phi",
                    "mu_pt",
                    "mu_eta",
                    "mu_phi",
                    "Truth.Electron.E",
                    "Truth.Electron.px",
                    "Truth.Electron.py",
                    "Truth.Electron.pz",
                    "Truth.Muon.E",
                    "Truth.Muon.px",
                    "Truth.Muon.py",
                    "Truth.Muon.pz",
                    "truth_e_pt",
                    "truth_e_eta",
                    "truth_e_phi",
                    "truth_mu_pt",
                    "truth_mu_eta",
                    "truth_mu_phi",
                    "Truth.Neutrino.Electron.E",
                    "Truth.Neutrino.Electron.px",
                    "Truth.Neutrino.Electron.py",
                    "Truth.Neutrino.Electron.pz",
                    "Truth.Neutrino.Muon.E",
                    "Truth.Neutrino.Muon.px",
                    "Truth.Neutrino.Muon.py",
                    "Truth.Neutrino.Muon.pz",
                    "truth_ve_pt",
                    "truth_ve_eta",
                    "truth_ve_phi",
                    "truth_vmu_pt",
                    "truth_vmu_eta",
                    "truth_vmu_phi",
                ],
                inplace=True,
            )

            df["mpx"] = df.apply(
                lambda row: row["met"] * np.cos(row["met_phi"]), axis=1
            )
            df["mpy"] = df.apply(
                lambda row: row["met"] * np.sin(row["met_phi"]), axis=1
            )

            df.drop(columns=["met", "met_eta", "met_phi"], inplace=True)

            df.to_csv(self.data_path + ".csv", index=False)

    def load_data(self):
        self.data = pd.read_csv(self.data_path + ".csv")
        # Only take first 50000 rows
        # self.data = self.data.head(50000)

    def process_features(self):
        # ML dataset
        lep0 = self.data[["p_l_1_E", "p_l_1_x", "p_l_1_y", "p_l_1_z"]]
        lep1 = self.data[["p_l_2_E", "p_l_2_x", "p_l_2_y", "p_l_2_z"]]

        if "mpx" not in self.data.columns:
            mpx = (self.data["p_v_1_x"] + self.data["p_v_2_x"]).to_frame(name="mpx")
            mpy = (self.data["p_v_1_y"] + self.data["p_v_2_y"]).to_frame(name="mpy")
        else:
            mpx = self.data[["mpx"]]
            mpy = self.data[["mpy"]]

        self.X = pd.concat([lep0, lep1, mpx, mpy], axis=1)
        self.X["Event.Type"] = self.data["Event.Type"].copy()

        if self.drop_zeroes:
            # Check detector_sim columns for any zeroes in either of leptons and drop them
            zero_columns = [
                "p_l_1_E",
                "p_l_1_x",
                "p_l_1_y",
                "p_l_1_z",
                "p_l_2_E",
                "p_l_2_x",
                "p_l_2_y",
                "mpx",
                "mpy",
            ]
            self.X = self.X[~(self.X[zero_columns] == 0).any(axis=1)]
            # Drop nan values as well
            self.X = self.X.dropna()

        self.types = self.data["Event.Type"].copy()

        # Detector simulation
        self.detector_sim = self.data[
            [
                "p_l_1_E",
                "p_l_1_x",
                "p_l_1_y",
                "p_l_1_z",
                "p_l_2_E",
                "p_l_2_x",
                "p_l_2_y",
                "p_l_2_z",
                "mpx",
                "mpy",
                "Event.Type",
            ]
        ].copy()

        # Truth
        self.truth = self.data[
            [
                "p_l_1_E_truth",
                "p_l_1_x_truth",
                "p_l_1_y_truth",
                "p_l_1_z_truth",
                "p_l_2_E_truth",
                "p_l_2_x_truth",
                "p_l_2_y_truth",
                "p_l_2_z_truth",
                "p_v_1_E_truth",
                "p_v_1_x_truth",
                "p_v_1_y_truth",
                "p_v_1_z_truth",
                "p_v_2_E_truth",
                "p_v_2_x_truth",
                "p_v_2_y_truth",
                "p_v_2_z_truth",
                "Event.Type",
            ]
        ].copy()

    def process_targets(self):
        y1 = self.data[
            ["p_v_1_E_truth", "p_v_1_x_truth", "p_v_1_y_truth", "p_v_1_z_truth"]
        ]
        y2 = self.data[
            ["p_v_2_E_truth", "p_v_2_x_truth", "p_v_2_y_truth", "p_v_2_z_truth"]
        ]

        self.y = pd.concat([y1, y2], axis=1)

        if self.drop_zeroes:
            # Use the indices of the rows that were not dropped in X
            self.y = self.y.loc[self.X.index]

    def scale_data(self):
        self.X /= 1e3
        self.y /= 1e3

    def save_data(self):
        datasets = (self.X, self.y, self.detector_sim, self.truth)
        paths = (
            self.processed_features_path,
            self.processed_targets_path,
            self.detector_sim_path,
            self.truth_path,
        )

        for dataset, path in zip(datasets, paths):
            if path:
                if self.cuts:
                    path = path + "_cuts"

                dataset.to_csv(path + ".csv", index=False)

    def create_split(self, return_numpy=True):
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        if return_numpy:
            return (
                X_train.to_numpy(),
                X_val.to_numpy(),
                X_test.to_numpy(),
                y_train.to_numpy(),
                y_val.to_numpy(),
                y_test.to_numpy(),
            )
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test

    def p_T(self, lep_x, lep_y):
        return np.sqrt(lep_x**2 + lep_y**2)

    def p(self, lep_x, lep_y, lep_z):
        return np.sqrt(lep_x**2 + lep_y**2 + lep_z**2)

    def eta(self, p, pz):
        return np.abs(1 / 2 * np.log((p + pz) / (p - pz)))

    def apply_selection_cuts(self, use_event_type=True):
        eps = 1e-15

        # helper to compute pT, |eta|
        def kin(px, py, pz):
            pT = np.hypot(px, py)
            p = np.sqrt(pT**2 + pz**2)
            eta = 0.5 * np.log((p + pz + eps) / (p - pz + eps))
            return pT, np.abs(eta)

        # build detector-level mask ONCE using self.detector_sim
        d = self.detector_sim
        d_pT1, d_eta1 = kin(
            d["p_l_1_x"].to_numpy(), d["p_l_1_y"].to_numpy(), d["p_l_1_z"].to_numpy()
        )
        d_pT2, d_eta2 = kin(
            d["p_l_2_x"].to_numpy(), d["p_l_2_y"].to_numpy(), d["p_l_2_z"].to_numpy()
        )
        evt = self.data["Event.Type"].to_numpy()

        if use_event_type:
            is_ee, is_mumu = (evt == 0), (evt == 1)
            det_mask = (
                (d_pT1 > 22.0)
                & (d_pT2 > 15.0)
                & (
                    (is_ee & (d_eta1 < 2.47) & (d_eta2 < 2.5))
                    | (is_mumu & (d_eta1 < 2.5) & (d_eta2 < 2.47))
                )
            )
        else:
            det_mask = (d_pT1 > 22.0) & (d_pT2 > 15.0) & (d_eta1 < 2.5) & (d_eta2 < 2.5)

        # OPTIONAL: also require truth to pass similar cuts (use only if you intend symmetry)
        t = self.truth
        t_pT1, t_eta1 = kin(
            t["p_l_1_x_truth"].to_numpy(),
            t["p_l_1_y_truth"].to_numpy(),
            t["p_l_1_z_truth"].to_numpy(),
        )
        t_pT2, t_eta2 = kin(
            t["p_l_2_x_truth"].to_numpy(),
            t["p_l_2_y_truth"].to_numpy(),
            t["p_l_2_z_truth"].to_numpy(),
        )

        if use_event_type:
            is_ee, is_mumu = (evt == 0), (evt == 1)
            tru_mask = (
                (t_pT1 > 22.0)
                & (t_pT2 > 15.0)
                & (
                    (is_ee & (t_eta1 < 2.47) & (t_eta2 < 2.5))
                    | (is_mumu & (t_eta1 < 2.5) & (t_eta2 < 2.47))
                )
            )
        else:
            tru_mask = (t_pT1 > 22.0) & (t_pT2 > 15.0) & (t_eta1 < 2.5) & (t_eta2 < 2.5)

        # final shared index: intersection of the masks (and any earlier drop_zeroes index)
        idx_keep = self.data.index[
            det_mask & tru_mask
        ]  # or det_mask only, if that’s your policy

        # apply the SAME index to all views
        self.X = self.X.loc[idx_keep].copy()
        self.y = self.y.loc[idx_keep].copy()
        self.detector_sim = self.detector_sim.loc[idx_keep].copy()
        self.truth = self.truth.loc[idx_keep].copy()
        self.types = self.types.loc[idx_keep].copy()

        # sanity
        n = len(idx_keep)
        assert (
            len(self.X)
            == len(self.y)
            == len(self.detector_sim)
            == len(self.truth)
            == len(self.types)
            == n
        )

    def run_preprocessing(self, return_numpy=True) -> tuple:
        self.extract_data()
        self.load_data()
        self.process_features()
        self.process_targets()

        # self.scale_data()
        if self.cuts:
            self.apply_selection_cuts()

        self.save_data()

        if self.splits:
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_split(
                return_numpy
            )
            return X_train, X_val, X_test, y_train, y_val, y_test, self.types
        else:
            if return_numpy:
                return self.X.to_numpy(), self.y.to_numpy(), self.types
            else:
                return self.X, self.y, self.types
