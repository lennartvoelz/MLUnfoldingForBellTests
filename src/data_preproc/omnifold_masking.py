from src.data_preproc.preprocessing import DataPreprocessor

class OmnifoldMasking(DataPreprocessor):
    def __init__(self, data_path, cuts_masking=True, missing_masking=True, splits=True):
        super().__init__(data_path, splits)
        self.cuts_masking = cuts_masking
        self.missing_masking = missing_masking
    
    def mask_selection_cuts(self):
        self.X = self.X.assign(lep0_pT = self.p_T(self.X['p_l_1_x'], self.X['p_l_1_y']))
        self.X = self.X.assign(lep1_pT = self.p_T(self.X['p_l_2_x'], self.X['p_l_2_y']))
        self.X = self.X.assign(lep0_p = self.p(self.X['p_l_1_x'], self.X['p_l_1_y'], self.X['p_l_1_z']))
        self.X = self.X.assign(lep1_p = self.p(self.X['p_l_2_x'], self.X['p_l_2_y'], self.X['p_l_2_z']))
        self.X = self.X.assign(lep0_eta = self.eta(self.X['lep0_p'], self.X['p_l_1_z']))
        self.X = self.X.assign(lep1_eta = self.eta(self.X['lep1_p'], self.X['p_l_2_z']))

        # Add boolean mask to X
        mask = (
            (self.X.lep0_pT > 22.0) &
            (self.X.lep1_pT > 15.0) &
            (self.X.lep0_eta < 2.47) &
            (self.X.lep1_eta < 2.47)
        )

        self.selection_cuts_mask = mask.to_numpy()

        self.X = self.X.drop(columns=['lep0_pT', 'lep1_pT', 'lep0_p', 'lep1_p', 'lep0_eta', 'lep1_eta'])

    def mask_missing_events(self):
        zero_counts = (self.X == 0).sum(axis=1)
        mask = zero_counts > 2
        self.missing_events_mask = mask.to_numpy()

    def run_omnifold_preprocessing(self):
        self.load_data()
        self.process_features()
        self.process_targets()
        if self.cuts_masking:
            self.mask_selection_cuts()
        if self.missing_masking:
            self.mask_missing_events()
        return self.X.to_numpy(), self.y.to_numpy(), self.missing_events_mask, self.selection_cuts_mask