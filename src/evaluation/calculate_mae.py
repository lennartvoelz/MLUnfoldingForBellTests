import numpy as np
import os

class results:
    def __init__(self, truth, pred):
        self.truth = truth
        self.pred = pred
        self.mae = self.calculate_mae()
    
    def calculate_mae_componentwise(self):
        return np.mean(np.abs(self.truth - self.pred), axis=0)
    
    def calculate_mae(self):
        return np.mean(self.calculate_mae_componentwise())
    
    def run(self, path):
        print("Mean Absolute Error Componentwise: ", self.calculate_mae_componentwise())
        print("Mean Absolute Error: ", self.mae)
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, "mae.txt")
        np.savetxt(full_path, [self.mae])