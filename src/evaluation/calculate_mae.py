import numpy as np

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
        print("Mean Absolute Error: ", self.mae)
        np.savetxt(path + "mae.txt", self.mae)