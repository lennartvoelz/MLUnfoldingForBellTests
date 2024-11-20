from src.evaluation.bell_test import I_3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
import os

class calculate_results():
    def __init__(self, truth=None, analytical_reconstruction=None, neural_network_reconstruction=None):
        self.analytical_reconstruction = analytical_reconstruction
        self.neural_network_reconstruction = neural_network_reconstruction
        self.truth = truth
        self.num_samples = len(truth)

    def calculate_gellmann_coefficients(self, array):
        """
        Calculates the Gell-Mann coefficients for each event in the reconstruction array
        """
        num_samples = self.num_samples

        lep1 = array[:,0]
        lep2 = array[:,1]
        neutrino1 = array[:,2]
        neutrino2 = array[:,3]
        print(lep1.shape)

        pW1 = np.zeros((num_samples, 8))
        pW2 = np.zeros((num_samples, 8))
        cov = np.zeros((num_samples, 8, 8))
        cov_sym = np.zeros((num_samples, 8, 8))

        for i in range(num_samples):
            I_3_obj = I_3(
            lep1[i],
            lep2[i],
            neutrino1[i],
            neutrino2[i])
            pW1_event, pW2_event, cov_event, cov_sym_event = I_3_obj.analysis()
            pW1[i] = pW1_event
            pW2[i] = pW2_event
            cov[i] = cov_event
            cov_sym[i] = cov_sym_event

        cov_2d = cov.reshape(num_samples, -1)
        cov_sym_2d = cov_sym.reshape(num_samples, -1)

        return pW1, pW2, cov_2d, cov_sym_2d
    
    def initialize_datasets(self):
        """
        Initializes datasets by calculating Gell-Mann coefficients for non-empty reconstructions
        """
        self.datasets = []
        self.labels = []
        self.colors = []
        
        if self.analytical_reconstruction is not None:
            pW1_a, pW2_a, cov_2d_a, cov_sym_2d_a = self.calculate_gellmann_coefficients(self.analytical_reconstruction)
            self.datasets.append(cov_2d_a)
            self.labels.append('Analytical Reconstruction')
            self.colors.append('blue')
        
        if self.truth is not None:
            pW1_t, pW2_t, cov_2d_t, cov_sym_2d_t = self.calculate_gellmann_coefficients(self.truth)
            self.datasets.append(cov_2d_t)
            self.labels.append('Truth')
            self.colors.append('green')
        
        if self.neural_network_reconstruction is not None:
            pW1_n, pW2_n, cov_2d_n, cov_sym_2d_n = self.calculate_gellmann_coefficients(self.neural_network_reconstruction)
            self.datasets.append(cov_2d_n)
            self.labels.append('Neural Network Reconstruction')
            self.colors.append('red')
        
        if not self.datasets:
            print("No datasets available for processing.")
    
    def plot_gellmann_coefficients(self, target_path):
        """
        Plots the Gellmann coefficients in a 8x8 grid
        """
        num_samples = self.num_samples

        os.makedirs(target_path, exist_ok=True)

        for cov_2d, label in zip(self.datasets, self.labels):
            cov = cov_2d.reshape(num_samples, 8, 8)

            cov_mean = cov.mean(axis=0)

            plt.figure(figsize=(8, 6))
            plt.imshow(cov_mean, cmap='seismic', interpolation='nearest')
            plt.colorbar()
            plt.title(f"Coefficients $c_{{ij}}$ - {label}")
            plt.xlabel('Index $j$')
            plt.ylabel('Index $i$')
            plt.tight_layout()

            # Create a unique identifier based on current date and time and add the label
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'coefficients_plot_{label.lower()}_{current_datetime}.png'

            file_path = os.path.join(target_path, filename)

            plt.savefig(file_path)
            plt.close()

            print(f"Plot saved to {file_path}")

    def plot_gellmann_coefficients_hist(self, target_path):
        """
        Plots the histograms of the Gell-Mann coefficients in a 8x8 grid
        """
        num_samples = self.num_samples
        reshaped_covs = [cov_2d.reshape(num_samples, 8, 8) for cov_2d in self.datasets]

        plt.figure(figsize=(20, 20))
        for i in range(8):
            for j in range(8):
                plt.subplot(8, 8, i*8 + j + 1)
                for reshaped_cov, label, color in zip(reshaped_covs, self.labels, self.colors):
                    plt.hist(reshaped_cov[:, i, j], bins=25, alpha=0.5, label=label, color=color)
                
                plt.ylim(0, plt.ylim()[1]*1.1)
                
                if (i, j) in [(0, 5), (1, 6), (2, 7), (2, 2), (3, 3), (4, 4), (5, 0), (6, 1), (7, 2)]:
                    ax = plt.gca()
                    for spine in ax.spines.values():
                        spine.set_color('red')
                        spine.set_linewidth(4)
                
                plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                plt.gca().ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
                if i == 0 and j == 0:
                    plt.legend(fontsize=8)
    
        plt.tight_layout()

        # Create unique identifier based on current date and time
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'coefficients_hist_{current_datetime}.png'

        os.makedirs(target_path, exist_ok=True)
        file_path = os.path.join(target_path, filename)

        plt.savefig(file_path)
        plt.close()
        print(f"Histogram plot saved to {file_path}")