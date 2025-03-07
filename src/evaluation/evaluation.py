from src.evaluation.bell_test import I_3
from src.utils.lorentz_vector import LorentzVector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
from datetime import datetime
import os

class calculate_results():
    def __init__(self, arrays, labels, title):
        self.reconstructions = {label: array for label, array in zip(labels, arrays)}
        self.title = title

    def calculate_gellmann_coefficients(self, array):
        """
        Calculates the Gell-Mann coefficients for each event in the reconstruction array
        """
        num_samples = len(array)
        lep1 = array[:,:4]
        lep2 = array[:,4:8]
        neutrino1 = array[:,8:12]
        neutrino2 = array[:,12:]
        
        # Convert to LorentzVector objects
        lep1 = [LorentzVector(lep1[i], type="four-vector") for i in range(num_samples)]
        lep2 = [LorentzVector(lep2[i], type="four-vector") for i in range(num_samples)]
        neutrino1 = [LorentzVector(neutrino1[i], type="four-vector") for i in range(num_samples)]
        neutrino2 = [LorentzVector(neutrino2[i], type="four-vector") for i in range(num_samples)]

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

        cov_2d = cov
        cov_sym_2d = cov_sym

        return pW1, pW2, cov_2d, cov_sym_2d
    
    def initialize_datasets(self):
        """
        Initializes datasets by calculating Gell-Mann coefficients for non-empty reconstructions
        """
        self.datasets = []
        self.labels = []
        self.colors = []

        color_map = plt.get_cmap('Set1')
        
        for idx, (label, array) in enumerate(self.reconstructions.items()):
            pW1, pW2, cov_2d, cov_sym_2d = self.calculate_gellmann_coefficients(array)
            self.datasets.append(cov_2d)
            self.labels.append(label)
            self.colors.append(color_map(idx))
    
    def plot_gellmann_coefficients(self, target_path):
        """
        Plots the Gellmann coefficients in a 8x8 grid
        """

        os.makedirs(target_path, exist_ok=True)

        for cov_2d, label in zip(self.datasets, self.labels):
            cov_mean = cov_2d.mean(axis=0)/4

            vmin = cov_mean.min()
            vmax = cov_mean.max()
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            plt.figure(figsize=(8, 6))
            plt.imshow(cov_mean, cmap='seismic', norm=norm, interpolation='nearest')
            plt.colorbar()
            plt.title(f"Coefficients $c_{{ij}}$ - {self.title}", fontsize=16)
            plt.xlabel(r'$W^+$ Index $j$', fontsize=14)
            plt.ylabel(r'$W^-$ Index $i$', fontsize=14)
            plt.gca().invert_yaxis()
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
        reshaped_covs = [cov_2d.reshape(len(cov_2d), 8, 8) for cov_2d in self.datasets]

        hist_data = {}
        for reshaped_cov, label in zip(reshaped_covs, self.labels):
            hist_data[label] = {}
            for i in range(8):
                for j in range(8):
                    hist_data[label][(i, j)] = np.histogram(reshaped_cov[:, i, j], bins=50, density=True)

        plt.figure(figsize=(20, 20))
        plt.suptitle(f"{self.title}", fontsize=30)
        for i in range(8):
            for j in range(8):
                plt.subplot(8, 8, i*8 + j + 1)
                for label, color in zip(self.labels, self.colors):
                    hist, bins = hist_data[label][(i, j)]
                    plt.step(bins[:-1], hist, label=label, color=color, linewidth=1.2, alpha=0.8)
                    # Save the histogram data and bin edges for further analysis
                    data_and_edges = {'hist': hist, 'bins': bins}
                    # np.save(f'{target_path}histogram_data_{label}_{i}_{j}.npy', data_and_edges)
                    

                plt.ylim(0, plt.ylim()[1]*1.1)

                if (i, j) in [(0, 2), (1, 1), (2, 0), (4, 3), (3, 4), (5, 2), (7, 5), (6, 6), (5, 7)]:
                    ax = plt.gca()
                    for spine in ax.spines.values():
                        spine.set_color('red')
                        spine.set_linewidth(4)

                plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                plt.gca().ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        plt.figtext(0.5, 0.01, r'$W^+$ Index $j$', ha='center', fontsize=22)
        plt.figtext(0.01, 0.5, r'$W^-$ Index $i$', va='center', rotation='vertical', fontsize=22)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='upper right', fontsize=22)
        plt.tight_layout(rect=[0.03, 0.03, 0.90, 0.97])

        # Create unique identifier based on current date and time
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'coefficients_hist_{current_datetime}.png'

        os.makedirs(target_path, exist_ok=True)
        file_path = os.path.join(target_path, filename)

        plt.savefig(file_path)
        plt.close()
        print(f"Histogram plot saved to {file_path}")

    def run(self, target_path):
        self.initialize_datasets()
        self.plot_gellmann_coefficients(target_path)
        self.plot_gellmann_coefficients_hist(target_path)