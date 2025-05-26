from src.evaluation.bell_test import I_3
from src.utils.lorentz_vector import LorentzVector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
from datetime import datetime
import os
from scipy import stats
# import dcor
from sklearn.feature_selection import mutual_info_regression

class calculate_results():
    def __init__(self, arrays, labels, title, types=None):
        self.reconstructions = {label: array for label, array in zip(labels, arrays)}
        self.title = title
        self.types = types


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
            if self.types is None:
                I_3_obj = I_3(
                    lep1[i],
                    lep2[i],
                    neutrino1[i],
                    neutrino2[i]
                )
            else:
                I_3_obj = I_3(
                    lep1[i],
                    lep2[i],
                    neutrino1[i],
                    neutrino2[i],
                    self.types.iloc[i]
                )
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
        self.pW1 = []
        self.pW2 = []

        color_map = plt.get_cmap('Set1')

        for idx, (label, array) in enumerate(self.reconstructions.items()):
            pW1_num, pW2_num, cov_2d, cov_sym_2d = self.calculate_gellmann_coefficients(array)
            self.datasets.append(cov_2d)
            self.pW1.append(pW1_num)
            self.pW2.append(pW2_num)
            self.labels.append(label)
            self.colors.append(color_map(idx))

    def plot_gellmann_coefficients(self, target_path):
        """
        Plots the Gellmann coefficients in a 8x8 grid
        """

        os.makedirs(target_path, exist_ok=True)

        for cov_2d, label in zip(self.datasets, self.labels):
            cov_mean = cov_2d.mean(axis=0)/4
            # Instead of taking the mean, take a trimmed mean to reduce the influence of outliers
            # cov_mean = trim_mean(cov_2d, 0.15, axis=0)/4
            # Instead of the mean, take the mode
            # cov_mean = np.zeros((8, 8))
            # for i in range(8):
            #     for j in range(8):
            #         # Bin each coefficient in 50 bins
            #         hist, bins = np.histogram(cov_2d[:, i, j], bins=50)
            #         # Find the bin with the highest frequency
            #         mode_idx = np.argmax(hist)
            #         # Take the middle of the bin as the mode
            #         mode_value = (bins[mode_idx] + bins[mode_idx + 1]) / 2
            #         cov_mean[i, j] = mode_value

            vmin = cov_mean.min()
            vmax = cov_mean.max()
            # vmin = -1.25
            # vmax = 1.25
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            plt.figure(figsize=(8, 6))
            plt.imshow(cov_mean, cmap='seismic', norm=norm, interpolation='nearest')
            plt.colorbar()
            # # Write the value in each bin of the 2D histogram
            # for i in range(cov_mean.shape[0]):
            #     for j in range(cov_mean.shape[1]):
            #         plt.text(j, i, f"{cov_mean[i, j]:.2f}", ha='center', va='center', color='black')
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
                    hist_data[label][(i, j)] = np.histogram(reshaped_cov[:, i, j], bins=50, density=True, range=(-10, 10))

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

    def plot_wigner_scatter(self, target_path):
        """
        Plots the Wigner scatter plot for the Wigner P functions
        """

        os.makedirs(target_path, exist_ok=True)
        pW1 = np.array(self.pW1)
        pW2 = np.array(self.pW2)

        for pW1_num, pW2_num, name in zip(pW1, pW2, self.labels):
            plt.figure(figsize=(20, 20))
            plt.suptitle(f"{name} {self.title}", fontsize=30)

            for i in range(8):
                for j in range(8):
                    plt.subplot(8, 8, i*8 + j + 1)
                    plt.scatter(pW1_num[:, i], pW2_num[:, j], s=0.5)

                    if (i, j) in [(0, 2), (1, 1), (2, 0), (4, 3), (3, 4), (5, 2), (7, 5), (6, 6), (5, 7)]:
                        ax = plt.gca()
                        for spine in ax.spines.values():
                            spine.set_color('red')
                            spine.set_linewidth(4)

                    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    plt.gca().ticklabel_format(style='sci', axis='both', scilimits=(0,0))

            plt.figtext(0.5, 0.01, r'$pW^+_{i}$', ha='center', fontsize=22)
            plt.figtext(0.01, 0.5, r'$pW^-_{j}$', va='center', rotation='vertical', fontsize=22)
            # Create unique identifier based on current date and time
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'wigner_scatter_{current_datetime}.png'
            file_path = os.path.join(target_path, filename)
            plt.savefig(file_path)
            plt.close()

    def calc_correlation(self, target_path):
        """
        Calculates the correlation between the Gell-Mann coefficients
        """
        pW1 = np.array(self.pW1)
        pW2 = np.array(self.pW2)

        os.makedirs(target_path, exist_ok=True)

        for pW1_num, pW2_num, label in zip(pW1, pW2, self.labels):

            pearson_corr_matrix   = np.zeros((8, 8))
            spearman_corr_matrix  = np.zeros((8, 8))
            kendall_corr_matrix   = np.zeros((8, 8))

            for i in range(8):
                for j in range(8):
                    col1 = pW1_num[:, i]
                    col2 = pW2_num[:, j]

                    # Pearson correlation
                    pearson_corr, _ = stats.pearsonr(col1, col2)
                    pearson_corr_matrix[i, j] = pearson_corr

                    # Spearman correlation
                    spearman_corr, _ = stats.spearmanr(col1, col2)
                    spearman_corr_matrix[i, j] = spearman_corr

                    # Kendall correlation
                    kendall_corr, _ = stats.kendalltau(col1, col2)
                    kendall_corr_matrix[i, j] = kendall_corr

            # For each correlation matrix, plot the Gell-Mann coefficients
            correlation_matrices = {
                'Pearson': pearson_corr_matrix,
                'Spearman': spearman_corr_matrix,
                'Kendall': kendall_corr_matrix
            }

            for name, matrix in correlation_matrices.items():
                cov_mean = matrix
                plt.figure(figsize=(8, 6))
                vmin = cov_mean.min()
                vmax = cov_mean.max()
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                plt.imshow(cov_mean, cmap='seismic', norm=norm, interpolation='nearest')
                plt.colorbar()
                plt.title(f"{name} Correlation Coefficients - {self.title}", fontsize=16)
                plt.xlabel(r'$W^+$ Index $j$', fontsize=14)
                plt.ylabel(r'$W^-$ Index $i$', fontsize=14)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                # Create a unique identifier based on current date and time and add the label
                current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'{name.lower()}_correlation_plot_{current_datetime}.png'
                file_path = os.path.join(target_path, filename)
                plt.savefig(file_path)
                plt.close()

    def calc_zscore(self, target_path):
        """
        Calculates the Z-score for each Gell-Mann coefficient
        """
        pW1 = np.array(self.pW1)
        pW2 = np.array(self.pW2)

        os.makedirs(target_path, exist_ok=True)

        plt.figure(figsize=(20, 20))
        plt.suptitle(f"{self.title}", fontsize=30)

        for i in range(8):
            for j in range(8):
                ax = plt.subplot(8, 8, i*8 + j + 1)
                for pW1_num, pW2_num, label, color in zip(pW1, pW2, self.labels, self.colors):
                    mean_pW1 = np.mean(pW1_num, axis=0)
                    std_pW1 = np.std(pW1_num, axis=0)
                    mean_pW2 = np.mean(pW2_num, axis=0)
                    std_pW2 = np.std(pW2_num, axis=0)

                    zscore_pW1 = (pW1_num[:, i] - mean_pW1[i]) / std_pW1[i]
                    zscore_pW2 = (pW2_num[:, j] - mean_pW2[j]) / std_pW2[j]

                    z_prod = zscore_pW1 * zscore_pW2

                    ax.hist(z_prod, bins=50, density=True, alpha=1, color=color, label=label, range=(-5, 5), histtype='step', linewidth=1.2)

                ax.set_ylim(0, ax.get_ylim()[1]*1.1)

                if (i, j) in [(0, 2), (1, 1), (2, 0), (4, 3), (3, 4), (5, 2), (7, 5), (6, 6), (5, 7)]:
                    for spine in ax.spines.values():
                        spine.set_color('red')
                        spine.set_linewidth(4)

            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        plt.figtext(0.5, 0.01, r'$W^+$ Index $j$', ha='center', fontsize=22)
        plt.figtext(0.01, 0.5, r'$W^-$ Index $i$', va='center', rotation='vertical', fontsize=22)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='upper right', fontsize=22)
        plt.tight_layout(rect=[0.03, 0.03, 0.90, 0.97])

        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'zscore_product_plot_{current_datetime}.png'
        file_path = os.path.join(target_path, filename)
        plt.savefig(file_path)
        plt.close()


    def run(self, target_path):
        self.initialize_datasets()
        self.plot_gellmann_coefficients(target_path)
        self.plot_gellmann_coefficients_hist(target_path)
        self.plot_wigner_scatter(target_path)
        self.calc_correlation(target_path)
        self.calc_zscore(target_path)