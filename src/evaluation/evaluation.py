from src.evaluation.bell_test import I_3
from src.utils.lorentz_vector import LorentzVector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
from datetime import datetime
import os
from scipy import stats
from matplotlib.colors import LogNorm
import pandas as pd


class calculate_results:
    def __init__(self, arrays, labels, title, types):
        self.reconstructions = {
            label: (array, t) for label, array, t in zip(labels, arrays, types)
        }
        self.title = title

    def calculate_gellmann_coefficients(self, array, type):
        """
        Calculates the Gell-Mann coefficients for each event in the reconstruction array
        """
        num_samples = len(array)
        lep1 = array[:, :4]
        lep2 = array[:, 4:8]
        neutrino1 = array[:, 8:12]
        neutrino2 = array[:, 12:]

        # Convert to LorentzVector objects
        lep1 = [LorentzVector(lep1[i], type="four-vector") for i in range(num_samples)]
        lep2 = [LorentzVector(lep2[i], type="four-vector") for i in range(num_samples)]
        neutrino1 = [
            LorentzVector(neutrino1[i], type="four-vector") for i in range(num_samples)
        ]
        neutrino2 = [
            LorentzVector(neutrino2[i], type="four-vector") for i in range(num_samples)
        ]

        pW1 = np.zeros((num_samples, 8))
        pW2 = np.zeros((num_samples, 8))
        cov = np.zeros((num_samples, 8, 8))
        cov_sym = np.zeros((num_samples, 8, 8))
        angles_ = np.zeros((num_samples, 4))

        for i in range(num_samples):
            if type is None:
                I_3_obj = I_3(lep1[i], lep2[i], neutrino1[i], neutrino2[i])
            else:
                I_3_obj = I_3(
                    lep1[i], lep2[i], neutrino1[i], neutrino2[i], type.iloc[i]
                )
            pW1_event, pW2_event, cov_event, cov_sym_event, angles = I_3_obj.analysis()
            pW1[i] = pW1_event
            pW2[i] = pW2_event
            cov[i] = cov_event
            cov_sym[i] = cov_sym_event
            angles_[i] = angles

        cov_2d = cov
        cov_sym_2d = cov_sym

        return pW1, pW2, cov_2d, cov_sym_2d, angles_

    def initialize_datasets(self):
        """
        Initializes datasets by calculating Gell-Mann coefficients for non-empty reconstructions
        """
        self.datasets = []
        self.labels = []
        self.colors = []
        self.pW1 = []
        self.pW2 = []
        self.angles_ = []

        color_map = plt.get_cmap("Set1")

        for idx, (label, (array, t)) in enumerate(self.reconstructions.items()):
            pW1_num, pW2_num, cov_2d, cov_sym_2d, angles = (
                self.calculate_gellmann_coefficients(array, t)
            )
            self.datasets.append(cov_2d)
            self.pW1.append(pW1_num)
            self.pW2.append(pW2_num)
            self.angles_.append(angles)
            self.labels.append(label)
            self.colors.append(color_map(idx))

            # Concatenate array, type, pw1, pw2 and write to a csv file for each loop iteration
            # data = np.concatenate(
            #     (array, t.values.reshape(-1, 1), pW1_num, pW2_num), axis=1
            # )
            # df = pd.DataFrame(
            #     data,
            #     columns=[
            #         "p_l_1_E_truth",
            #         "p_l_1_x_truth",
            #         "p_l_1_y_truth",
            #         "p_l_1_z_truth",
            #         "p_l_2_E_truth",
            #         "p_l_2_x_truth",
            #         "p_l_2_y_truth",
            #         "p_l_2_z_truth",
            #         "p_v_1_E_truth",
            #         "p_v_1_x_truth",
            #         "p_v_1_y_truth",
            #         "p_v_1_z_truth",
            #         "p_v_2_E_truth",
            #         "p_v_2_x_truth",
            #         "p_v_2_y_truth",
            #         "p_v_2_z_truth",
            #         "type",
            #         "pWFirst1",
            #         "pWFirst2",
            #         "pWFirst3",
            #         "pWFirst4",
            #         "pWFirst5",
            #         "pWFirst6",
            #         "pWFirst7",
            #         "pWFirst8",
            #         "pWSecond1",
            #         "pWSecond2",
            #         "pWSecond3",
            #         "pWSecond4",
            #         "pWSecond5",
            #         "pWSecond6",
            #         "pWSecond7",
            #         "pWSecond8",
            #     ],
            # )
            # df.to_csv(f"{label}_data.csv", index=False)

    def plot_gellmann_coefficients(self, target_path):
        """
        Plots the Gellmann coefficients in a 8x8 grid
        """

        os.makedirs(target_path, exist_ok=True)

        for cov_2d, label in zip(self.datasets, self.labels):
            cov_mean = cov_2d.mean(axis=0) / 4

            # vmin = cov_mean.min()
            # vmax = cov_mean.max()
            vmin = -0.25
            vmax = 0.25

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            plt.figure(figsize=(8, 6))
            plt.imshow(cov_mean, cmap="seismic", norm=norm, interpolation="nearest")
            plt.colorbar()

            plt.title(f"Coefficients $c_{{ij}}$ - {self.title}", fontsize=16)
            plt.xlabel(r"$W^+$ Index $j$", fontsize=14)
            plt.ylabel(r"$W^-$ Index $i$", fontsize=14)
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # Create a unique identifier based on current date and time and add the label
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coefficients_plot_{label.lower()}_{current_datetime}.png"

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
                    hist_data[label][(i, j)] = np.histogram(
                        reshaped_cov[:, i, j], bins=50, density=True, range=(-10, 10)
                    )

        plt.figure(figsize=(20, 20))
        plt.suptitle(f"{self.title}", fontsize=30)
        for i in range(8):
            for j in range(8):
                plt.subplot(8, 8, i * 8 + j + 1)
                for label, color in zip(self.labels, self.colors):
                    hist, bins = hist_data[label][(i, j)]
                    plt.step(
                        bins[:-1],
                        hist,
                        label=label,
                        color=color,
                        linewidth=1.2,
                        alpha=0.8,
                    )
                    # Save the histogram data and bin edges for further analysis
                    data_and_edges = {"hist": hist, "bins": bins}
                    # np.save(f'{target_path}histogram_data_{label}_{i}_{j}.npy', data_and_edges)

                plt.ylim(0, plt.ylim()[1] * 1.1)

                if (i, j) in [
                    (0, 2),
                    (1, 1),
                    (2, 0),
                    (4, 3),
                    (3, 4),
                    (5, 2),
                    (7, 5),
                    (6, 6),
                    (5, 7),
                ]:
                    ax = plt.gca()
                    for spine in ax.spines.values():
                        spine.set_color("red")
                        spine.set_linewidth(4)

                plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                plt.gca().ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

        plt.figtext(0.5, 0.01, r"$W^+$ Index $j$", ha="center", fontsize=22)
        plt.figtext(
            0.01, 0.5, r"$W^-$ Index $i$", va="center", rotation="vertical", fontsize=22
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles, labels, loc="upper right", fontsize=22)
        plt.tight_layout(rect=[0.03, 0.03, 0.90, 0.97])

        # Create unique identifier based on current date and time
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coefficients_hist_{current_datetime}.png"

        os.makedirs(target_path, exist_ok=True)
        file_path = os.path.join(target_path, filename)

        plt.savefig(file_path)
        plt.close()
        print(f"Histogram plot saved to {file_path}")

    def run(self, target_path):
        self.initialize_datasets()
        self.plot_gellmann_coefficients(target_path)
        # self.plot_gellmann_coefficients_hist(target_path)
        # self.plot_wigner_heatmap(target_path)
        # self.plot_wigner_heatmap_diff(target_path)


class calculate_results_diff_analysis(calculate_results):
    def __init__(self, arrays, labels, title, types):
        super().__init__(arrays, labels, title, types)
        self.reconstructions = {
            label: (array, t) for label, array, t in zip(labels, arrays, types)
        }
        self.title = title

    def plot_2d_angle_hist(self, target_path: str, bins: int = 50):
        N_cos = 12
        N_phi = 20
        os.makedirs(target_path, exist_ok=True)

        cos_edges = np.linspace(-1, 1, N_cos + 1)
        phi_edges = np.linspace(-np.pi, np.pi, N_phi + 1)

        H_Wminus, *_ = np.histogram2d(
            self.angles_[0][:, 2], self.angles_[0][:, 0], bins=[cos_edges, phi_edges]
        )

        H_Wplus, *_ = np.histogram2d(
            self.angles_[0][:, 3], self.angles_[0][:, 1], bins=[cos_edges, phi_edges]
        )

        H_Wminus_cuts, *_ = np.histogram2d(
            self.angles_[1][:, 2], self.angles_[1][:, 0], bins=[cos_edges, phi_edges]
        )

        H_Wplus_cuts, *_ = np.histogram2d(
            self.angles_[1][:, 3], self.angles_[1][:, 1], bins=[cos_edges, phi_edges]
        )

        print("Minimum count after cuts for W-:", H_Wminus_cuts.min())
        print("Minimum count after cuts for W+:", H_Wplus_cuts.min())

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle(f"{self.title}", fontsize=30)
        axs[0, 0].imshow(
            H_Wminus,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[0, 0].set_title(r"$W^-$ before cuts", fontsize=20)
        axs[0, 1].imshow(
            H_Wplus,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[0, 1].set_title(r"$W^+$ before cuts", fontsize=20)
        axs[1, 0].imshow(
            H_Wminus_cuts,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[1, 0].set_title(r"$W^-$ after cuts", fontsize=20)
        axs[1, 1].imshow(
            H_Wplus_cuts,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[1, 1].set_title(r"$W^+$ after cuts", fontsize=20)
        for ax in axs.flat:
            ax.set_xlabel(r"$\phi$", fontsize=20)
            ax.set_ylabel(r"$\cos(\theta)$", fontsize=20)
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        plt.colorbar(
            axs[0, 0].images[0],
            ax=axs[:, 0],
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        plt.colorbar(
            axs[0, 1].images[0],
            ax=axs[:, 1],
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        plt.colorbar(
            axs[1, 0].images[0],
            ax=axs[:, 0],
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        plt.colorbar(
            axs[1, 1].images[0],
            ax=axs[:, 1],
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"2d_angle_hist_{stamp}.png".replace(" ", "_")
        plt.savefig(os.path.join(target_path, fname), dpi=150)
        plt.close(fig)

        # Substract before cuts from after cuts
        H_Wminus_diff = H_Wminus_cuts - H_Wminus
        H_Wplus_diff = H_Wplus_cuts - H_Wplus
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f"{self.title}", fontsize=30)
        axs[0].imshow(
            H_Wminus_diff,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[0].set_title(r"$W^-$ difference", fontsize=20)
        axs[1].imshow(
            H_Wplus_diff,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[1].set_title(r"$W^+$ difference", fontsize=20)
        for ax in axs.flat:
            ax.set_xlabel(r"$\phi$", fontsize=20)
            ax.set_ylabel(r"$\cos(\theta)$", fontsize=20)
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        plt.colorbar(
            axs[0].images[0],
            ax=axs[0],
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        plt.colorbar(
            axs[1].images[0],
            ax=axs[1],
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"2d_angle_hist_{stamp}.png".replace(" ", "_")
        plt.savefig(os.path.join(target_path, fname), dpi=150)
        plt.close(fig)

    def wignerP(self, k, sign, sample):
        cos_theta = (
            self.angles_[sample][:, 2] if sign == -1 else self.angles_[sample][:, 3]
        )
        phi = self.angles_[sample][:, 0] if sign == -1 else self.angles_[sample][:, 1]

        theta = np.arccos(cos_theta)
        s, c = np.sin(theta), np.cos(theta)

        if k == 1:
            return np.sqrt(2) * (5 * c + sign) * s * np.cos(phi)
        if k == 2:
            return np.sqrt(2) * (5 * c + sign) * s * np.sin(phi)
        if k == 3:
            return 0.25 * (sign * 4 * c + 15 * np.cos(2 * theta) + 5)
        if k == 4:
            return 5 * s**2 * np.cos(2 * phi)
        if k == 5:
            return 5 * s**2 * np.sin(2 * phi)
        if k == 6:
            return np.sqrt(2) * (sign - 5 * c) * s * np.cos(phi)
        if k == 7:
            return np.sqrt(2) * (sign - 5 * c) * s * np.sin(phi)
        if k == 8:
            return (1 / (4 * np.sqrt(3))) * (sign * 12 * c - 15 * np.cos(2 * theta) - 5)
        raise ValueError

    def plot_wignerP_1d_hist(self, target_path, bins=50):
        os.makedirs(target_path, exist_ok=True)

        for k in range(1, 9):
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.suptitle(f"{self.title}: Φ$_{{{k}}}$", fontsize=18)

            for sample, lbl, col, sign in (
                (0, "truth  W⁻", "tab:blue", -1),
                (1, "selected W⁻", "tab:cyan", -1),
                (0, "truth  W⁺", "tab:red", +1),
                (1, "selected W⁺", "tab:orange", +1),
            ):
                data = self.wignerP(k, sign, sample)
                data = data[np.isfinite(data)]
                hist, edges = np.histogram(data, bins=bins, density=True)
                centres = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centres, hist, where="mid", label=lbl, color=col)

            ax.set_xlabel("Φ value")
            ax.set_ylabel("normalised counts")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            fname = f"wignerP_hist_k{k}_{datetime.now():%Y%m%d_%H%M%S}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(target_path, fname), dpi=150)
            plt.close(fig)

    def bell_inequality_batch(self, target_path, bins=50):
        os.makedirs(target_path, exist_ok=True)

        # Create list to store all density matrices for batch processing
        density_matrices = []

        for i in range(2):  # For each dataset (truth/reconstruction)
            for j in range(len(self.pW1[i])):  # For each event
                # Create 9x9 density matrix for this event
                mat = np.zeros((9, 9))

                # Fill covariance terms (indices 1-8, 1-8)
                for k in range(1, 9):
                    for m in range(1, 9):
                        mat[k, m] = self.datasets[i][j][k - 1, m - 1] / 4.0

                # Fill pW1 terms (row 0, columns 1-8)
                for k in range(1, 9):
                    mat[0, k] = self.pW2[i][j][k - 1] / 2.0

                # Fill pW2 terms (column 0, rows 1-8)
                for k in range(1, 9):
                    mat[k, 0] = self.pW1[i][j][k - 1] / 2.0

                density_matrices.append(mat)

        # Now pass the list of 2D matrices to the batch function
        bell_values = I_3.CGLMP_test_batch(density_matrices)

        # Plot the bell values for both datasets
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle(f"{self.title}: Bell Inequality", fontsize=18)

        dataset_bell_values = []
        for i, label in enumerate(self.labels):
            # Extract bell values for this dataset
            bell_values_dataset = bell_values[
                i * len(self.pW1[0]) : (i + 1) * len(self.pW1[0])
            ]
            dataset_bell_values.append(bell_values_dataset)
            ax.hist(
                bell_values_dataset,
                bins=bins,
                label=label,
                alpha=0.5,
                density=True,
            )

        ax.set_xlabel("Bell Value")
        ax.set_ylabel("Normalised Counts")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"bell_inequality_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(os.path.join(target_path, fname), dpi=150)
        plt.close(fig)

        # Print statistics for each dataset
        for i, (label, bell_vals) in enumerate(zip(self.labels, dataset_bell_values)):
            print(f"Mean Bell inequality for {label}: {np.mean(bell_vals):.6f}")

            # Most probable value
            hist, bin_edges = np.histogram(bell_vals, bins=bins, density=True)
            most_probable_bin = np.argmax(hist)
            most_probable_value = 0.5 * (
                bin_edges[most_probable_bin] + bin_edges[most_probable_bin + 1]
            )
            print(f"Most probable Bell value for {label}: {most_probable_value:.6f}")

        return bell_values

    def bell_inequality_averaged(self, target_path):
        """
        Calculate Bell inequality values using averaged density matrices.
        First averages cov, pW1, pW2 over all events, then creates one density matrix
        per dataset and calculates one Bell value per dataset.
        """
        os.makedirs(target_path, exist_ok=True)

        averaged_bell_values = []

        print(len(self.datasets[0]))
        print(len(self.pW1[0]))
        print(len(self.pW2[0]))

        print(len(self.datasets[1]))
        print(len(self.pW1[1]))
        print(len(self.pW2[1]))

        for i in range(len(self.labels)):  # For each dataset (truth/reconstruction)
            # Average over all events for this dataset
            avg_cov = np.mean(self.datasets[i], axis=0)  # Average covariance matrix
            avg_pW1 = np.mean(self.pW1[i], axis=0)  # Average pW1 coefficients
            avg_pW2 = np.mean(self.pW2[i], axis=0)  # Average pW2 coefficients

            # Create averaged 9x9 density matrix
            avg_mat = np.zeros((9, 9))

            # Fill covariance terms (indices 1-8, 1-8)
            for k in range(1, 9):
                for m in range(1, 9):
                    avg_mat[k, m] = avg_cov[k - 1, m - 1] / 4.0

            # Fill pW1 terms (row 0, columns 1-8)
            for k in range(1, 9):
                avg_mat[0, k] = avg_pW2[k - 1] / 2.0

            # Fill pW2 terms (column 0, rows 1-8)
            for k in range(1, 9):
                avg_mat[k, 0] = avg_pW1[k - 1] / 2.0

            # Calculate single Bell value for this averaged matrix
            bell_value = I_3.CGLMP_test(avg_mat)
            averaged_bell_values.append(bell_value)

            print(f"Averaged Bell inequality for {self.labels[i]}: {bell_value:.6f}")

        return averaged_bell_values

    def run(self, target_path):
        self.initialize_datasets()
        self.plot_2d_angle_hist(target_path)
        self.plot_wignerP_1d_hist(target_path)
        self.bell_inequality_batch(target_path)
        self.bell_inequality_averaged(target_path)
