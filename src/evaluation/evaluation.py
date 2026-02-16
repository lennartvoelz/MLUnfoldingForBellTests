from src.evaluation.bell_test import I_3
from src.utils.lorentz_vector import LorentzVector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
from datetime import datetime
import os


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

        cov_avg = np.mean(cov, axis=0)
        mat = cov_avg

        bell = I_3.CGLMP_test(mat)
        bell_values = I_3.CGLMP_test_batch(cov)
        print("Bell Value: ", bell)

        return pW1, pW2, cov_2d, cov_sym_2d, angles_, bell_values

    def plot_bell_hist(self, target_path="./plots/"):
        for label, bell_values in zip(self.labels, self.bell_values):
            plt.hist(
                bell_values,
                bins=50,
                alpha=0.5,
                label=label,
                density=True,
                range=[-30, 30],
            )
        plt.legend()
        plt.savefig(
            os.path.join(target_path, f"bell_hist_{datetime.now():%Y%m%d_%H%M%S}.png")
        )
        plt.close()

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
        self.bell_values = []

        color_map = plt.get_cmap("Set1")

        for idx, (label, (array, t)) in enumerate(self.reconstructions.items()):
            pW1_num, pW2_num, cov_2d, cov_sym_2d, angles, bell_values = (
                self.calculate_gellmann_coefficients(array, t)
            )
            self.datasets.append(cov_2d)
            self.pW1.append(pW1_num)
            self.pW2.append(pW2_num)
            self.angles_.append(angles)
            self.labels.append(label)
            self.colors.append(color_map(idx))
            self.bell_values.append(bell_values)

            # Write bell values to csv
            np.savetxt(f"bell_values_{label}.csv", bell_values, delimiter=",")

    def plot_gellmann_coefficients(self, target_path):
        """
        Plots the Gellmann coefficients in a 8x8 grid
        """

        os.makedirs(target_path, exist_ok=True)

        for pW1, pW2, cov_2d, label in zip(self.pW1, self.pW2, self.datasets, self.labels):
            cov_mean = cov_2d.mean(axis=0) / 4
            pW1 /= 2
            pW2 /= 2

            vmin = min(cov_mean.min(), pW1.mean(axis=0).min(), pW2.mean(axis=0).min())
            vmax = max(cov_mean.max(), pW1.mean(axis=0).max(), pW2.mean(axis=0).max())

            full = np.zeros((9, 9), dtype=float)
            full[1:9, 1:9] = cov_mean
            full[0, 0] = 0
            full[0, 1:9] = pW1.mean(axis=0)
            full[1:9, 0] = pW2.mean(axis=0)

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            plt.figure(figsize=(8, 6))
            plt.imshow(full, cmap="seismic", norm=norm, interpolation="nearest")
            plt.axhline(0.5, color="black", linewidth=1, linestyle="--")
            plt.axvline(0.5, color="black", linewidth=1, linestyle="--")
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=13)

            # plt.title(f"Coefficients $c_{{ij}}$ - {self.title}", fontsize=16)
            plt.xlabel(r"$W^+$ Index $j$", fontsize=16)
            plt.ylabel(r"$W^-$ Index $i$", fontsize=16)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

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
        self.plot_bell_hist()
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

        # print("Minimum count after cuts for W-:", H_Wminus_cuts.min())
        # print("Minimum count after cuts for W+:", H_Wplus_cuts.min())

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
        axs[0, 0].set_title(r"$W^-$ Truth", fontsize=20)
        axs[0, 1].imshow(
            H_Wplus,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[0, 1].set_title(r"$W^+$ Truth", fontsize=20)
        axs[1, 0].imshow(
            H_Wminus_cuts,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[1, 0].set_title(r"$W^-$ Diffusion", fontsize=20)
        axs[1, 1].imshow(
            H_Wplus_cuts,
            origin="lower",
            aspect="auto",
            extent=[phi_edges[0], phi_edges[-1], cos_edges[0], cos_edges[-1]],
            cmap="Blues",
            interpolation="nearest",
        )
        axs[1, 1].set_title(r"$W^+$ Diffusion", fontsize=20)
        for ax in axs.flat:
            ax.set_xlabel(r"$\phi$", fontsize=20)
            ax.set_ylabel(r"$\cos(\theta)$", fontsize=20)
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        # plt.colorbar(
        #     axs[0, 0].images[0],
        #     ax=axs[:, 0],
        #     orientation="vertical",
        #     fraction=0.02,
        #     pad=0.04,
        # )
        # plt.colorbar(
        #     axs[0, 1].images[0],
        #     ax=axs[:, 1],
        #     orientation="vertical",
        #     fraction=0.02,
        #     pad=0.04,
        # )
        # plt.colorbar(
        #     axs[1, 0].images[0],
        #     ax=axs[:, 0],
        #     orientation="vertical",
        #     fraction=0.02,
        #     pad=0.04,
        # )
        # plt.colorbar(
        #     axs[1, 1].images[0],
        #     ax=axs[:, 1],
        #     orientation="vertical",
        #     fraction=0.02,
        #     pad=0.04,
        # )
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
            # fig.suptitle(f"{self.title}: Φ$_{{{k}}}$", fontsize=18)

            for sample, lbl, col, sign in (
                (0, "truth  W⁻", "tab:blue", -1),
                (1, "truth cuts W⁻", "tab:cyan", -1),
                (0, "truth  W⁺", "tab:red", +1),
                (1, "truth cuts W⁺", "tab:orange", +1),
            ):
                data = self.wignerP(k, sign, sample)
                data = data[np.isfinite(data)]
                hist, edges = np.histogram(data, bins=bins, density=True)
                centres = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centres, hist, where="mid", label=lbl, color=col)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_bounds(-4.2, 4.2)
            # ax.spines['left'].set_bounds(0.2, .15)

            ax.set_ylabel("sample density")
            ax.set_xlabel(r"$\Phi^{P^{\pm}}_4$ value distribution")
            ax.legend(frameon=False,bbox_to_anchor=[.6, 0.6],fontsize=12)

            fname = f"wignerP_hist_k{k}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
            plt.tight_layout()
            plt.savefig(os.path.join(target_path, fname), dpi=150)
            plt.close(fig)

    def plot_gellman_cov44(self, target_path):
        os.makedirs(target_path, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for pW1, pW2, cov_2d, label in zip(self.pW1, self.pW2, self.datasets, self.labels):
            cov_vals = cov_2d[:, 3, 3] / 4.0
            cov_vals = cov_vals[np.isfinite(cov_vals)]
            hist, edges = np.histogram(cov_vals, bins=50, density=True, range=(-10, 10))
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centres, hist, where="mid", label=label, color=None)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r"$c_{44}$ values", fontsize=16)
        ax.set_ylabel("sample density", fontsize=16)
        ax.legend(frameon=False, fontsize=12)
        
        fname = f"gellmann_cov44_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        plt.tight_layout()
        plt.savefig(os.path.join(target_path, fname), dpi=150)
        plt.close(fig)

    def efficiency_map(self):
        N_cos = 18
        N_phi = 32

        cos_edges = np.linspace(-1, 1, N_cos + 1)
        phi_edges = np.linspace(-np.pi, np.pi, N_phi + 1)

        stacked_angles_pre = np.column_stack(
            [
                self.angles_[0][:, 2],
                self.angles_[0][:, 0],
                self.angles_[0][:, 3],
                self.angles_[0][:, 1],
            ]
        )
        stacked_angles_post = np.column_stack(
            [
                self.angles_[1][:, 2],
                self.angles_[1][:, 0],
                self.angles_[1][:, 3],
                self.angles_[1][:, 1],
            ]
        )

        finite_mask = np.isfinite(stacked_angles_pre).all(axis=1)
        stacked_angles_pre = stacked_angles_pre[finite_mask]
        finite_mask = np.isfinite(stacked_angles_post).all(axis=1)
        stacked_angles_post = stacked_angles_post[finite_mask]

        bins = [cos_edges, phi_edges, cos_edges, phi_edges]

        H_all, _ = np.histogramdd(stacked_angles_pre, bins=bins)
        H_pass, _ = np.histogramdd(stacked_angles_post, bins=bins)

        eff4d = np.divide(
            H_pass, H_all, out=np.zeros_like(H_pass, dtype=float), where=H_all > 1e-5
        )

        # alpha = 0.5  # Jeffreys
        # eff4d = (H_pass + alpha) / (H_all + 2*alpha)

        return eff4d, cos_edges, phi_edges

    @staticmethod
    def get_efficiency_weights_4d(
        cos_minus, phi_minus, cos_plus, phi_plus, eff4d, edges, default=0.0
    ):
        """
        Vectorized per-event weights from the 4D efficiency map.
        """
        cos_em, phi_em, cos_ep, phi_ep = edges

        # Bin indices (digitize returns 1..len-1 for interior; shift to 0-based)
        i_cm = np.digitize(cos_minus, cos_em) - 1
        i_pm = np.digitize(phi_minus, phi_em) - 1
        i_cp = np.digitize(cos_plus, cos_ep) - 1
        i_pp = np.digitize(phi_plus, phi_ep) - 1

        # Clip to valid interior bins [0, len-2]
        i_cm = np.clip(i_cm, 0, len(cos_em) - 2)
        i_pm = np.clip(i_pm, 0, len(phi_em) - 2)
        i_cp = np.clip(i_cp, 0, len(cos_ep) - 2)
        i_pp = np.clip(i_pp, 0, len(phi_ep) - 2)

        # Gather efficiencies
        eff_vals = eff4d[i_cm, i_pm, i_cp, i_pp]

        # Safe inversion: 1/eff, with fill-in for zeros
        weights = np.divide(
            1.0,
            eff_vals,
            out=np.full_like(eff_vals, default, dtype=float),
            where=eff_vals > 0,
        )
        return weights, eff_vals

    @staticmethod
    def compute_weighted_covariance(pW1, pW2, weights):
        num_samples = len(pW1)
        cov_weighted = np.zeros((8, 8))

        pW1 = np.array(pW1)
        pW2 = np.array(pW2)

        for i in range(8):
            for j in range(8):
                cov_weighted[i][j] = np.sum(weights * pW2[:, i] * pW1[:, j])

        return cov_weighted / num_samples

    @staticmethod
    def plot_gellmann_weighted(cov_mean, target_path):
        cov_mean = cov_mean / 4.0
        vmin = cov_mean.min()
        vmax = cov_mean.max()

        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        plt.figure(figsize=(8, 6))
        plt.imshow(cov_mean, cmap="seismic", norm=norm, interpolation="nearest")
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=18)

        # plt.title(f"Coefficients $c_{{ij}}$ - {'Weighted'}", fontsize=16)

        plt.xlabel(r"$W^+$ Index $j$", fontsize=25)
        plt.ylabel(r"$W^-$ Index $i$", fontsize=25)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Create a unique identifier based on current date and time and add the label
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coefficients_plot_weighted_{current_datetime}.png"

        os.makedirs(target_path, exist_ok=True)
        file_path = os.path.join(target_path, filename)

        plt.savefig(file_path)
        plt.close()
        print(f"Weighted plot saved to {file_path}")

    def run(self, target_path):
        self.initialize_datasets()
        self.plot_2d_angle_hist(target_path)
        self.plot_wignerP_1d_hist(target_path)
        self.plot_gellman_cov44(target_path)