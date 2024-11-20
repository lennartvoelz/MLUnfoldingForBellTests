from src.evaluation.evaluation import calculate_results
from src.utils.lorentz_vector import LorentzVector
from src.reconstruction.analytical_reconstruction import Baseline
import numpy as np

lep0_E, lep0_px, lep0_py, lep0_pz, lep1_E, lep1_px, lep1_py, lep1_pz, mpx, mpy = np.loadtxt("data/X_features.csv", delimiter=",", unpack=True)
lep0_E_truth, lep0_px_truth, lep0_py_truth, lep0_pz_truth, lep1_E_truth, lep1_px_truth, lep1_py_truth, lep1_pz_truth, neutrino1_E_truth, neutrino1_px_truth, neutrino1_py_truth, neutrino1_pz_truth, neutrino2_E_truth, neutrino2_px_truth, neutrino2_py_truth, neutrino2_pz_truth, _ = np.loadtxt("data/truth_finalstate.csv", delimiter=",", unpack=True)

# Initialize LorentzVector truth objects
lep0_truth = np.array([LorentzVector([lep0_E_truth[i], lep0_px_truth[i], lep0_py_truth[i], lep0_pz_truth[i]]) for i in range(len(lep0_E_truth))])
lep1_truth = np.array([LorentzVector([lep1_E_truth[i], lep1_px_truth[i], lep1_py_truth[i], lep1_pz_truth[i]]) for i in range(len(lep1_E_truth))])
neutrino1_truth = np.array([LorentzVector([neutrino1_E_truth[i], neutrino1_px_truth[i], neutrino1_py_truth[i], neutrino1_pz_truth[i]]) for i in range(len(neutrino1_E_truth))])
neutrino2_truth = np.array([LorentzVector([neutrino2_E_truth[i], neutrino2_px_truth[i], neutrino2_py_truth[i], neutrino2_pz_truth[i]]) for i in range(len(neutrino2_E_truth))])
                      
# Initialize LorentzVector objects
lep0 = np.array([LorentzVector([lep0_E[i], lep0_px[i], lep0_py[i], lep0_pz[i]]) for i in range(len(lep0_E))])
lep1 = np.array([LorentzVector([lep1_E[i], lep1_px[i], lep1_py[i], lep1_pz[i]]) for i in range(len(lep1_E))])

# Calculate analytical reconstruction
neutrinos = np.array([Baseline(lep0[i], lep1[i], mpx[i], mpy[i]).calculate_neutrino_solutions() for i in range(len(lep0))])
neutrino0 = neutrinos[:,0]
neutrino1 = neutrinos[:,1]

# Calculate results
results = calculate_results(truth=np.array([lep0_truth, lep1_truth, neutrino1_truth, neutrino2_truth]).T, analytical_reconstruction=np.array([lep0, lep1, neutrino0, neutrino1]).T)
results.initialize_datasets()

# Plot results
results.plot_gellmann_coefficients("reports/figures/")
results.plot_gellmann_coefficients_hist("reports/figures/")