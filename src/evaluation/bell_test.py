import numpy as np
from src.utils.lorentz_vector import LorentzVector


class I_3():
    """
    Class to perform analysis on lepton and neutrino events,
    calculating Wigner symbols and testing Bell inequalities.
    """

    # Precompute Gell-Mann matrices as class variables
    GMLIST = [None] * 8

    def __init__(self, lep1, lep2, neutrino1, neutrino2, lep0_charge=-1):
        self.update_data(lep1, lep2, neutrino1, neutrino2, lep0_charge)

    def update_data(self, lep1, lep2, neutrino1, neutrino2, lep0_charge=-1):
        """
        Updates the data for the I_3 analysis.

        Parameters:
            lep1 (LorentzVector): First lepton.
            lep2 (LorentzVector): Second lepton.
            neutrino1 (LorentzVector): First neutrino.
            neutrino2 (LorentzVector): Second neutrino.
            lep0_charge (int): Charge of lep0 (default: -1).
        """
        self.lep1 = lep1
        self.lep2 = lep2
        self.neutrino1 = neutrino1
        self.neutrino2 = neutrino2
        self.lep0_charge = lep0_charge
        self.beam = LorentzVector([1, 0, 0, 1])

        # Initialize Gell-Mann matrices if not already done
        if I_3.GMLIST[0] is None:
            I_3._initialize_gellmann_matrices()

        self.pW1 = np.zeros(8)
        self.pW2 = np.zeros(8)
        self.cov = np.zeros((8, 8), dtype=float, order='C')
        self.cov_sym = np.zeros((8, 8), dtype=float, order='C')

    @staticmethod
    def _initialize_gellmann_matrices():
        """
        Initializes the Gell-Mann matrices and stores them in the class variable GMLIST.
        """
        d = 3  # Dimension for Gell-Mann matrices
        I_3.GMLIST = [
            I_3.gellmann(2, 1, d),
            I_3.gellmann(1, 2, d),
            I_3.gellmann(1, 1, d),
            I_3.gellmann(3, 1, d),
            I_3.gellmann(1, 3, d),
            I_3.gellmann(3, 2, d),
            I_3.gellmann(2, 3, d),
            I_3.gellmann(2, 2, d),
        ]

    def get_type(self):
        """
        Determines the type based on lep0_charge.

        Returns:
            int: 1 if lep0_charge > 0, else 2.
        """
        return 1 if self.lep0_charge > 0 else 2

    def analysis_prep(self):
        """
        Prepares the data for analysis by performing boosts and calculating direction cosines.

        Returns:
            tuple: wig_1, wig_2 (numpy arrays of Wigner symbols), vars (array of LorentzVectors).
        """
        lab_1, lab_2, lab_3, lab_4 = self._get_lab_variables()

        # Calculate momenta in lab frame
        lab_W1 = lab_1 + lab_4
        lab_W2 = lab_2 + lab_3
        lab_H = lab_W1 + lab_W2

        # Boost to Higgs rest frame
        beta_H = lab_H.get_beta()
        H_W1 = lab_W1.lorentz_boost(beta_H)
        H_W2 = lab_W2.lorentz_boost(beta_H)
        H_beam = self.beam.lorentz_boost(beta_H)

        # Direction vectors
        k = H_W2.get_momentum()
        k /= np.linalg.norm(k)
        p = H_beam.get_momentum()
        p /= np.linalg.norm(p)

        # Orthonormal basis vectors
        y = np.dot(p, k)
        r = np.sqrt(1 - y**2)
        if r == 0:
            r = 0.0001
        
        r_vec = (p - y * k) / r
        n = np.cross(p, k) / r

        # Boost leptons into W rest frames
        lep0_H = lab_1.lorentz_boost(beta_H)
        lep1_H = lab_2.lorentz_boost(beta_H)
        W1 = lep0_H.lorentz_boost(H_W1.get_beta())
        W2 = lep1_H.lorentz_boost(H_W2.get_beta())

        dir_1 = W1.get_momentum()
        dir_2 = W2.get_momentum()

        # Normalize direction vectors
        dir_1_norm = np.linalg.norm(dir_1)
        dir_2_norm = np.linalg.norm(dir_2)
        if dir_1_norm != 0:
            dir_1 /= dir_1_norm
        if dir_2_norm != 0:
            dir_2 /= dir_2_norm

        # Direction cosines
        xi_x_1 = np.dot(n, dir_1)
        xi_y_1 = np.dot(r_vec, dir_1)
        xi_z_1 = np.dot(k, dir_1)

        xi_x_2 = np.dot(n, dir_2)
        xi_y_2 = np.dot(r_vec, dir_2)
        xi_z_2 = np.dot(k, dir_2)

        # Clip values to avoid numerical errors
        xi_x_1 = np.clip(xi_x_1, -1, 1)
        xi_y_1 = np.clip(xi_y_1, -1, 1)
        xi_z_1 = np.clip(xi_z_1, -1, 1)

        xi_x_2 = np.clip(xi_x_2, -1, 1)
        xi_y_2 = np.clip(xi_y_2, -1, 1)
        xi_z_2 = np.clip(xi_z_2, -1, 1)

        wig_1 = self.wig_neg(xi_x_1, xi_y_1, xi_z_1)  # Lepton -
        wig_2 = self.wig_plus(xi_x_2, xi_y_2, xi_z_2)  # Lepton +

        vars = np.array([lab_1, lab_2, lab_3, lab_4])

        return wig_1, wig_2, vars

    def analysis(self):
        """
        Performs the analysis by updating Wigner symbols and calculating covariances.

        Returns:
            tuple: Updated pW1, pW2, cov, cov_sym arrays.
        """
        pW1_event, pW2_event, _ = self.analysis_prep()

        # Update Wigner symbol sums
        self.pW1 += pW1_event
        self.pW2 += pW2_event

        # Calculate correlations using vectorized operations
        cov_increment = np.outer(pW2_event, pW1_event)
        self.cov += cov_increment
        self.cov_sym += cov_increment + cov_increment.T

        return self.pW1, self.pW2, self.cov, self.cov_sym

    @staticmethod
    def gellmann(j, k, d):
        """
        Returns a generalized Gell-Mann matrix of dimension d.

        Parameters:
            j (int): Index j.
            k (int): Index k.
            d (int): Dimension of the matrix.

        Returns:
            numpy.ndarray: Gell-Mann matrix.
        """
        if j > k:
            gjkd = np.zeros((d, d), dtype=np.complex128)
            gjkd[j - 1, k - 1] = 1
            gjkd[k - 1, j - 1] = 1
        elif k > j:
            gjkd = np.zeros((d, d), dtype=np.complex128)
            gjkd[j - 1, k - 1] = -1j
            gjkd[k - 1, j - 1] = 1j
        elif j == k and j < d:
            diag_elements = [1 if n <= j else -j if n == (j + 1) else 0 for n in range(1, d + 1)]
            factor = np.sqrt(2 / (j * (j + 1)))
            gjkd = factor * np.diag(diag_elements)
        else:
            gjkd = np.eye(d, dtype=np.complex128)
        return gjkd

    @staticmethod
    def wig_plus(etax, etay, etaz):
        """
        Calculates the Wigner symbols for positive leptons.

        Parameters:
            etax (float): Direction cosine along x.
            etay (float): Direction cosine along y.
            etaz (float): Direction cosine along z.

        Returns:
            numpy.ndarray: Array of Wigner symbol values.
        """
        sqrt2 = np.sqrt(2)
        cos2theta = 2 * etaz**2 - 1

        term1 = sqrt2 * (5 * etaz + 1)
        term2 = sqrt2 * (1 - 5 * etaz)
        term3 = etaz + (15 / 4) * cos2theta + 5 / 4
        term4 = 5 * (etax**2 - etay**2)
        term5 = 10 * etax * etay
        term6 = (0.25 / np.sqrt(3)) * (12 * etaz - 15 * cos2theta - 5)

        wig_plus_event = np.array([
            term1 * etax,
            term1 * etay,
            term3,
            term4,
            term5,
            term2 * etax,
            term2 * etay,
            term6
        ])

        return wig_plus_event

    @staticmethod
    def wig_neg(etax, etay, etaz):
        """
        Calculates the Wigner symbols for negative leptons.

        Parameters:
            etax (float): Direction cosine along x.
            etay (float): Direction cosine along y.
            etaz (float): Direction cosine along z.

        Returns:
            numpy.ndarray: Array of Wigner symbol values.
        """
        sqrt2 = np.sqrt(2)
        cos2theta = 2 * etaz**2 - 1

        term1 = -sqrt2 * (-5 * etaz + 1)
        term2 = -sqrt2 * (1 + 5 * etaz)
        term3 = 0.25 * (-4 * etaz + 15 * cos2theta + 5)
        term4 = 5 * (etax**2 - etay**2)
        term5 = 10 * etax * etay
        term6 = (0.25 / np.sqrt(3)) * (-12 * etaz - 15 * cos2theta - 5)

        wig_minus_event = np.array([
            term1 * etax,
            term1 * etay,
            term3,
            term4,
            term5,
            term2 * etax,
            term2 * etay,
            term6
        ])

        return wig_minus_event

    @staticmethod
    def CGLMP_test(gmarray):
        """
        Performs the CGLMP Bell inequality test.

        Parameters:
            gmarray (numpy.ndarray): Density matrix from tomography.

        Returns:
            float: Expectation value of the Bell operator.
        """
        sqrt3 = np.sqrt(3)
        term = (sqrt3 * (gmarray[0, 0] + gmarray[0, 5] + gmarray[1, 1] + gmarray[1, 6]) -3 * (gmarray[3, 3] + gmarray[4, 4]) + sqrt3 * (gmarray[5, 0] + gmarray[5, 5] + gmarray[6, 1] + gmarray[6, 6]))
        bell_value = -(4 / 3) * term
        return bell_value

    def _get_lab_variables(self):
        """
        Determines lab variables based on the type.

        Returns:
            tuple: lab_1, lab_2, lab_3, lab_4 (LorentzVectors).
        """
        if self.get_type() == 1:
            lab_1 = self.lep2  # Electron
            lab_2 = self.lep1  # Anti-muon
            lab_3 = self.neutrino1  # Muon-neutrino
            lab_4 = self.neutrino2  # Anti-electron-neutrino
        else:
            lab_1 = self.lep1  # Muon
            lab_2 = self.lep2  # Positron
            lab_3 = self.neutrino2  # Electron-neutrino
            lab_4 = self.neutrino1  # Anti-muon-neutrino
        return lab_1, lab_2, lab_3, lab_4