import numpy as np
from src.utils.lorentz_vector import LorentzVector


class Baseline(LorentzVector):
    def __init__(self, lep0, lep1, mpx, mpy):
        """
        Initializes a Baseline object.

        Parameters:
            input_list (list): List of LorentzVector objects.
            mpx (float): Missing transverse momentum in the x-direction.
            mpy (float): Missing transverse momentum in the y-direction.
        """
        self.update_data(lep0, lep1, mpx, mpy)

    def update_data(self, lep0, lep1, mpx, mpy):
        """
        Updates the data of the Baseline object.

        Parameters:
            input_list (list): List of LorentzVector objects.
            mpx (float): Missing transverse momentum in the x-direction.
            mpy (float): Missing transverse momentum in the y-direction.
        """
        self.p4_lep0 = lep0
        self.p4_lep1 = lep1
        self.p4_ll = self.p4_lep0 + self.p4_lep1
        self.mpx = mpx
        self.mpy = mpy
        self.mpt = np.sqrt(mpx**2 + mpy**2)

    # --------------------------------------
    # Energy and Momentum Reconstruction
    # --------------------------------------

    def di_neutrino_energy(self, M_vv, p_vv_z):
        """
        Calculates the di-neutrino energy.

        Parameters:
            M_vv (float): Invariant mass of the di-neutrino system.
            p_vv_z (float): z-component of the di-neutrino four-momentum.

        Returns:
            float: Di-neutrino energy.
        """
        return np.sqrt(M_vv**2 + self.mpt**2 + p_vv_z**2)

    @staticmethod
    def quadratic_formula(a, b, c):
        """
        Solves the quadratic equation ax^2 + bx + c = 0.

        Parameters:
            a (float): Coefficient of x^2.
            b (float): Coefficient of x.
            c (float): Constant term.

        Returns:
            np.array or None: Solutions to the quadratic equation.
        """
        # print(a, b, c)
        discriminant = b**2 - 4 * a * c
        # print(discriminant)
        if discriminant < 0:
            return None
        elif discriminant == 0:
            return np.array([-b / (2 * a)])
        else:
            sqrt_discriminant = np.sqrt(discriminant)
            return np.array(
                [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]
            )

    def solve_for_p_vv_z(self, M_vv, M_H, M_ll, p_ll, E_ll, missing_px, missing_py):
        """
        Sets up the quadratic equation to solve for the z-component of the missing momentum.

        Parameters:
            M_vv (float): Invariant mass of the di-neutrino system.
            M_H (float): Higgs boson mass.
            M_ll (float): Invariant mass of the dilepton system.
            p_ll (np.array): Momentum components of the dilepton system.
            E_ll (float): Energy of the dilepton system.
            missing_px (float): Missing transverse momentum in x-direction.
            missing_py (float): Missing transverse momentum in y-direction.

        Returns:
            np.array or None: Solutions to the quadratic equation.
        """
        M_fixed_squared = (
            M_H**2
            - M_ll**2
            - M_vv**2
            + 2 * p_ll[0] * missing_px
            + 2 * p_ll[1] * missing_py
        )
        a = p_ll[2] ** 2 - E_ll**2
        b = M_fixed_squared * p_ll[2]
        c = 0.25 * M_fixed_squared**2 - E_ll**2 * (M_vv**2 + self.mpt**2)

        return self.quadratic_formula(a, b, c)

    def reconstruct_p_vv_z(self, M_H=125, M_vv=30):
        """
        Reconstructs the z-component of the di-neutrino four-momentum.

        Parameters:
            M_H (float): Higgs boson mass (default: 124.97 GeV).
            M_vv (float): Invariant mass of the di-neutrino system (default: 30 GeV).

        Returns:
            float: z-component of the neutrino four-momentum.
        """
        p_ll = self.p4_ll.get_momentum()
        E_ll = self.p4_ll.get_energy()
        M_ll = self.p4_ll.get_invariant_mass()
        missing_px, missing_py = self.mpx, self.mpy

        solutions = self.solve_for_p_vv_z(
            M_vv, M_H, M_ll, p_ll, E_ll, missing_px, missing_py
        )

        if solutions is None:
            M_vv = 0
            solutions = self.solve_for_p_vv_z(
                M_vv, M_H, M_ll, p_ll, E_ll, missing_px, missing_py
            )

        if solutions is None:
            return (p_ll[2] * self.mpt) / np.sqrt(E_ll**2 - p_ll[2] ** 2)

        p_vv_z_1, p_vv_z_2 = solutions
        E_vv_1 = self.di_neutrino_energy(M_vv, p_vv_z_1)
        E_vv_2 = self.di_neutrino_energy(M_vv, p_vv_z_2)

        # Calculate Higgs four-momenta for each solution
        p4_H_1 = np.array(
            [
                E_ll + E_vv_1,
                p_ll[0] + missing_px,
                p_ll[1] + missing_py,
                p_ll[2] + p_vv_z_1,
            ]
        )
        p4_H_2 = np.array(
            [
                E_ll + E_vv_2,
                p_ll[0] + missing_px,
                p_ll[1] + missing_py,
                p_ll[2] + p_vv_z_2,
            ]
        )

        # Choose solution with smallest |cos(ψ_ll)|
        beta_1 = p4_H_1[1:] / p4_H_1[0]
        beta_2 = p4_H_2[1:] / p4_H_2[0]

        cos_psi_1 = self.calculate_cos_psi(beta_1)
        cos_psi_2 = self.calculate_cos_psi(beta_2)

        if np.abs(cos_psi_1) < np.abs(cos_psi_2):
            return p_vv_z_1
        else:
            return p_vv_z_2

    def calculate_cos_psi(self, beta):
        """
        Calculates the cosine of the angle between the boosted dilepton momentum and the boosted z-axis.

        Parameters:
            beta (np.array): Beta vector for Lorentz boost.

        Returns:
            float: Cosine of the angle ψ between the boosted dilepton and z-axis.
        """
        boosted_ll = self.p4_ll.lorentz_boost(beta)
        z_axis = LorentzVector([1, 0, 0, 1]).lorentz_boost(beta)
        boosted_ll_momentum = boosted_ll.get_momentum()
        z_axis_momentum = z_axis.get_momentum()

        # Calculate cosine of the angle between boosted_ll and z_axis
        cos_theta = np.dot(boosted_ll_momentum, z_axis_momentum) / (
            np.linalg.norm(boosted_ll_momentum) * np.linalg.norm(z_axis_momentum)
        )

        return cos_theta

    # --------------------------------------
    # Seperate neutrino solutions
    # --------------------------------------

    def restrictions(self):
        p_vv_z = self.reconstruct_p_vv_z()
        M_vv = 30
        p_vv_x = self.mpx
        p_vv_y = self.mpy
        E_vv = self.di_neutrino_energy(M_vv, p_vv_z)
        p4_vv = LorentzVector([E_vv, p_vv_x, p_vv_y, p_vv_z])

        # approach 2
        p_miss = LorentzVector([self.mpt, self.mpx, self.mpy, 0])
        # p_miss = p4_vv
        dir_w1 = self.p4_lep0.get_momentum() / np.linalg.norm(
            self.p4_lep0.get_momentum()
        ) + p_miss.get_momentum() / (np.linalg.norm(p_miss.get_momentum()) * 2)
        p_abs_w1 = np.linalg.norm(
            self.p4_lep0.get_momentum() + p_miss.get_momentum() / 2
        )
        # p_abs_w1 = 35*10**3

        E_W1 = np.sqrt(70**2 + p_abs_w1**2)
        W1 = LorentzVector(
            [E_W1, p_abs_w1 * dir_w1[0], p_abs_w1 * dir_w1[1], p_abs_w1 * dir_w1[2]]
        )

        dir_w2 = self.p4_lep1.get_momentum() / np.linalg.norm(
            self.p4_lep1.get_momentum()
        ) + p_miss.get_momentum() / (np.linalg.norm(p_miss.get_momentum()) * 2)
        p_abs_w2 = np.linalg.norm(
            self.p4_lep1.get_momentum() + p_miss.get_momentum() / 2
        )
        # p_abs_w2 = 29*10**3

        E_W2 = np.sqrt(39**2 + p_abs_w2**2)
        W2 = LorentzVector(
            [E_W2, p_abs_w2 * dir_w2[0], p_abs_w2 * dir_w2[1], p_abs_w2 * dir_w2[2]]
        )

        p4_v_on = (p4_vv + self.p4_lep1 - self.p4_lep0 + W1 - W2) * 0.5

        p4_v_off = p4_vv - p4_v_on

        return p4_v_on.to_numpy(), p4_v_off.to_numpy()

    def calculate_neutrino_solutions(self):
        if self.p4_lep0.get_pt() > self.p4_lep1.get_pt():
            p4_lep_onshell = self.p4_lep0
            # p4_lep_offshell = self.p4_lep1
            switched = False
        else:
            p4_lep_onshell = self.p4_lep1
            # p4_lep_offshell = self.p4_lep0
            switched = True

        p_vv_z = self.reconstruct_p_vv_z()
        M_vv = 0
        p_vv = LorentzVector(
            [self.di_neutrino_energy(M_vv, p_vv_z), self.mpx, self.mpy, p_vv_z]
        )

        M_W_onshell = p4_lep_onshell.get_invariant_mass()
        alpha_squared = M_W_onshell**2 / (
            2
            * (
                p4_lep_onshell.get_energy() * p_vv.get_energy()
                - np.dot(p4_lep_onshell.get_momentum(), p_vv.get_momentum())
            )
        )

        if alpha_squared < 0:
            alpha_squared = 0
        elif alpha_squared > 1:
            alpha_squared = 1

        square_root = (
            p_vv.get_energy() - alpha_squared * np.linalg.norm(p_vv.get_momentum())
        ) ** 2 - (1 - alpha_squared) ** 2 * np.linalg.norm(p_vv.get_momentum()) ** 2
        if square_root < 0:
            square_root = 0
        M_vf = np.sqrt(square_root)

        p_v_onshell = np.array(
            [
                0,
                alpha_squared * p_vv.get_momentum()[0],
                alpha_squared * p_vv.get_momentum()[1],
                alpha_squared * p_vv.get_momentum()[2],
            ]
        )
        p_v_offshell = np.array(
            [
                M_vf,
                (1 - alpha_squared) * p_vv.get_momentum()[0],
                (1 - alpha_squared) * p_vv.get_momentum()[1],
                (1 - alpha_squared) * p_vv.get_momentum()[2],
            ]
        )

        if switched:
            # lep0 = p4_lep_offshell
            # lep1 = p4_lep_onshell
            v0 = p_v_offshell
            v1 = p_v_onshell
        else:
            # lep0 = p4_lep_onshell
            # lep1 = p4_lep_offshell
            v0 = p_v_onshell
            v1 = p_v_offshell

        return v0, v1
