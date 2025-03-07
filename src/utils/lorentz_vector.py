import numpy as np
from src.utils import change_of_coordinates

class LorentzVector():
    def __init__(self, input_array, type="four-vector"):
        """
        Initializes a LorentzVector object

        Parameters:
            input_array (list): input array
            type (str): type of input array
        """
        if type == "four-vector":
            self.E = input_array[0]
            self.p_x = input_array[1]
            self.p_y = input_array[2]
            self.p_z = input_array[3]

        elif type == "experimental":
            self.E = change_of_coordinates.exp_to_four_vec(input_array[0], input_array[1], input_array[2], input_array[3])[0]
            self.p_x = change_of_coordinates.exp_to_four_vec(input_array[0], input_array[1], input_array[2], input_array[3])[1]
            self.p_y = change_of_coordinates.exp_to_four_vec(input_array[0], input_array[1], input_array[2], input_array[3])[2]
            self.p_z = change_of_coordinates.exp_to_four_vec(input_array[0], input_array[1], input_array[2], input_array[3])[3]

        else:
            raise ValueError("Invalid type")
        
    def __add__(self, other):
        """
        Adds two LorentzVector objects

        Parameters:
            other (LorentzVector): other LorentzVector object

        Returns:
            LorentzVector: sum of two LorentzVector objects
        """
        return LorentzVector([self.E + other.E, self.p_x + other.p_x, self.p_y + other.p_y, self.p_z + other.p_z], type="four-vector")
    
    def __sub__(self, other):
        """
        Subtracts two LorentzVector objects

        Parameters:
            other (LorentzVector): other LorentzVector object

        Returns:
            LorentzVector: difference of two LorentzVector objects
        """
        return LorentzVector([self.E - other.E, self.p_x - other.p_x, self.p_y - other.p_y, self.p_z - other.p_z], type="four-vector")
    
    def __mul__(self, other):
        """
        Multiplies two LorentzVector objects

        Parameters:
            other : int, float

        Returns:
            float: dot product of LorentzVector and int, float
        """
        return LorentzVector([self.E * other, self.p_x * other, self.p_y * other, self.p_z * other], type="four-vector")
    
    def to_numpy(self):
        """
        Returns:
            np.array: LorentzVector object as numpy array
        """
        return np.array([self.E, self.p_x, self.p_y, self.p_z])
    
    def get_eta(self):
        """
        Returns:
            float: pseudorapidity of a LorentzVector object
        """
        return 0.5 * np.log((np.linalg.norm(self.get_momentum()) + self.p_z)/(np.linalg.norm(self.get_momentum()) - self.p_z))
    
    def get_invariant_mass(self):
        """
        Calculates the invariant mass of a LorentzVector object

        Returns:
            float: invariant mass of a LorentzVector object
        """
        return np.sqrt(self.E**2 - self.p_x**2 - self.p_y**2 - self.p_z**2)
    
    def get_momentum(self):
        """
        Returns:
            float: momentum of a LorentzVector object
        """
        p3 = np.array([self.p_x, self.p_y, self.p_z])
        return p3

    def get_energy(self):
        """
        Returns:
            float: energy of a LorentzVector object
        """
        return self.E
    
    def get_beta(self):
        """
        Returns:
            float: beta of a LorentzVector object
        """
        p3 = np.array([self.p_x, self.p_y, self.p_z])
        return p3/self.E
    
    def get_pt(self):
        """
        Returns:
            float: transverse momentum of a LorentzVector object
        """
        pt = np.sqrt(self.p_x**2 + self.p_y**2)
        return pt
    
    def lorentz_boost(self, beta):
        """
        Boosts a LorentzVector object

        Parameters:
            beta (float): boost 3-vector
        """
        if np.dot(beta, beta) >= 1:
            gamma = 100
        else:
            gamma = 1/np.sqrt(1 - np.dot(beta, beta))
        E = self.E
        p = np.array([self.p_x, self.p_y, self.p_z])
        p4 = np.array([E, p[0], p[1], p[2]])
        beta_squared = np.dot(beta, beta)

        boost_matrix = np.array([[gamma, -gamma*beta[0], -gamma*beta[1], -gamma*beta[2]],
                      [-gamma*beta[0], 1 + (gamma - 1)/beta_squared*beta[0]**2, (gamma - 1)/beta_squared*beta[0]*beta[1], (gamma - 1)/beta_squared*beta[0]*beta[2]],
                      [-gamma*beta[1], (gamma - 1)/beta_squared*beta[1]*beta[0], 1 + (gamma - 1)/beta_squared*beta[1]**2, (gamma - 1)/beta_squared*beta[1]*beta[2]],
                      [-gamma*beta[2], (gamma - 1)/beta_squared*beta[2]*beta[0], (gamma - 1)/beta_squared*beta[2]*beta[1], 1 + (gamma - 1)/beta_squared*beta[2]**2]])
        
        #can be used but beta = -beta is needed then
        #boost_matrix_inv = np.linalg.inv(boost_matrix)
        #p4_prime = boost_matrix_inv @ p4
        p4_prime = boost_matrix @ p4

        return LorentzVector([p4_prime[0], p4_prime[1], p4_prime[2], p4_prime[3]], type="four-vector")