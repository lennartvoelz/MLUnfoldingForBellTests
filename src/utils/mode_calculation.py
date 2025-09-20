import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


class ModeCalculator:
    """Utility class for calculating modes of various distributions."""
    
    def __init__(self, method='histogram', **kwargs):
        """
        Initialize mode calculator.
        
        Parameters:
            method: Method for mode calculation ('histogram', 'kde', 'gaussian_fit')
            **kwargs: Additional parameters for specific methods
        """
        self.method = method
        self.kwargs = kwargs
        
    def calculate_mode_histogram(self, data, bins=50):
        """
        Calculate mode using histogram method.
        
        Parameters:
            data: Input data array
            bins: Number of bins or bin edges
            
        Returns:
            Mode value
        """
        hist, bin_edges = np.histogram(data, bins=bins)
        max_bin_idx = np.argmax(hist)
        mode = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
        return mode
        
    def calculate_mode_kde(self, data, bandwidth='scott', kernel='gaussian'):
        """
        Calculate mode using kernel density estimation.
        
        Parameters:
            data: Input data array
            bandwidth: Bandwidth for KDE
            kernel: Kernel type
            
        Returns:
            Mode value
        """
        # Remove NaN and infinite values
        data_clean = data[np.isfinite(data)]
        
        if len(data_clean) == 0:
            return np.nan
            
        # Fit KDE
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(data_clean.reshape(-1, 1))
        
        # Create evaluation grid
        x_min, x_max = np.min(data_clean), np.max(data_clean)
        x_grid = np.linspace(x_min, x_max, 1000)
        
        # Evaluate density
        log_density = kde.score_samples(x_grid.reshape(-1, 1))
        density = np.exp(log_density)
        
        # Find mode
        mode_idx = np.argmax(density)
        mode = x_grid[mode_idx]
        
        return mode
        
    def calculate_mode_gaussian_fit(self, data):
        """
        Calculate mode by fitting a Gaussian distribution.
        
        Parameters:
            data: Input data array
            
        Returns:
            Mode value (mean of fitted Gaussian)
        """
        # Remove NaN and infinite values
        data_clean = data[np.isfinite(data)]
        
        if len(data_clean) == 0:
            return np.nan
            
        # Fit Gaussian
        mu, sigma = stats.norm.fit(data_clean)
        
        # For Gaussian, mode = mean
        return mu
        
    def calculate_mode(self, data, **kwargs):
        """
        Calculate mode using the specified method.
        
        Parameters:
            data: Input data array
            **kwargs: Method-specific parameters
            
        Returns:
            Mode value
        """
        if self.method == 'histogram':
            bins = kwargs.get('bins', self.kwargs.get('bins', 50))
            return self.calculate_mode_histogram(data, bins)
        elif self.method == 'kde':
            bandwidth = kwargs.get('bandwidth', self.kwargs.get('bandwidth', 'scott'))
            kernel = kwargs.get('kernel', self.kwargs.get('kernel', 'gaussian'))
            return self.calculate_mode_kde(data, bandwidth, kernel)
        elif self.method == 'gaussian_fit':
            return self.calculate_mode_gaussian_fit(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class KinematicModeCalculator:
    """Specialized mode calculator for particle physics kinematic variables."""
    
    def __init__(self, mode_calculator=None):
        """
        Initialize with a mode calculator.
        
        Parameters:
            mode_calculator: ModeCalculator instance
        """
        if mode_calculator is None:
            self.mode_calculator = ModeCalculator(method='kde')
        else:
            self.mode_calculator = mode_calculator
            
    def calculate_four_vector_modes(self, four_vectors):
        """
        Calculate modes for four-vector components.
        
        Parameters:
            four_vectors: Array of shape (N, 4) with [E, px, py, pz]
            
        Returns:
            Dictionary with mode values for each component
        """
        if four_vectors.shape[1] != 4:
            raise ValueError("Four-vectors must have shape (N, 4)")
            
        E, px, py, pz = four_vectors[:, 0], four_vectors[:, 1], four_vectors[:, 2], four_vectors[:, 3]
        
        modes = {
            'E': self.mode_calculator.calculate_mode(E),
            'px': self.mode_calculator.calculate_mode(px),
            'py': self.mode_calculator.calculate_mode(py),
            'pz': self.mode_calculator.calculate_mode(pz)
        }
        
        return modes
        
    def calculate_kinematic_modes(self, four_vectors):
        """
        Calculate modes for derived kinematic variables.
        
        Parameters:
            four_vectors: Array of shape (N, 4) with [E, px, py, pz]
            
        Returns:
            Dictionary with mode values for kinematic variables
        """
        E, px, py, pz = four_vectors[:, 0], four_vectors[:, 1], four_vectors[:, 2], four_vectors[:, 3]
        
        # Calculate derived quantities
        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        
        # Avoid division by zero
        eta = np.where(p != 0, 0.5 * np.log((p + pz) / (p - pz + 1e-8)), 0)
        phi = np.arctan2(py, px)
        
        # Calculate invariant mass (assuming massless particles for neutrinos)
        m = np.sqrt(np.maximum(0, E**2 - p**2))
        
        modes = {
            'pt': self.mode_calculator.calculate_mode(pt),
            'p': self.mode_calculator.calculate_mode(p),
            'eta': self.mode_calculator.calculate_mode(eta),
            'phi': self.mode_calculator.calculate_mode(phi),
            'mass': self.mode_calculator.calculate_mode(m)
        }
        
        return modes
        
    def calculate_all_modes(self, four_vectors):
        """
        Calculate modes for both four-vector components and derived quantities.
        
        Parameters:
            four_vectors: Array of shape (N, 4) with [E, px, py, pz]
            
        Returns:
            Dictionary with all mode values
        """
        four_vec_modes = self.calculate_four_vector_modes(four_vectors)
        kinematic_modes = self.calculate_kinematic_modes(four_vectors)
        
        # Combine dictionaries
        all_modes = {**four_vec_modes, **kinematic_modes}
        
        return all_modes
        
    def calculate_event_modes(self, events):
        """
        Calculate modes for multiple particles in events.
        
        Parameters:
            events: Array of shape (N, n_particles * 4) with concatenated four-vectors
            
        Returns:
            Dictionary with modes for each particle
        """
        n_components = events.shape[1]
        n_particles = n_components // 4
        
        if n_components % 4 != 0:
            raise ValueError("Event data must contain complete four-vectors")
            
        modes_by_particle = {}
        
        for i in range(n_particles):
            start_idx = i * 4
            end_idx = (i + 1) * 4
            particle_4vec = events[:, start_idx:end_idx]
            
            particle_modes = self.calculate_all_modes(particle_4vec)
            modes_by_particle[f'particle_{i}'] = particle_modes
            
        return modes_by_particle


def calculate_distribution_properties(data, include_modes=True):
    """
    Calculate comprehensive distribution properties including modes.
    
    Parameters:
        data: Input data array
        include_modes: Whether to calculate modes using different methods
        
    Returns:
        Dictionary with distribution properties
    """
    # Remove NaN and infinite values
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        return {'error': 'No valid data points'}
        
    properties = {
        'mean': np.mean(data_clean),
        'median': np.median(data_clean),
        'std': np.std(data_clean),
        'min': np.min(data_clean),
        'max': np.max(data_clean),
        'q25': np.percentile(data_clean, 25),
        'q75': np.percentile(data_clean, 75)
    }
    
    if include_modes:
        # Calculate modes using different methods
        hist_calc = ModeCalculator(method='histogram')
        kde_calc = ModeCalculator(method='kde')
        gauss_calc = ModeCalculator(method='gaussian_fit')
        
        properties['mode_histogram'] = hist_calc.calculate_mode(data_clean)
        properties['mode_kde'] = kde_calc.calculate_mode(data_clean)
        properties['mode_gaussian'] = gauss_calc.calculate_mode(data_clean)
        
    return properties
