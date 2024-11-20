import numpy as np

def exp_to_four_vec(p_t, eta, phi, m):
    """
    Converts the transverse momentum, pseudorapidity, mass and azimuthal angle to a four-vector

    Parameters:
        p_t (float): transverse momentum
        eta (float): pseudorapidity
        phi (float): azimuthal angle
        m (float): mass

    Returns:
        p (np.array): four-vector
    """
    #calculate rapidity
    y = np.log((np.sqrt(m**2 + p_t**2*np.cosh(eta)**2) + p_t*np.sinh(eta))/np.sqrt(m**2 + p_t**2))

    #calculate transverse mass
    m_t = np.sqrt(m**2 + p_t**2)

    #calculate energy
    E = m_t*np.cosh(y)

    #calculate x, y, z components of momentum
    p_x = p_t*np.cos(phi)
    p_y = p_t*np.sin(phi)
    p_z = m_t*np.sinh(y)

    return np.array([E, p_x, p_y, p_z])