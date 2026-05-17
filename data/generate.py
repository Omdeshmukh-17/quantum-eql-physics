"""
Quantum Physics Data Generator
Generates synthetic data for three quantum systems:
  1. Quantum Harmonic Oscillator  — E_n = hbar * omega * (n + 0.5)
  2. Hydrogen Atom Energy Levels  — E_n = -13.6 / n^2  (eV)
  3. de Broglie Relation          — lambda = h / p
  
No downloads needed. Pure NumPy. Runs on your laptop instantly.
"""

import numpy as np
import os

# Constants (SI units) 
HBAR  = 1.0545718e-34   # reduced Planck constant (J·s)
H     = 6.6260701e-34   # Planck constant (J·s)
EV    = 1.60218e-19     # 1 electron volt in Joules

# 1. Quantum Harmonic Oscillator

def generate_qho(num_samples=2000, omega_range=(1e12, 1e14),
                 n_max=20, noise_level=0.0, seed=42):
    """
    Generate energy level data for quantum harmonic oscillator.
    
    Inputs:  n (quantum number), omega (angular frequency)
    Output:  E_n = hbar * omega * (n + 0.5)
    
    Args:
        num_samples  : number of data points
        omega_range  : range of angular frequencies (rad/s)
        n_max        : maximum quantum number
        noise_level  : fractional Gaussian noise to add (0.0 = clean)
        seed         : random seed for reproducibility
    
    Returns:
        X : (num_samples, 2) array  [n, omega]
        y : (num_samples,)   array  [E_n in Joules]
    """
    rng   = np.random.default_rng(seed)
    n     = rng.integers(0, n_max + 1, size=num_samples).astype(np.float64)
    omega = rng.uniform(omega_range[0], omega_range[1], size=num_samples)
    
    # Ground truth equation
    E_n = HBAR * omega * (n + 0.5)
    
    # Add noise if requested
    if noise_level > 0.0:
        noise = rng.normal(0, noise_level * np.abs(E_n))
        E_n   = E_n + noise
    
    X = np.column_stack([n, omega])
    y = E_n
    return X, y


# 2. Hydrogen Atom Energy Levels

def generate_hydrogen(num_samples=2000, n_max=10,
                      noise_level=0.0, seed=42):
    """
    Generate energy level data for hydrogen atom.
    
    Input:   n (principal quantum number, n >= 1)
    Output:  E_n = -13.6 / n^2   (eV)
    
    Args:
        num_samples  : number of data points
        n_max        : maximum principal quantum number
        noise_level  : fractional Gaussian noise (0.0 = clean)
        seed         : random seed
    
    Returns:
        X : (num_samples, 1) array  [n]
        y : (num_samples,)   array  [E_n in eV]
    """
    rng = np.random.default_rng(seed)
    n   = rng.integers(1, n_max + 1, size=num_samples).astype(np.float64)
    
    # Ground truth equation
    E_n = -13.6 / (n ** 2)
    
    if noise_level > 0.0:
        noise = rng.normal(0, noise_level * np.abs(E_n))
        E_n   = E_n + noise
    
    X = n.reshape(-1, 1)
    y = E_n
    return X, y


#3. de Broglie Relation 

def generate_de_broglie(num_samples=2000, mass_range=(1e-31, 1e-27),
                        velocity_range=(1e3, 1e7),
                        noise_level=0.0, seed=42):
    """
    Generate wavelength data from de Broglie relation.
    
    Inputs:  m (mass), v (velocity)  →  p = m * v
    Output:  lambda = h / (m * v)
    
    Args:
        num_samples     : number of data points
        mass_range      : particle mass range (kg)
        velocity_range  : particle velocity range (m/s)
        noise_level     : fractional Gaussian noise
        seed            : random seed
    
    Returns:
        X : (num_samples, 2) array  [mass, velocity]
        y : (num_samples,)   array  [wavelength in metres]
    """
    rng      = np.random.default_rng(seed)
    mass     = rng.uniform(mass_range[0],     mass_range[1],     size=num_samples)
    velocity = rng.uniform(velocity_range[0], velocity_range[1], size=num_samples)
    
    # Ground truth equation
    momentum   = mass * velocity
    wavelength = H / momentum
    
    if noise_level > 0.0:
        noise      = rng.normal(0, noise_level * np.abs(wavelength))
        wavelength = wavelength + noise
    
    X = np.column_stack([mass, velocity])
    y = wavelength
    return X, y


# Save datasets to disk 

def save_dataset(X, y, name, folder='data/generated'):
    """Save X and y arrays as .npy files."""
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, f'{name}_X.npy'), X)
    np.save(os.path.join(folder, f'{name}_y.npy'), y)
    print(f'Saved {name}: X={X.shape}, y={y.shape}')


#  Noise sweep helper 

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]   # 0%, 1%, 5%, 10%

def generate_all(num_samples=2000, save=True):
    """
    Generate clean + noisy versions of all three quantum datasets.
    Returns a nested dict: datasets[system][noise_level] = (X, y)
    """
    datasets = {}

    systems = {
        'qho':        generate_qho,
        'hydrogen':   generate_hydrogen,
        'de_broglie': generate_de_broglie,
    }

    for name, fn in systems.items():
        datasets[name] = {}
        for noise in NOISE_LEVELS:
            X, y = fn(num_samples=num_samples, noise_level=noise, seed=42)
            datasets[name][noise] = (X, y)
            if save:
                tag = f'{name}_noise{int(noise*100):02d}pct'
                save_dataset(X, y, tag)

    return datasets


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating all quantum physics datasets...\n')
    datasets = generate_all(num_samples=2000, save=True)

    print('\nSanity checks:')

    # QHO: check first sample manually
    X_qho, y_qho = datasets['qho'][0.0]
    n0, w0 = X_qho[0]
    E0     = y_qho[0]
    E_calc = HBAR * w0 * (n0 + 0.5)
    print(f'  QHO   : n={n0:.0f}, omega={w0:.2e} → '
          f'E={E0:.4e}, expected={E_calc:.4e}, match={np.isclose(E0, E_calc)}')

    # Hydrogen: check first sample
    X_hyd, y_hyd = datasets['hydrogen'][0.0]
    n0   = X_hyd[0, 0]
    E0   = y_hyd[0]
    E_calc = -13.6 / (n0 ** 2)
    print(f'  H-atom: n={n0:.0f} → '
          f'E={E0:.4f} eV, expected={E_calc:.4f} eV, match={np.isclose(E0, E_calc)}')

    # de Broglie: check first sample
    X_db, y_db = datasets['de_broglie'][0.0]
    m0, v0 = X_db[0]
    lam0   = y_db[0]
    lam_calc = H / (m0 * v0)
    print(f'  dBrog : m={m0:.2e}, v={v0:.2e} → '
          f'λ={lam0:.4e} m, expected={lam_calc:.4e} m, match={np.isclose(lam0, lam_calc)}')

    print('\nAll datasets generated successfully.')
    print(f'Noise levels covered: {[f"{n*100:.0f}%" for n in NOISE_LEVELS]}')