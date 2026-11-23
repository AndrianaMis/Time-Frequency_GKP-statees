"""
Time-Frequency GKP Wigner Function Visualizer
=============================================
Generates publication-quality 2D chronocyclic Wigner functions W(t, \omega)
for single-photon time-frequency GKP states:
1. Logical |0_L> (Peaks on even grid sites)
2. Logical |1_L> (Shifted grid sites)
3. Logical |+_L> = (|0_L> + |1_L>)/sqrt(2) (Rich quantum interference & Wigner negativity)
4. Noisy / Displaced State W(t - tau, \omega - mu) under the displacement channel

Author: Andriana Misailidi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def compute_gkp_wigner(t_mesh, w_mesh, alpha=1.0/np.sqrt(2), beta=1.0/np.sqrt(2), 
                       shift_t=0.0, shift_w=0.0, broadening=1.0,
                       Omega=2.0*np.pi, sigma=0.22, kappa=14.0):
    """
    Computes the 2D chronocyclic Wigner function W(t, omega)
    for an arbitrary superposition alpha |0_L> + beta |1_L>.
    """
    T = np.pi / Omega
    T_env = np.sqrt(2.0) / sigma
    
    t_rel = t_mesh - shift_t
    w_rel = w_mesh - shift_w
    
    # Intrinsic tooth broadening factor (broadening >= 1.0)
    # When noise broadens teeth, the higher-frequency interference fringes damp out
    dephasing_factor = 1.0 / broadening
    
    # 1. Self-interference term (grid lattice of |0_L> and |1_L>)
    W_00 = np.cos(2.0 * Omega * t_rel) * np.cos((np.pi / T) * w_rel)
    W_11 = np.cos(2.0 * Omega * (t_rel - T)) * np.cos((np.pi / T) * w_rel)
    
    # 2. Cross-interference term (quantum interference fringes)
    # Placed halfway between grid sites; responsible for Wigner negativity in |+_L>
    W_cross = np.cos(Omega * t_rel) * np.cos(2.0 * (np.pi / T) * w_rel) * dephasing_factor
    
    # 3. Superposition weighting
    W = (np.abs(alpha)**2 * W_00 + 
         np.abs(beta)**2 * W_11 + 
         2.0 * np.real(alpha * np.conj(beta)) * W_cross)
    
    # 4. Macroscopic finite-energy Gaussian envelope
    env = np.exp(-(t_rel**2) / (2.0 * (T_env * 0.35)**2) - (w_rel**2) / (2.0 * (kappa * 0.35)**2))
    
    return W * env

def generate_wigner_gallery(save_path="tf_gkp_wigner_gallery.png"):
    Omega = 2.0 * np.pi
    T = np.pi / Omega
    
    # High-resolution phase-space grid
    t_pts = np.linspace(-2.2 * T, 2.2 * T, 350)
    w_pts = np.linspace(-2.2 * Omega, 2.2 * Omega, 350)
    T_mesh, W_mesh = np.meshgrid(t_pts, w_pts)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13
    })
    
    states = [
        {"title": r"(a) Logical $|\bar{0}_L\rangle$ (Even Grid)", 
         "alpha": 1.0, "beta": 0.0, "tau": 0.0, "mu": 0.0, "broad": 1.0},
        {"title": r"(b) Logical $|\bar{1}_L\rangle$ (Odd Grid, Shifted by $T$)", 
         "alpha": 0.0, "beta": 1.0, "tau": 0.0, "mu": 0.0, "broad": 1.0},
        {"title": r"(c) Superposition $|\bar{+}_L\rangle$ (Quantum Negativity)", 
         "alpha": 1.0/np.sqrt(2), "beta": 1.0/np.sqrt(2), "tau": 0.0, "mu": 0.0, "broad": 1.0},
        {"title": r"(d) Noisy State (Displaced by $(\tau, \mu)$ + Broadened)", 
         "alpha": 1.0/np.sqrt(2), "beta": 1.0/np.sqrt(2), "tau": 0.18*T, "mu": 0.22*Omega, "broad": 1.45}
    ]
    
    for idx, (ax, st) in enumerate(zip(axes.flatten(), states)):
        W = compute_gkp_wigner(
            T_mesh, W_mesh, 
            alpha=st["alpha"], beta=st["beta"],
            shift_t=st["tau"], shift_w=st["mu"],
            broadening=st["broad"], Omega=Omega
        )
        
        # Diverging colormap centered strictly at zero (Blue = negative, Red = positive)
        vmax = np.max(np.abs(W))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        
        cf = ax.contourf(T_mesh / T, W_mesh / Omega, W, 70, cmap='RdBu_r', norm=norm)
        
        # Grid lines marking fundamental Voronoi cell boundaries
        ax.axhline(0, color='black', ls='--', lw=0.9, alpha=0.4)
        ax.axvline(0, color='black', ls='--', lw=0.9, alpha=0.4)
        for k in [-1, 1]:
            ax.axvline(k, color='gray', ls=':', lw=0.8, alpha=0.5)
            ax.axhline(k, color='gray', ls=':', lw=0.8, alpha=0.5)
            
        # For the noisy panel, plot displacement vector
        if st["tau"] != 0.0 or st["mu"] != 0.0:
            ax.scatter([st["tau"] / T], [st["mu"] / Omega], color='gold', 
                       edgecolor='black', s=100, zorder=6, label=r'Shift $(\tau, \mu)$')
            ax.annotate('', xy=(st["tau"]/T, st["mu"]/Omega), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2.0))
            ax.legend(loc='upper right', framealpha=0.9)
            
        ax.set_title(st["title"], fontweight='bold', pad=10)
        ax.set_xlabel(r'Arrival Time $t / T$', fontweight='bold')
        ax.set_ylabel(r'Detuning Frequency $\omega / \Omega$', fontweight='bold')
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$W(t, \omega)$', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    pdf_path = save_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Successfully saved Wigner gallery to:\n  - {save_path}\n  - {pdf_path}")

if __name__ == "__main__":
    generate_wigner_gallery()
