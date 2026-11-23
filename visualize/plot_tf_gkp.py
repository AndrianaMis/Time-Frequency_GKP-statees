"""
Time-Frequency GKP States Visualization Script
==============================================
Plots general GKP logical qubit states:
    |\psi_L> = \alpha |0_L> + \beta |1_L>

Examples:
- Logical |0_L>: alpha = 1.0, beta = 0.0 (Only even teeth)
- Logical |1_L>: alpha = 0.0, beta = 1.0 (Only odd teeth)
- Logical |+_L>: alpha = 1/sqrt(2), beta = 1/sqrt(2) (All teeth present)

Author: Andriana Misailidi
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip

def generate_tf_gkp_plots(alpha=1.0 / np.sqrt(2), beta=1.0 / np.sqrt(2), 
                          save_path="tf_gkp_states_visualization.png"):
    """
    Plots the frequency, time, and 2D Wigner phase-space representation 
    of a physical GKP qubit in an arbitrary superposition \alpha |0_L> + \beta |1_L>.
    """
    #σ<<Ω<<κ
    Omega = 2.0 * np.pi       # Comb spacing / FSR (bar{omega})
    T = np.pi / Omega         # Fundamental pulse period in time

    sigma_d = 0.42              # Individual tooth linewidth (sigma << Omega)
    kappa = 14.0              # Global spectral envelope bandwidth (Omega << kappa)
    sigma_t = 1.0 / kappa     # Temporal pulse duration (sigma_t << T)
    broadening=0.3
    sigma_broad=sigma_d+broadening
    sigma_ancilla=0.4

    T_env = np.sqrt(2.0) / sigma_d # Macroscopic envelope duration in time

    mu = 0.6                # Frequency displacement error (within Voronoi cell: mu < Omega/2)
    tau = 0.12                # Time displacement error (within Voronoi cell: tau < T/2)

   
    omega_pts = np.linspace(-3.5 * Omega, 3.5 * Omega, 2500)
    t_pts = np.linspace(-3.5 * T, 3.5 * T, 2500)

    def psi_freq(w, sigma, shift_mu=0.0):
        """Computes spectral amplitude for superposition \alpha |0_L> + \beta |1_L>."""
        val = np.zeros_like(w, dtype=complex)
        for n in range(-8, 9):
            # Even n represents |0_L> weighted by alpha; Odd n represents |1_L> weighted by beta
            c_n = alpha if (n % 2 == 0) else beta
            center = n * Omega + shift_mu 
            # + shifted by mu
            envelope = np.exp(-(center**2) / (2.0 * kappa**2)) #before the large comb approximation
            val += c_n * envelope * np.exp(-((w - center)**2) / (2.0 * sigma**2))
        return val

    def psi_time(t, shift_tau=0.0):
        """Computes temporal amplitude for the GKP state."""
        val = np.zeros_like(t, dtype=complex)
        for k in range(-10, 11):
            # Dual coefficients in time basis
            d_k = (alpha + ((-1)**k) * beta) / np.sqrt(2.0)
            center = k * T + shift_tau
            envelope = np.exp(-(center**2) / (2.0 * T_env**2))
            val += d_k * envelope * np.exp(-((t - center)**2) / (2.0 * sigma_t**2))
        return val

    # -------------------------------------------------------------
    # 4. PLOTTING SETUP
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(15, 11), dpi=300)
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9.5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    # Label for the state
    if np.isclose(alpha, 1.0) and np.isclose(beta, 0.0):
        state_label = r'$|\bar{0}_L\rangle$'
    elif np.isclose(alpha, 0.0) and np.isclose(beta, 1.0):
        state_label = r'$|\bar{1}_L\rangle$'
    elif np.isclose(alpha, 1.0/np.sqrt(2)) and np.isclose(beta, 1.0/np.sqrt(2)):
        state_label = r'$|\bar{+}_L\rangle$'
    else:
        state_label = rf'$\alpha|\bar{{0}}_L\rangle + \beta|\bar{{1}}_L\rangle$'

    # --- Panel (a): Frequency Spectrum with Spectral Envelope ---
    ax1 = plt.subplot(2, 2, 1)
    clean_f = np.abs(psi_freq(omega_pts, shift_mu=0.0, sigma=sigma_d))**2
    noisy_f = np.abs(psi_freq(omega_pts, shift_mu=mu, sigma=sigma_broad))**2
    ancilla_f = np.abs(psi_freq(omega_pts, shift_mu=0.0, sigma=sigma_ancilla))**2
    norm_f = np.max(clean_f)
    
    # Global Spectral Envelope (sets sigma_t in time)
    env_freq = np.exp(-(omega_pts**2) / (kappa**2))

    ax1.plot(omega_pts / Omega, clean_f / norm_f, color='#1f77b4', lw=2.2, label=rf'Clean {state_label} (Centered at $n\Omega$)')
    ax1.plot(omega_pts / Omega, noisy_f / norm_f, color='#d62728', ls='--', lw=1.8, label=rf'Displaced by $\mu$ and broadened ($\mu/\Omega = {mu/Omega:.2f}$, $\sigma/\Omega = {sigma_broad/Omega:.2f}$)')
    #ax1.plot(omega_pts / Omega, ancilla_f / norm_f, color='#2ca02c', ls='-.', lw=1.8, label=rf'Ancilla State (Centered at $n\Omega$ with $\sigma/\Omega = {sigma_ancilla/Omega:.2f}$)')
    ax1.plot(omega_pts / Omega, env_freq, color='#ff7f0e', ls=':', lw=2.2, label=r'Spectral Envelope $\kappa$ (sets $\sigma_t = 1/\kappa$ in time)')
    ax1.axvline(0, color='gray', ls=':', alpha=0.5)
    for n in range(-3, 4):
        ax1.axvline(n, color='lightgray', ls='--', alpha=0.4, zorder=0)
    ax1.set_xlabel(r'Frequency $\omega / \Omega$', fontweight='bold')
    ax1.set_ylabel(r'Spectral Intensity $|\tilde{\psi}(\omega)|^2$', fontweight='bold')
    ax1.set_title(r'(a) Frequency Representation ($\sigma \ll \Omega \ll \kappa$)', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.92)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-0.05, 1.15)

    # --- Panel (b): Temporal Pulse Train with Macroscopic Envelope ---
    ax2 = plt.subplot(2, 2, 2)
    clean_t = np.abs(psi_time(t_pts, shift_tau=0.0))**2
    noisy_t = np.abs(psi_time(t_pts, shift_tau=tau))**2
    norm_t = np.max(clean_t)

    # Macroscopic Temporal Envelope (sets sigma in frequency)
    env_time = np.exp(-(t_pts**2) / (T_env**2))

    ax2.plot(t_pts / T, clean_t / norm_t, color='#1f77b4', lw=2.2, label=rf'Clean Pulse Train ({state_label})')
    ax2.plot(t_pts / T, noisy_t / norm_t, color='#d62728', ls='--', lw=1.8, label=rf'Displaced by $\tau$ ($\tau/T = {tau/T:.2f}$)')
    ax2.plot(t_pts / T, env_time, color='#ff7f0e', ls=':', lw=2.2, label=r'Temporal Envelope $T_{\mathrm{env}} = \sqrt{2}/\sigma$ (sets $\sigma$ in freq)')
    ax2.axvline(0, color='gray', ls=':', alpha=0.5)
    for k in range(-3, 4):
        ax2.axvline(k, color='lightgray', ls='--', alpha=0.4, zorder=0)
    ax2.set_xlabel(r'Arrival Time $t / T$', fontweight='bold')
    ax2.set_ylabel(r'Temporal Intensity $|\psi(t)|^2$', fontweight='bold')
    ax2.set_title(r'(b) Time Representation ($\sigma_t \ll T \ll T_{\text{envelope}}$)', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.92)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-0.05, 1.15)

    # # # --- Panel (c): 2D Wigner Function (Clean State) ---
    # ax3 = plt.subplot(2, 2, 3)
    # wig_clean = qutip.wigner(qutip.Qobj(psi_freq(omega_pts, shift_mu=0.0, sigma=sigma_d)), t_grid, w_grid)
    # cf3 = ax3.contourf(T_mat / T, W_mat / Omega, W_clean, 60, cmap='RdBu_r')
    # ax3.set_xlabel(r'Arrival Time $t / T$', fontweight='bold')
    # ax3.set_ylabel(r'Detuning Frequency $\omega / \Omega$', fontweight='bold')
    # ax3.set_title(rf'(c) Clean Wigner Function {state_label}', fontweight='bold')
    # ax3.axhline(0, color='black', ls='--', alpha=0.35)
    # ax3.axvline(0, color='black', ls='--', alpha=0.35)
    # cb3 = fig.colorbar(cf3, ax=ax3, fraction=0.046, pad=0.04)
    # cb3.set_label(r'$W(t, \omega)$', fontweight='bold')

    # # # --- Panel (d): 2D Wigner Function (Shifted State under Channel) ---
    # ax4 = plt.subplot(2, 2, 4)
    # wig_noisy = qutip.wigner(qutip.Qobj(psi_freq(omega_pts, shift_mu=mu, sigma=sigma_broad)), t_grid, w_grid)
    # cf4 = ax4.contourf(T_mat / T, W_mat / Omega, W_noisy, 60, cmap='RdBu_r')
    # ax4.scatter([tau / T], [mu / Omega], color='#ffd700', edgecolor='black', s=90, zorder=5, label=rf'Shift $(\tau, \mu)$')
    # ax4.set_xlabel(r'Arrival Time $t / T$', fontweight='bold')
    # ax4.set_ylabel(r'Detuning Frequency $\omega / \Omega$', fontweight='bold')
    # ax4.set_title(rf'(d) Displaced Wigner $W(t-\tau, \omega-\mu)$', fontweight='bold')
    # ax4.axhline(0, color='black', ls='--', alpha=0.35)
    # ax4.axvline(0, color='black', ls='--', alpha=0.35)
    # ax4.legend(loc='upper right', framealpha=0.9)
    # cb4 = fig.colorbar(cf4, ax=ax4, fraction=0.046, pad=0.04)
    # cb4.set_label(r'$W(t-\tau, \omega-\mu)$', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Generated plots with envelopes for state: {state_label}")
    print(f"Saved to: {save_path} ")

if __name__ == "__main__":
    generate_tf_gkp_plots(alpha=1.0 / np.sqrt(2), beta=1.0 / np.sqrt(2))
