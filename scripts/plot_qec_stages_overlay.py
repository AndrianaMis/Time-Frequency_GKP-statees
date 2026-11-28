"""
Stage 1: Clean Initial GKP State
Stage 2: After Optical Channel Noise (Displacement + Broadening)
Stage 3: After Round 1 (Frequency Correction + Width Re-narrowing)
Stage 4: After Round 2 (Time Correction + Complete State Restoration)

With dashed input curves for pre-operation states and solid curves for post-operation states
(following Nicolas's recommendation for publication-ready QEC visualization).
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def spectral_intensity(points, period, disp, env_width, signal_width, n_teeth):
    I = np.zeros_like(points)
    for k in range(-n_teeth, n_teeth + 1):
        center = k * period + disp
        env = np.exp(-(center**2) / (env_width**2))
        I += env * np.exp(-((points - center)**2) / (signal_width**2))
    I /= np.max(I)
    return I

def get_env(points, env_width):
    env = np.exp(-(points**2) / (env_width**2))
    return env


def load_config(config_path="qec_params.yaml"):
    """Loads configuration with fallback search paths."""
    search_paths = [
        config_path,
        os.path.join(os.path.dirname(__file__), config_path),
        os.path.join(os.path.dirname(__file__), "qec_params.yaml"),
        "TFstates/visualize/qec_params.yaml",
        "qec_params.yaml"
    ]
    for p in search_paths:
        if os.path.isfile(p):
            with open(p, "r") as f:
                return yaml.safe_load(f)
    print("Warning: qec_params.yaml not found. Using default parameters.")
    return {}



def generate_qec_stages_overlay_plot(save_path="stages_wigner_overlay.png", config_file="qec_params.yaml", plot_wigner=True):
    Omega = 2.0 * np.pi       # Grid comb spacing
    T = np.pi / Omega         # Time pulse spacing

    cfg = load_config(config_file)
    

    # Clean state 
    clean_cfg = cfg.get("clean_state", {})
    sigma_0 = clean_cfg.get("sigma_0", 0.22)
    kappa = clean_cfg.get("kappa", 14.0)
    sigma_t0 = (1.0 / kappa)
    print("σt: ", sigma_t0)
    T_env0 = (1.0 / sigma_0)
    print("T_env: ", T_env0)


    n_freq = int(np.ceil(2.2 * kappa / Omega))
    n_time = int(np.ceil(2.5 * T_env0 / T)) 
    n_teeth = max(n_freq, n_time) 
    print("teeth: ", n_teeth)


    # Noise parameter
    noise_cfg = cfg.get("optical_channel_noise", {})
    mu = noise_cfg.get("mu", 0.38 * Omega)
    tau = noise_cfg.get("tau", 0.30 * T)
    sigma_w_noise = noise_cfg.get("sigma_w_noise", 0.32)
    sigma_t_noise = noise_cfg.get("sigma_t_noise", 0.11)


    # Ancillas
    anc_cfg = cfg.get("ancilla_parameters", {})
    sigma_a_f = anc_cfg.get("sigma_a_w", 0.16)
    sigma_a_t = anc_cfg.get("sigma_a_t", 0.045)


    '''noise channel application'''
    # frequency broadening
    sigma_broad_f = np.sqrt(sigma_0**2 + sigma_w_noise**2)  # below equation 54 in overleaf
    # safe to say that the time width did the same 
    sigma_broad_t = np.sqrt(sigma_t0**2 + sigma_t_noise**2)


    '''round 1 - freq'''
    # ancilla addition with clean state
    # Post-Round 1 (Frequency corrected & narrowed) since freq width narrowed, time envelope wider. 
    sigma_f_r1 = np.sqrt((sigma_0**2 * sigma_a_f**2) / (sigma_0**2 + sigma_a_f**2))  # below equation 67 in overleaf
    T_env_r1 = 1.0 / sigma_f_r1  # >T_env0
    print("T_env_r1: ", T_env_r1)


    '''round 2 - time'''
    # Post-Round 2 (Time corrected & narrowed via parallel-resistor)
    sigma_t_r2 = np.sqrt((sigma_t0**2 * sigma_a_t**2) / (sigma_t0**2 + sigma_a_t**2))
    print("σ_efft: ", sigma_t_r2)

    # Plotting configuration
    plot_cfg = cfg.get("plotting", {})
    w_factor = plot_cfg.get("w_range_factor", 2.2 * (kappa / Omega))   
    t_factor = plot_cfg.get("t_range_factor", 1.6 * (T_env0 / T)) 

    # Wigner grid: allow wider range on x-axis (arrival time t/T) so combs decay smoothly
    wigner_factor_t = plot_cfg.get("wigner_factor_t", plot_cfg.get("wigner_factor", 6.5))
    wigner_factor_w = plot_cfg.get("wigner_factor_w", plot_cfg.get("w_range_factor", 4.5))
    res_1d = plot_cfg.get("resolution_1d", 2000)

    w_pts = np.linspace(-w_factor * Omega, w_factor * Omega, res_1d)
    t_pts = np.linspace(-t_factor * T, t_factor * T, res_1d)

    # Wigner meshgrid
    res_wign = plot_cfg.get("wigner_resolution", 220)
    t_wign = np.linspace(-wigner_factor_t * T, wigner_factor_t * T, res_wign)
    w_wign = np.linspace(-wigner_factor_w * Omega, wigner_factor_w * Omega, res_wign)   
    T_mesh, W_mesh = np.meshgrid(t_wign, w_wign)


    # =========================================================================
    # THE 4 QEC STAGES: PHYSICAL DEFINITIONS & PARAMETERS
    # With input references for dashed overlay
    # =========================================================================
    stages = [
        {
            "name": "Stage 1: Clean GKP State",
            "desc": r"Error-free input: teeth of width $\sigma_0$, pulses of duration $\sigma_{t,0}$",
            "mu": 0.0, "tau": 0.0,
            "sigma_w": sigma_0, "sigma_t": sigma_t0,
            "T_env": T_env0, "kappa": kappa,
            "color": "#1f77b4",
            "has_input": False
        },
        {
            "name": "Stage 2: After Channel Degradation",
            "desc": r"Noisy state: shifted by $(\tau, \mu)$, broadened by $\sigma_\omega, \sigma_t$",
            "mu": mu, "tau": tau,
            "sigma_w": sigma_broad_f,
            "sigma_t": sigma_broad_t,
            "T_env": T_env0, "kappa": kappa,
            "color": "#d62728",
            "has_input": True,
            # Input to channel was Stage 1 (clean state)
            "input_mu": 0.0, "input_tau": 0.0,
            "input_sigma_w": sigma_0, "input_sigma_t": sigma_t0,
            "input_T_env": T_env0,
            "input_label": r"Input (Clean St. 1)"
        },
        {
            "name": "Stage 3: After Round 1 (Freq Round)",
            "desc": r"$\mu \to 0$ cancelled, frequency re-narrowed ($\sigma_{\mathrm{eff}} < \sigma_0$), time anti-squeezed",
            "mu": 0.0, "tau": tau,
            "sigma_w": sigma_f_r1,
            "sigma_t": sigma_broad_t,
            "T_env": T_env_r1, "kappa": kappa,
            "color": "#9467bd",
            "has_input": True,
            # Input to Round 1 was Stage 2 (noisy state)
            "input_mu": mu, "input_tau": tau,
            "input_sigma_w": sigma_broad_f, "input_sigma_t": sigma_broad_t,
            "input_T_env": T_env0,
            "input_label": r"Input (Noisy St. 2)"
        },
        {
            "name": "Stage 4: After Round 2 (Time Round)",
            "desc": r"$\tau \to 0$ cancelled, pulses re-narrowed ($\sigma_{t,\mathrm{eff}} < \sigma_{t,0}$), restored state",
            "mu": 0.0, "tau": 0.0,
            "sigma_w": sigma_f_r1,
            "sigma_t": sigma_t_r2,
            "T_env": T_env_r1, "kappa": kappa,
            "color": "#2ca02c",
            "has_input": True,
            # Input to Round 2 was Stage 3 (post-frequency round)
            "input_mu": 0.0, "input_tau": tau,
            "input_sigma_w": sigma_f_r1, "input_sigma_t": sigma_broad_t,
            "input_T_env": T_env_r1,
            "input_label": r"Input (Post-R1 St. 3)"
        }
    ]

    def compute_exact_wigner(T_mesh, W_mesh, tau, mu, sigma_w, sigma_t, T_env, kappa, Omega, n_teeth=4):
        """
        Complete, closed-form chronocyclic Wigner function W(t, omega)
        modeling all physical parameters of each QEC stage.
        """
        W = np.zeros_like(T_mesh, dtype=complex)
        teeth = range(-n_teeth, n_teeth + 1)
        
        # Relative coordinates with temporal shift tau
        t_rel = T_mesh - tau
        
        for n in teeth:
            for m in teeth:
                c_n = (1.0 / np.sqrt(2.0)) * np.exp(-((n * Omega)**2) / (2.0 * kappa**2))
                c_m = (1.0 / np.sqrt(2.0)) * np.exp(-((m * Omega)**2) / (2.0 * kappa**2))
                
                w_center = 0.5 * (n + m) * Omega + mu
                delta_w = (n - m) * Omega
                dephasing = np.exp(-0.5 * (sigma_t**2) * (delta_w**2))
                freq_peak = np.exp(-((W_mesh - w_center)**2) / (sigma_w**2))
                fringe = np.exp(-1j * delta_w * t_rel)
                
                term = c_n * np.conj(c_m) * dephasing * freq_peak * fringe
                W += term
                
        env_t = np.exp(-(t_rel**2) / (T_env**2))
        return np.real(W) * env_t

    if plot_wigner == True:
        cols = 3
    else:
        cols = 2
    
    # Publication-ready typography: LaTeX Computer Modern serif, no bold, matching body/caption font size
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.8,
        "lines.linewidth": 1.4,
    })

    fig, axes = plt.subplots(4, cols, figsize=(10.8, 9.8), dpi=300)

    for idx, (st, row_axes) in enumerate(zip(stages, axes)):
        if plot_wigner == True:
            ax_f, ax_t, ax_w = row_axes
        else:
            ax_f, ax_t = row_axes

        # Compute output (current stage) profiles
        I_w = spectral_intensity(points=w_pts, period=Omega, disp=st["mu"], env_width=st["kappa"], signal_width=st["sigma_w"], n_teeth=n_teeth)
        I_t = spectral_intensity(points=t_pts, period=T, disp=st["tau"], env_width=st["T_env"], signal_width=st["sigma_t"], n_teeth=n_teeth)
        env_f = get_env(points=w_pts, env_width=st["kappa"])
        env_t = get_env(points=t_pts, env_width=st["T_env"])

        # Compute input profiles for dashed overlay (Nicolas's suggestion)
        if st.get("has_input", False):
            I_w_in = spectral_intensity(points=w_pts, period=Omega, disp=st["input_mu"], env_width=st["kappa"], signal_width=st["input_sigma_w"], n_teeth=n_teeth)
            I_t_in = spectral_intensity(points=t_pts, period=T, disp=st["input_tau"], env_width=st["input_T_env"], signal_width=st["input_sigma_t"], n_teeth=n_teeth)

        # -------------------------------------------------------------
        # 1. Frequency Domain
        # -------------------------------------------------------------
        # Draw input in dashed line if applicable
        if st.get("has_input", False):
            ax_f.plot(w_pts / Omega, I_w_in, color='gray', ls='--', lw=1.2, alpha=0.75, label=st["input_label"])

        # Draw output (current stage) in solid line
        ax_f.plot(w_pts / Omega, I_w, color=st["color"], lw=1.6, ls='-', label=r'Output')
        ax_f.plot(w_pts / Omega, env_f, color='darkorange', ls=':', lw=1.2, label=r'Envelope $\kappa$')
        ax_f.axvline(0, color='gray', ls='--', alpha=0.35)
        for n in range(-3, 4):
            ax_f.axvline(n, color='lightgray', ls=':', alpha=0.3)
        ax_f.set_xlim(-w_factor, w_factor)
        ax_f.set_ylim(-0.05, 1.15)
        ax_f.set_ylabel(r'Intensity $|\tilde{\psi}(\omega)|^2$')
        ax_f.set_title(f"{st['name']} – Frequency", pad=5)
        ax_f.grid(True, alpha=0.25)
        ax_f.legend(loc='upper right', framealpha=0.88)

        # -------------------------------------------------------------
        # 2. Time Domain
        # -------------------------------------------------------------
        # Draw input in dashed line if applicable
        if st.get("has_input", False):
            ax_t.plot(t_pts / T, I_t_in, color='gray', ls='--', lw=1.2, alpha=0.75, label=st["input_label"])

        # Draw output (current stage) in solid line
        ax_t.plot(t_pts / T, I_t, color=st["color"], lw=1.6, ls='-', label=r'Output')
        ax_t.plot(t_pts / T, env_t, color='darkorange', ls=':', lw=1.2, label=r'Envelope $T_{\mathrm{env}}$')
        ax_t.axvline(0, color='gray', ls='--', alpha=0.35)
        for k in range(-3, 4):
            ax_t.axvline(k, color='lightgray', ls=':', alpha=0.3)
        ax_t.set_xlim(-t_factor, t_factor)
        ax_t.set_ylim(-0.05, 1.15)
        ax_t.set_ylabel(r'Intensity $|\psi(t)|^2$')
        ax_t.set_title(f"{st['name']} – Time", pad=5)
        ax_t.grid(True, alpha=0.25)
        ax_t.legend(loc='upper right', framealpha=0.88)

        # -------------------------------------------------------------
        # 3. Wigner Function W(t, omega)
        # -------------------------------------------------------------
        if plot_wigner == True:
            W = compute_exact_wigner(
                T_mesh=T_mesh, 
                W_mesh=W_mesh, 
                tau=st["tau"], 
                mu=st["mu"], 
                sigma_w=st["sigma_w"], 
                sigma_t=st["sigma_t"], 
                T_env=st["T_env"],
                kappa=st["kappa"],
                Omega=Omega,
                n_teeth=6
            )
            vmax = np.max(np.abs(W))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

            cf = ax_w.contourf(T_mesh / T, W_mesh / Omega, W, 60, cmap='RdBu_r', norm=norm)
            ax_w.axhline(0, color='black', ls='--', lw=0.8, alpha=0.4)
            ax_w.axvline(0, color='black', ls='--', lw=0.8, alpha=0.4)
            for k in [-1, 1]:
                ax_w.axvline(k, color='gray', ls=':', lw=0.7, alpha=0.4)
                ax_w.axhline(k, color='gray', ls=':', lw=0.7, alpha=0.4)

            # Show input shift (dashed marker/arrow) and output shift (solid marker)
            if st.get("has_input", False):
                # If input had a non-zero shift, mark it with an open dashed circle
                if st["input_tau"] != 0.0 or st["input_mu"] != 0.0:
                    ax_w.scatter([st["input_tau"] / T], [st["input_mu"] / Omega], 
                                facecolors='none', edgecolors='gray', ls='--', s=32, lw=1.2, zorder=5, label=r'Input center')
                
                # If the stage performed a shift correction, show the correction transition arrow
                if (st["input_tau"] != st["tau"]) or (st["input_mu"] != st["mu"]):
                    ax_w.annotate('', xy=(st["tau"]/T, st["mu"]/Omega), 
                                xytext=(st["input_tau"]/T, st["input_mu"]/Omega),
                                arrowprops=dict(arrowstyle="->", color="black", lw=1.3, ls='-'))

            # Current state center marker
            if st["tau"] != 0.0 or st["mu"] != 0.0:
                ax_w.scatter([st["tau"] / T], [st["mu"] / Omega], color='gold',
                            edgecolor='black', s=28, zorder=6, label=r'Shift $(\tau, \mu)$')
                if not st.get("has_input", False):
                    ax_w.annotate('', xy=(st["tau"]/T, st["mu"]/Omega), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
            
            if st.get("has_input", False) or st["tau"] != 0.0 or st["mu"] != 0.0:
                ax_w.legend(loc='upper right', framealpha=0.9, fontsize=7.8)

            ax_w.set_xlim(-wigner_factor_t, wigner_factor_t)
            ax_w.set_ylim(-wigner_factor_w, wigner_factor_w)
            ax_w.set_xlabel(r'Arrival time $t / T$')
            ax_w.set_ylabel(r'Detuning $\omega / \Omega$')
            ax_w.set_title(rf"{st['name']} – Wigner $W(t, \omega)$", pad=5)
            cb = fig.colorbar(cf, ax=ax_w, fraction=0.046, pad=0.04)
            cb.set_label(r'$W(t, \omega)$', fontsize=8.5)

        if idx == 3:
            ax_f.set_xlabel(r'Frequency detuning $\omega / \Omega$')
            ax_t.set_xlabel(r'Arrival time $t / T$')

    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    pdf_path = save_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Generated QEC comparison figures with input overlays from '{config_file}':\n  - {os.path.abspath(save_path)}\n  - {os.path.abspath(pdf_path)}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "..", "figures")
    out_file = os.path.join(figures_dir, "stages_wigner_overlay.png") if os.path.isdir(figures_dir) else "stages_wigner_overlay.png"
    generate_qec_stages_overlay_plot(save_path=out_file, plot_wigner=True)
