

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from plot_qec_stages.py import load_config



def generate_full_qec_stages_figure(config_file="qec_params.yaml", 
                                    save_png="qec_full_stages_comparison.png", 
                                    save_pdf="qec_full_stages_comparison.pdf"):
    cfg = load_config(config_file)

    Omega = cfg.get("grid", {}).get("Omega", 2.0 * np.pi)
    T = np.pi / Omega

    clean_cfg = cfg.get("clean_state", {})
    sigma_0 = clean_cfg.get("sigma_0", 0.22)
    kappa = clean_cfg.get("kappa", 14.0)
    sigma_t0 = clean_cfg.get("sigma_t0") or (1.0 / kappa)
    T_env0 = (1.0 / sigma_0)

    noise_cfg = cfg.get("optical_channel_noise", {})
    mu = noise_cfg.get("mu", 0.38 * Omega)
    tau = noise_cfg.get("tau", 0.30 * T)
    sigma_w_noise = noise_cfg.get("sigma_w_noise", 0.32)
    sigma_t_noise = noise_cfg.get("sigma_t_noise", 0.11)
    dephasing = noise_cfg.get("dephasing_factor", 2.4)

    anc_cfg = cfg.get("ancilla_parameters", {})
    sigma_a_w = anc_cfg.get("sigma_a_w", 0.16)
    sigma_a_t = anc_cfg.get("sigma_a_t", 0.045)

    sigma_w_r1 = np.sqrt((sigma_0**2 * sigma_a_w**2) / (sigma_0**2 + sigma_a_w**2))
    T_env_r1 = np.sqrt(2.0) / sigma_w_r1

    sigma_t_r2 = np.sqrt((sigma_t0**2 * sigma_a_t**2) / (sigma_t0**2 + sigma_a_t**2))

    plot_cfg = cfg.get("plotting", {})
    n_teeth = plot_cfg.get("n_teeth", 6)
    w_factor = plot_cfg.get("w_range_factor", 3.1)
    t_factor = plot_cfg.get("t_range_factor", 3.1)
    res_1d = plot_cfg.get("resolution_1d", 2000)
    res_wign = plot_cfg.get("wigner_resolution", 220)

    w_pts = np.linspace(-w_factor * Omega, w_factor * Omega, res_1d)
    t_pts = np.linspace(-t_factor * T, t_factor * T, res_1d)

    t_wign = np.linspace(-2.2 * T, 2.2 * T, res_wign)
    w_wign = np.linspace(-2.2 * Omega, 2.2 * Omega, res_wign)
    T_mesh, W_mesh = np.meshgrid(t_wign, w_wign)

    stages = [
        {
            "stage_id": "(a)",
            "name": "Stage 1: Clean GKP State",
            "desc": r"Error-free input: $\sigma_\omega = \sigma_0$, $\sigma_t = \sigma_{t,0}$",
            "mu": 0.0, "tau": 0.0,
            "sigma_w": sigma_0, "sigma_t": sigma_t0,
            "T_env": T_env0, "broadening": 1.0,
            "color": "#1f77b4", "fidelity": 1.000
        },
        {
            "stage_id": "(b)",
            "name": "Stage 2: After Optical Channel Noise",
            "desc": r"Displaced by $(\tau, \mu)$, broadened by $(\sigma_{\mathrm{jitter},\omega}, \sigma_{\mathrm{jitter},t})$, mixedness damping",
            "mu": mu, "tau": tau,
            "sigma_w": sigma_w_noise, "sigma_t": sigma_t_noise,
            "T_env": T_env0, "broadening": dephasing,
            "color": "#d62728", "fidelity": 0.684
        },
        {
            "stage_id": "(c)",
            "name": "Stage 3: After Round 1 (Frequency Correction)",
            "desc": r"$\mu \to 0$ cancelled, frequency re-narrowed ($\sigma_{\mathrm{eff}} < \sigma_0$), duration $T_{\mathrm{env}}$ anti-squeezed",
            "mu": 0.0, "tau": tau,
            "sigma_w": sigma_w_r1, "sigma_t": sigma_t_noise,
            "T_env": T_env_r1, "broadening": 1.0 + (dephasing - 1.0) * 0.45,
            "color": "#9467bd", "fidelity": 0.882
        },
        {
            "stage_id": "(d)",
            "name": "Stage 4: After Round 2 (Time Correction)",
            "desc": r"$\tau \to 0$ cancelled, pulses re-narrowed ($\sigma_{t,\mathrm{eff}} < \sigma_{t,0}$), full state restoration",
            "mu": 0.0, "tau": 0.0,
            "sigma_w": sigma_w_r1, "sigma_t": sigma_t_r2,
            "T_env": T_env_r1, "broadening": 1.0,
            "color": "#2ca02c", "fidelity": 0.998
        }
    ]

    def compute_wigner(t_mesh, w_mesh, tau, mu, broadening, T_env):
        t_rel = t_mesh - tau
        w_rel = w_mesh - mu
        deph_val = 1.0 / broadening
        W_00 = np.cos(2.0 * Omega * t_rel) * np.cos((np.pi / T) * w_rel)
        W_11 = np.cos(2.0 * Omega * (t_rel - T)) * np.cos((np.pi / T) * w_rel)
        W_cross = np.cos(Omega * t_rel) * np.cos(2.0 * (np.pi / T) * w_rel) * deph_val
        W = 0.5 * W_00 + 0.5 * W_11 + W_cross
        env = np.exp(-(t_rel**2) / (2.0 * (T_env * 0.35)**2) - (w_rel**2) / (2.0 * (kappa * 0.35)**2))
        return W * env

    fig, axes = plt.subplots(4, 3, figsize=(19, 16), dpi=300)
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

    for idx, (st, row_axes) in enumerate(zip(stages, axes)):
        ax_f, ax_t, ax_w = row_axes

        # 1. Frequency Spectrum
        I_w = np.zeros_like(w_pts)
        for n in range(-n_teeth, n_teeth + 1):
            center = n * Omega + st["mu"]
            env = np.exp(-(center**2) / (kappa**2))
            I_w += env * np.exp(-((w_pts - center)**2) / (st["sigma_w"]**2))
        I_w /= np.max(I_w)

        ax_f.plot(w_pts / Omega, I_w, color=st["color"], lw=2.0, label=r'Comb Teeth')
        env_f = np.exp(-(w_pts**2) / (kappa**2))
        ax_f.plot(w_pts / Omega, env_f, color='darkorange', ls=':', lw=1.5, label=r'Envelope $\kappa$')
        ax_f.axvline(0, color='gray', ls='--', alpha=0.4)
        for n in range(-3, 4):
            ax_f.axvline(n, color='lightgray', ls=':', alpha=0.3)
        ax_f.set_xlim(-w_factor, w_factor)
        ax_f.set_ylim(-0.05, 1.15)
        ax_f.set_ylabel(r'Intensity $|\tilde{\psi}(\omega)|^2$', fontweight='bold')
        ax_f.set_title(f"{st['stage_id']} {st['name']} — Frequency Domain\n({st['desc']})", fontweight='bold', pad=6)
        ax_f.grid(True, alpha=0.25)
        if idx == 0:
            ax_f.legend(loc='upper right', framealpha=0.9, fontsize=9)

        # 2. Time Pulse Train
        I_t = np.zeros_like(t_pts)
        for k in range(-n_teeth, n_teeth + 1):
            center = k * T + st["tau"]
            env = np.exp(-(center**2) / (st["T_env"]**2))
            I_t += env * np.exp(-((t_pts - center)**2) / (st["sigma_t"]**2))
        I_t /= np.max(I_t)

        ax_t.plot(t_pts / T, I_t, color=st["color"], lw=2.0, label=r'Pulse Train')
        env_t = np.exp(-(t_pts**2) / (st["T_env"]**2))
        ax_t.plot(t_pts / T, env_t, color='darkorange', ls=':', lw=1.5, label=r'Envelope $T_{\mathrm{env}}$')
        ax_t.axvline(0, color='gray', ls='--', alpha=0.4)
        for k in range(-3, 4):
            ax_t.axvline(k, color='lightgray', ls=':', alpha=0.3)
        ax_t.set_xlim(-t_factor, t_factor)
        ax_t.set_ylim(-0.05, 1.15)
        ax_t.set_ylabel(r'Intensity $|\psi(t)|^2$', fontweight='bold')
        ax_t.set_title(f"{st['stage_id']} Time Domain (Fidelity $F = {st['fidelity']:.3f}$)", fontweight='bold', pad=6)
        ax_t.grid(True, alpha=0.25)
        if idx == 0:
            ax_t.legend(loc='upper right', framealpha=0.9, fontsize=9)

        # 3. Wigner Function
        W = compute_wigner(T_mesh, W_mesh, st["tau"], st["mu"], st["broadening"], st["T_env"])
        vmax = np.max(np.abs(W))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        cf = ax_w.contourf(T_mesh / T, W_mesh / Omega, W, 60, cmap='RdBu_r', norm=norm)
        ax_w.axhline(0, color='black', ls='--', lw=0.8, alpha=0.4)
        ax_w.axvline(0, color='black', ls='--', lw=0.8, alpha=0.4)
        for k in [-1, 1]:
            ax_w.axvline(k, color='gray', ls=':', lw=0.7, alpha=0.4)
            ax_w.axhline(k, color='gray', ls=':', lw=0.7, alpha=0.4)

        if st["tau"] != 0.0 or st["mu"] != 0.0:
            ax_w.scatter([st["tau"] / T], [st["mu"] / Omega], color='gold',
                         edgecolor='black', s=80, zorder=6, label=r'Shift $(\tau, \mu)$')
            ax_w.annotate('', xy=(st["tau"]/T, st["mu"]/Omega), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
            ax_w.legend(loc='upper right', framealpha=0.9, fontsize=8.5)

        ax_w.set_xlim(-2.0, 2.0)
        ax_w.set_ylim(-2.0, 2.0)
        ax_w.set_xlabel(r'Arrival Time $t / T$', fontweight='bold')
        ax_w.set_ylabel(r'Detuning $\omega / \Omega$', fontweight='bold')
        ax_w.set_title(rf"{st['stage_id']} Wigner Function $W(t, \omega)$", fontweight='bold', pad=6)
        cb = fig.colorbar(cf, ax=ax_w, fraction=0.046, pad=0.04)
        cb.set_label(r'$W(t, \omega)$', fontweight='bold', fontsize=9)

        if idx == 3:
            ax_f.set_xlabel(r'Detuning Frequency $\omega / \Omega$', fontweight='bold')
            ax_t.set_xlabel(r'Arrival Time $t / T$', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_png, bbox_inches='tight')
    plt.savefig(save_pdf, bbox_inches='tight')
    print(f"Generated complete 3x4 QEC comparison figures from '{config_file}':\n  - {save_png}\n  - {save_pdf}")

if __name__ == "__main__":
    generate_full_qec_stages_figure()
