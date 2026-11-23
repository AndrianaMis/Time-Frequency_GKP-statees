"""
Time-Frequency GKP Quantum Error Correction (CV-QEC) Dynamic Animation
=====================================================================
YAML Config-Driven Version:
Reads physical and simulation parameters from `qec_params.yaml`.

Authors: Andriana Misailidi & DeepMind Antigravity
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation
from plot_qec_stages import load_config


def create_qec_animation(config_file="qec_params.yaml", output_gif="figures/qec_process_animation_exact_wigner.gif"):
    cfg = load_config(config_file)
    print(f"Loaded parameters from: {config_file}")

    Omega = Omega = 2.0 * np.pi
    T = np.pi / Omega

    clean_cfg = cfg.get("clean_state", {})
    sigma_0 = clean_cfg.get("sigma_0", 0.22)
    kappa = clean_cfg.get("kappa", 14.0)
    sigma_t0 =  (1.0 / kappa)
    T_env0 = (np.sqrt(2.0) / sigma_0)

    noise_cfg = cfg.get("optical_channel_noise", {})
    mu_noise = noise_cfg.get("mu", 0.38 * Omega)
    tau_noise = noise_cfg.get("tau", 0.30 * T)
    sigma_w_noise = noise_cfg.get("sigma_w_noise", 0.32)
    sigma_t_noise = noise_cfg.get("sigma_t_noise", 0.11)
    dephasing_max = noise_cfg.get("dephasing_factor", 2.5)

    # Broadened widths after channel noise (matching plot_qec_stages.py)
    sigma_w_max = np.sqrt(sigma_0**2 + sigma_w_noise**2)
    sigma_t_max = np.sqrt(sigma_t0**2 + sigma_t_noise**2)

    anc_cfg = cfg.get("ancilla_parameters", {})
    sigma_a_w = anc_cfg.get("sigma_a_w", 0.16)
    sigma_a_t = anc_cfg.get("sigma_a_t", 0.045)

    # Final re-narrowed widths after Round 1 & Round 2
    sigma_w_final = np.sqrt((sigma_0**2 * sigma_a_w**2) / (sigma_0**2 + sigma_a_w**2))
    sigma_t_final = np.sqrt((sigma_t0**2 * sigma_a_t**2) / (sigma_t0**2 + sigma_a_t**2))
    T_env_final = np.sqrt(2.0) / sigma_w_final

    anim_cfg = cfg.get("animation", {})
    fps = anim_cfg.get("fps", 15)
    dpi = anim_cfg.get("dpi", 90)
    n_frames = anim_cfg.get("total_frames", 140)

    plot_cfg = cfg.get("plotting", {})
    n_freq = int(np.ceil(2.2 * kappa / Omega))
    n_time = int(np.ceil(1.8 * T_env0 / T)) 
    n_teeth = max(n_freq, n_time) 
    print("teeth: ", n_teeth)
    w_factor = 2.2 * (kappa / Omega)
    t_factor = 1.6 * (T_env0 / T)

    w_pts = np.linspace(-w_factor * Omega, w_factor * Omega, 800)
    t_pts = np.linspace(-t_factor * T, t_factor * T, 800)

    res_wign = plot_cfg.get("wigner_resolution", 220)
    wigner_factor=plot_cfg.get("wigner_factor", 5)
    t_wign = np.linspace(-wigner_factor * T, wigner_factor * T, res_wign)
    w_wign = np.linspace(-wigner_factor * Omega, wigner_factor * Omega, res_wign) 
    T_mesh, W_mesh = np.meshgrid(t_wign, w_wign)

    def get_frame_state(frame):
        def smooth(p):
            return 0.5 * (1.0 - np.cos(np.pi * np.clip(p, 0.0, 1.0)))

        if frame <= 15:
            stage_name = "Stage 1: Clean GKP State"
            step_desc = "Initial logical state: ideal grid alignment, sharp peaks, deep Wigner negativity"
            mu = 0.0
            tau = 0.0
            sig_w = sigma_0
            sig_t = sigma_t0
            T_env = T_env0
         #   broadening = 1.0
           # fidelity = 1.00
            active_color = "#1f77b4"

        elif frame <= 42:
            p = smooth((frame - 15) / 27.0)
            stage_name = "Optical Channel Degradation (Incoherent Noise)"
            step_desc = f"Random walk shift (tau, mu) + tooth broadening + coherence loss ({int(p*100)}%)"
            mu = p * mu_noise
            tau = p * tau_noise
            sig_w = sigma_0 + p * (sigma_w_max - sigma_0)
            sig_t = sigma_t0 + p * (sigma_t_max - sigma_t0)
            T_env = T_env0
         #   broadening = 1.0 + p * (dephasing_max - 1.0)
          #  fidelity = 1.00 - p * 0.32
            active_color = "#d62728"

        elif frame <= 55:
            stage_name = "Stage 2: Corrupted Noisy State"
            step_desc = "Displaced, broadened peaks, washed-out off-diagonal coherences"
            mu = mu_noise
            tau = tau_noise
            sig_w = sigma_w_max
            sig_t = sigma_t_max
            T_env = T_env0
          #  broadening = dephasing_max
          #  fidelity = 0.68
            active_color = "#d62728"

        elif frame <= 82:
            p = smooth((frame - 55) / 27.0)
            stage_name = "Round 1: Frequency Error Correction"
            step_desc = f"SUM gate + syndrome kick (-mu -> 0), teeth re-narrowing ({int(p*100)}%)"
            mu = (1.0 - p) * mu_noise
            tau = tau_noise
            sig_w = sigma_w_max - p * (sigma_w_max - sigma_w_final)
            sig_t = sigma_t_max
            T_env = T_env0 + p * (T_env_final - T_env0) #changes little by little
            active_color = "#9467bd"

        elif frame <= 95:
            stage_name = "Stage 3: Post-Round 1 (Frequency Stabilized)"
            step_desc = "mu = 0 cancelled, sigma_w < sigma_0 re-narrowed, time jitter tau awaits Round 2"
            mu = 0.0
            tau = tau_noise
            sig_w = sigma_w_final
            sig_t = sigma_t_max  #fully noise
            T_env = T_env_final  #fully noisy
            active_color = "#9467bd"

        elif frame <= 122:
            p = smooth((frame - 95) / 27.0)
            stage_name = "Round 2: Time Error Correction"
            step_desc = f"SUM gate + time detection + delay kick (-tau -> 0), pulse duration re-narrowing ({int(p*100)}%)"
            mu = 0.0
            tau = (1.0 - p) * tau_noise
            sig_w = sigma_w_final
            sig_t = sigma_t_max - p * (sigma_t_max - sigma_t_final)
            T_env = T_env_final
            active_color = "#2ca02c"

        else:
            stage_name = "Stage 4: Fully Corrected & Stabilized GKP State"
            step_desc = "Both shifts cancelled, tooth widths narrower than initial input "
            mu = 0.0
            tau = 0.0
            sig_w = sigma_w_final
            sig_t = sigma_t_final
            T_env = T_env_final
            active_color = "#2ca02c"

        return {
            "stage_name": stage_name, "step_desc": step_desc,
            "mu": mu, "tau": tau, "sig_w": sig_w, "sig_t": sig_t,
            "T_env": T_env,  
            "color": active_color
        }

    fig = plt.figure(figsize=(15, 5.2), dpi=dpi)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.2], wspace=0.28)

    ax_f = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[0, 2])
    fig.patch.set_facecolor('#ffffff')

    # def compute_wigner(tau, mu, broadening, T_env):
    #     t_rel = T_mesh - tau
    #     w_rel = W_mesh - mu
    #     deph_val = 1.0 / broadening
    #     W_00 = np.cos(2.0 * Omega * t_rel) * np.cos((np.pi / T) * w_rel)
    #     W_11 = np.cos(2.0 * Omega * (t_rel - T)) * np.cos((np.pi / T) * w_rel)
    #     W_cross = np.cos(Omega * t_rel) * np.cos(2.0 * (np.pi / T) * w_rel) * deph_val
    #     W = 0.5 * W_00 + 0.5 * W_11 + W_cross
    #     env = np.exp(-(t_rel**2) / (2.0 * (T_env * 0.35)**2) - (w_rel**2) / (2.0 * (kappa * 0.35)**2))
    #     return W * env

    def compute_exact_wigner(T_mesh, W_mesh, tau, mu, sigma_w, sigma_t, T_env, kappa, Omega, n_teeth=4):
        """
        Complete, closed-form chronocyclic Wigner function W(t, omega)
        modeling all physical parameters of each QEC stage:
        
            W(t, w) = exp(-(t-tau)^2 / T_env^2)
                    * sum_{n,m} c_n c_m* exp(-((w - mu - (n+m)/2 * Omega)^2) / sigma_w^2)
                                        * exp(-0.5 * sigma_t^2 * (n-m)^2 * Omega^2)
                                        * exp(-i * (n-m)*Omega * (t - tau))
        
        Parameters:
            tau      : arrival time displacement (tau in St.2,3; 0 in St.1,4)
            mu       : frequency displacement (mu in St.2; 0 in St.1,3,4)
            sigma_w  : comb tooth linewidth in frequency (sigma_0 -> sigma_broad_f -> sigma_f_eff)
            sigma_t  : temporal pulse width / timing jitter (sigma_t0 -> sigma_broad_t -> sigma_t_eff)
            T_env    : global envelope duration in time (T_env0 -> T_env_r1)
            kappa    : spectral envelope width in frequency
            Omega    : comb spectral tooth spacing (2*pi)
            n_teeth  : number of comb teeth included in the sum
        """
        W = np.zeros_like(T_mesh, dtype=complex)
        teeth = range(-n_teeth, n_teeth + 1)
        
        # Relative coordinates with temporal shift tau
        t_rel = T_mesh - tau
        
        for n in teeth:
            for m in teeth:
                # Teeth weighted by spectral envelope kappa
                c_n = (1.0 / np.sqrt(2.0)) * np.exp(-((n * Omega)**2) / (2.0 * kappa**2))
                c_m = (1.0 / np.sqrt(2.0)) * np.exp(-((m * Omega)**2) / (2.0 * kappa**2))
                
                # 1. Frequency peak center: (n + m)/2 * Omega + mu
                w_center = 0.5 * (n + m) * Omega + mu
                
                # 2. Beat frequency in time: delta_w = (n - m) * Omega
                delta_w = (n - m) * Omega
                
                # 3. Coherence / dephasing factor from temporal width sigma_t:
                #    exp(-0.5 * sigma_t^2 * delta_w^2)
                dephasing = np.exp(-0.5 * (sigma_t**2) * (delta_w**2))
                
                # 4. Individual peak in frequency with width sigma_w
                freq_peak = np.exp(-((W_mesh - w_center)**2) / (sigma_w**2))
                
                # 5. Quantum interference fringe in time
                fringe = np.exp(-1j * delta_w * t_rel)
                
                term = c_n * np.conj(c_m) * dephasing * freq_peak * fringe
                W += term
                
        # 6. Global Gaussian envelope in time and take real part
        env_t = np.exp(-(t_rel**2) / (T_env**2))
        return np.real(W) * env_t


    line_w, = ax_f.plot([], [], lw=2.2)
    line_env_w, = ax_f.plot([], [], ls=':', color='darkorange', lw=1.6, label=r'Envelope $\kappa$')
    ax_f.axvline(0, color='gray', ls='--', alpha=0.5)
    for n in range(-3, 4):
        ax_f.axvline(n, color='lightgray', ls=':', alpha=0.3)
    ax_f.set_xlim(-w_factor, w_factor)
    ax_f.set_ylim(-0.05, 1.15)
    ax_f.set_xlabel(r'Detuning Frequency $\omega / \Omega$', fontweight='bold', fontsize=11)
    ax_f.set_ylabel(r'Intensity $|\tilde{\psi}(\omega)|^2$', fontweight='bold', fontsize=11)
    ax_f.grid(True, alpha=0.25)
    ax_f.legend(loc='upper right', framealpha=0.9, fontsize=9)

    line_t, = ax_t.plot([], [], lw=2.2)
    line_env_t, = ax_t.plot([], [], ls=':', color='darkorange', lw=1.6, label=r'Envelope $T_{\mathrm{env}}$')
    ax_t.axvline(0, color='gray', ls='--', alpha=0.5)
    for k in range(-3, 4):
        ax_t.axvline(k, color='lightgray', ls=':', alpha=0.3)
    ax_t.set_xlim(-t_factor, t_factor)
    ax_t.set_ylim(-0.05, 1.15)
    ax_t.set_xlabel(r'Arrival Time $t / T$', fontweight='bold', fontsize=11)
    ax_t.set_ylabel(r'Intensity $|\psi(t)|^2$', fontweight='bold', fontsize=11)
    ax_t.grid(True, alpha=0.25)
    ax_t.legend(loc='upper right', framealpha=0.9, fontsize=9)

    #W_init = compute_wigner(0, 0, 1.0, T_env0)
    W_init=compute_exact_wigner(T_mesh, W_mesh,0,0,sigma_0, sigma_t0, T_env0, kappa, Omega ,n_teeth)
    vmax0 = np.max(np.abs(W_init))
    extent = [-wigner_factor, wigner_factor, -wigner_factor, wigner_factor]
    im_wign = ax_w.imshow(W_init, extent=extent, origin='lower', cmap='RdBu_r', 
                          vmin=-vmax0, vmax=vmax0, interpolation='bilinear', aspect='auto')

    ax_w.axhline(0, color='black', ls='--', lw=0.9, alpha=0.4)
    ax_w.axvline(0, color='black', ls='--', lw=0.9, alpha=0.4)
    for k in [-1, 1]:
        ax_w.axvline(k, color='gray', ls=':', lw=0.8, alpha=0.4)
        ax_w.axhline(k, color='gray', ls=':', lw=0.8, alpha=0.4)
    ax_w.set_xlim(-wigner_factor, wigner_factor)
    ax_w.set_ylim(-wigner_factor, wigner_factor)
    ax_w.set_xlabel(r'Arrival Time $t / T$', fontweight='bold', fontsize=11)
    ax_w.set_ylabel(r'Detuning Frequency $\omega / \Omega$', fontweight='bold', fontsize=11)

    cbar = fig.colorbar(im_wign, ax=ax_w, fraction=0.046, pad=0.04)
    cbar.set_label(r'$W(t, \omega)$', fontweight='bold', fontsize=10)

    shift_scatter = ax_w.scatter([0], [0], color='gold', edgecolor='black', s=80, zorder=6)

    title_text = fig.suptitle("", fontsize=13, fontweight='bold', y=0.99)
    sub_text = fig.text(0.5, 0.93, "", ha='center', fontsize=10.5, fontstyle='italic', color='#333333')

    metrics_box = dict(boxstyle='round,pad=0.35', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9)
    badge_f = ax_f.text(0.04, 0.78, "", transform=ax_f.transAxes, fontsize=9.5, bbox=metrics_box)
    badge_t = ax_t.text(0.04, 0.78, "", transform=ax_t.transAxes, fontsize=9.5, bbox=metrics_box)
    badge_w = ax_w.text(0.04, 0.78, "", transform=ax_w.transAxes, fontsize=9.5, bbox=metrics_box)

    def update(frame):
        st = get_frame_state(frame)

        I_w = np.zeros_like(w_pts)
        for n in range(-n_teeth, n_teeth + 1):
            center = n * Omega + st["mu"]
            env = np.exp(-(center**2) / (kappa**2))
            I_w += env * np.exp(-((w_pts - center)**2) / (st["sig_w"]**2))
        I_w /= np.max(I_w)

        line_w.set_data(w_pts / Omega, I_w)
        line_w.set_color(st["color"])
        env_f = np.exp(-(w_pts**2) / (kappa**2))
        line_env_w.set_data(w_pts / Omega, env_f)
        ax_f.set_title(r"Frequency Spectrum $|\tilde{\psi}(\omega)|^2$", fontweight='bold', fontsize=11)
        badge_f.set_text(r"$\mu / \Omega$: " f"{st['mu']/Omega:+.2f}\n" r"$\sigma_\omega$: " f"{st['sig_w']:.3f}")

        I_t = np.zeros_like(t_pts)
        for k in range(-n_teeth, n_teeth + 1):
            center = k * T + st["tau"]
            env = np.exp(-(center**2) / (st["T_env"]**2))
            I_t += env * np.exp(-((t_pts - center)**2) / (st["sig_t"]**2))
        I_t /= np.max(I_t)

        line_t.set_data(t_pts / T, I_t)
        line_t.set_color(st["color"])
        env_t = np.exp(-(t_pts**2) / (st["T_env"]**2))
        line_env_t.set_data(t_pts / T, env_t)
        ax_t.set_title(r"Time Pulse Train $|\psi(t)|^2$", fontweight='bold', fontsize=11)
        badge_t.set_text(r"$\tau / T$: " f"{st['tau']/T:+.2f}\n" r"$\sigma_t$: " f"{st['sig_t']:.3f}")

     #   W = compute_wigner(st["tau"], st["mu"], st["broadening"], st["T_env"])
        W=compute_exact_wigner(T_mesh, W_mesh,st["tau"],st["mu"],st["sig_w"], st["sig_t"], st["T_env"], kappa, Omega ,n_teeth)
        vmax = max(0.4, np.max(np.abs(W)))
        im_wign.set_data(W)
        im_wign.set_clim(-vmax, vmax)

        shift_scatter.set_offsets([[st["tau"] / T, st["mu"] / Omega]])

        ax_w.set_title(r"Wigner Function $W(t, \omega)$", fontweight='bold', fontsize=11)
       # badge_w.set_text(f"Noise: {st['broadening']:.1f}x")

        title_text.set_text(st["stage_name"])
        title_text.set_color(st["color"])
        sub_text.set_text(st["step_desc"])

        return line_w, line_t, im_wign, shift_scatter

    print("Generating 140 frames of smooth QEC process animation from YAML config...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    print(f"Saving optimized animation to {output_gif} via Pillow...")
    writer = animation.PillowWriter(fps=fps, metadata=dict(artist='Andriana Misailidi'))
    anim.save(output_gif, writer=writer)
    plt.close(fig)
    print(f"Animation saved successfully: {output_gif} ({os.path.getsize(output_gif) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    create_qec_animation()
