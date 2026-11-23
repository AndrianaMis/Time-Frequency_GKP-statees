

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plot_qec_stages import load_config, spectral_intensity, get_env
def create_stages_animation(config_file="qec_params.yaml", output_gif="qec_stages_animation.gif", fps=15, dpi=100):
    cfg = load_config(config_file)
    print(f"Loaded config from: {config_file}")

    Omega = 2.0 * np.pi       # Grid comb spacing
    T = np.pi / Omega         # Time pulse spacing

    # Clean state
    clean_cfg = cfg.get("clean_state", {})
    sigma_0 = clean_cfg.get("sigma_0", 0.22)
    kappa = clean_cfg.get("kappa", 14.0)
    sigma_t0 = 1.0 / kappa
    T_env0 = np.sqrt(2.0) / sigma_0

    # Noise parameters
    noise_cfg = cfg.get("optical_channel_noise", {})
    mu_noise = noise_cfg.get("mu", 0.38 * Omega)
    tau_noise = noise_cfg.get("tau", 0.30 * T)
    sigma_w_noise = noise_cfg.get("sigma_w_noise", 0.32)
    sigma_t_noise = noise_cfg.get("sigma_t_noise", 0.11)

    # Ancillas
    anc_cfg = cfg.get("ancilla_parameters", {})
    sigma_a_f = anc_cfg.get("sigma_a_w", 0.16)
    sigma_a_t = anc_cfg.get("sigma_a_t", 0.045)

    sigma_broad_f = np.sqrt(sigma_0**2 + sigma_w_noise**2)
    sigma_broad_t = np.sqrt(sigma_t0**2 + sigma_t_noise**2)
    sigma_f_r1 = np.sqrt((sigma_0**2 * sigma_a_f**2) / (sigma_0**2 + sigma_a_f**2))
    T_env_r1 = np.sqrt(2.0) / sigma_f_r1
    sigma_t_r2 = np.sqrt((sigma_t0**2 * sigma_a_t**2) / (sigma_t0**2 + sigma_a_t**2))

    # Plotting configuration
    plot_cfg = cfg.get("plotting", {})
    n_teeth = plot_cfg.get("n_teeth", 6)
    w_factor = plot_cfg.get("w_range_factor", 3.1)
    t_factor = plot_cfg.get("t_range_factor", 3.1)
    res_1d = 600  # Resolution for smooth and fast animation

    w_pts = np.linspace(-w_factor * Omega, w_factor * Omega, res_1d)
    t_pts = np.linspace(-t_factor * T, t_factor * T, res_1d)

    # 130 total animation frames divided into stages and smooth transitions
    n_frames = 130

    def get_frame_state(frame):
        # Smooth cosine ease-in-out interpolation
        def ease(p):
            return 0.5 * (1.0 - np.cos(np.pi * np.clip(p, 0.0, 1.0)))

        if frame <= 15:
            # Stage 1: Clean GKP State
            stage_name = "Stage 1: Clean GKP State"
            desc = r"Error-free input: teeth of width $\sigma_0$, pulses of duration $\sigma_{t,0}$"
            mu = 0.0
            tau = 0.0
            sig_w = sigma_0
            sig_t = sigma_t0
            T_env = T_env0
            color = "#1f77b4"

        elif frame <= 40:
            # Channel Degradation Transition
            p = ease((frame - 15) / 25.0)
            stage_name = "Optical Channel Degradation (Noise Injection)"
            desc = f"Random walk shift (tau, mu) + tooth broadening ({int(p*100)}%)"
            mu = p * mu_noise
            tau = p * tau_noise
            sig_w = sigma_0 + p * (sigma_broad_f - sigma_0)
            sig_t = sigma_t0 + p * (sigma_broad_t - sigma_t0)
            T_env = T_env0
            color = "#d62728"

        elif frame <= 52:
            # Stage 2: Noisy Corrupted State
            stage_name = "Stage 2: After Channel Degradation"
            desc = r"Noisy state: shifted by $(\tau, \mu)$, broadened by $\sigma_\omega, \sigma_t$"
            mu = mu_noise
            tau = tau_noise
            sig_w = sigma_broad_f
            sig_t = sigma_broad_t
            T_env = T_env0
            color = "#d62728"

        elif frame <= 78:
            # Round 1: Frequency Correction Transition
            p = ease((frame - 52) / 26.0)
            stage_name = "Round 1: Frequency Error Correction"
            desc = f"Homodyne kick (-mu -> 0), teeth narrowing (sigma_eff < sigma_0) ({int(p*100)}%)"
            mu = (1.0 - p) * mu_noise
            tau = tau_noise  # Time jitter remains for Round 2
            sig_w = sigma_broad_f - p * (sigma_broad_f - sigma_f_r1)
            sig_t = sigma_broad_t  # Time pulses stay broadened
            T_env = T_env0 + p * (T_env_r1 - T_env0)  # Time envelope broadens (Heisenberg)
            color = "#9467bd"

        elif frame <= 90:
            # Stage 3: Post-Round 1 State
            stage_name = "Stage 3: After Round 1 (Freq Round)"
            desc = r"$\mu \to 0$ cancelled, frequency re-narrowed ($\sigma_{\mathrm{eff}} < \sigma_0$), time anti-squeezed"
            mu = 0.0
            tau = tau_noise
            sig_w = sigma_f_r1
            sig_t = sigma_broad_t
            T_env = T_env_r1
            color = "#9467bd"

        elif frame <= 115:
            # Round 2: Time Correction Transition
            p = ease((frame - 90) / 25.0)
            stage_name = "Round 2: Time Error Correction"
            desc = f"Time detection + delay kick (-tau -> 0), pulse duration re-narrowing ({int(p*100)}%)"
            mu = 0.0
            tau = (1.0 - p) * tau_noise
            sig_w = sigma_f_r1
            sig_t = sigma_broad_t - p * (sigma_broad_t - sigma_t_r2)
            T_env = T_env_r1
            color = "#2ca02c"

        else:
            # Stage 4: Restored State
            stage_name = "Stage 4: After Round 2 (Time Round)"
            desc = r"$\tau \to 0$ cancelled, pulses re-narrowed ($\sigma_{t,\mathrm{eff}} < \sigma_{t,0}$), restored state"
            mu = 0.0
            tau = 0.0
            sig_w = sigma_f_r1
            sig_t = sigma_t_r2
            T_env = T_env_r1
            color = "#2ca02c"

        return {
            "stage_name": stage_name, "desc": desc,
            "mu": mu, "tau": tau, "sig_w": sig_w, "sig_t": sig_t,
            "T_env": T_env, "color": color
        }

    # Setup 2-Panel Figure matching plot_qec_stages.py
    fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=dpi)
    fig.patch.set_facecolor('#ffffff')

    # Frequency Domain plot elements
    line_w, = ax_f.plot([], [], lw=2.2)
    line_env_w, = ax_f.plot([], [], ls=':', color='darkorange', lw=1.6, label=r'Envelope $\kappa$')
    ax_f.axvline(0, color='gray', ls='--', alpha=0.4)
    for n in range(-3, 4):
        ax_f.axvline(n, color='lightgray', ls=':', alpha=0.3)
    ax_f.set_xlim(-w_factor, w_factor)
    ax_f.set_ylim(-0.05, 1.15)
    ax_f.set_xlabel(r'Detuning Frequency $\omega / \Omega$', fontweight='bold', fontsize=11)
    ax_f.set_ylabel(r'Intensity $|\tilde{\psi}(\omega)|^2$', fontweight='bold', fontsize=11)
    ax_f.grid(True, alpha=0.25)
    ax_f.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    # Time Domain plot elements
    line_t, = ax_t.plot([], [], lw=2.2)
    line_env_t, = ax_t.plot([], [], ls=':', color='darkorange', lw=1.6, label=r'Envelope $T_{\mathrm{env}}$')
    ax_t.axvline(0, color='gray', ls='--', alpha=0.4)
    for k in range(-3, 4):
        ax_t.axvline(k, color='lightgray', ls=':', alpha=0.3)
    ax_t.set_xlim(-t_factor, t_factor)
    ax_t.set_ylim(-0.05, 1.15)
    ax_t.set_xlabel(r'Arrival Time $t / T$', fontweight='bold', fontsize=11)
    ax_t.set_ylabel(r'Intensity $|\psi(t)|^2$', fontweight='bold', fontsize=11)
    ax_t.grid(True, alpha=0.25)
    ax_t.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    # Top headers
    title_text = fig.suptitle("", fontsize=13, fontweight='bold', y=0.99)
    sub_text = fig.text(0.5, 0.93, "", ha='center', fontsize=10.5, fontstyle='italic', color='#333333')

    # Metrics badges on subplots
    badge_style = dict(boxstyle='round,pad=0.35', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9)
    badge_f = ax_f.text(0.04, 0.78, "", transform=ax_f.transAxes, fontsize=9.5, bbox=badge_style)
    badge_t = ax_t.text(0.04, 0.78, "", transform=ax_t.transAxes, fontsize=9.5, bbox=badge_style)

    plt.subplots_adjust(top=0.86, wspace=0.22, bottom=0.14)

    # Animation update function
    def update(frame):
        st = get_frame_state(frame)

        # 1. Frequency calculation using your spectral_intensity & get_env
        I_w = spectral_intensity(points=w_pts, period=Omega, disp=st["mu"],
                                 env_width=kappa, signal_width=st["sig_w"], n_teeth=n_teeth)
        env_f = get_env(points=w_pts, env_width=kappa)

        line_w.set_data(w_pts / Omega, I_w)
        line_w.set_color(st["color"])
        line_env_w.set_data(w_pts / Omega, env_f)
        ax_f.set_title(r"Frequency Domain $|\tilde{\psi}(\omega)|^2$", fontweight='bold', fontsize=11)
        badge_f.set_text(rf"$\mu / \Omega$: {st['mu']/Omega:+.2f}" + "\n" + rf"$\sigma_\omega$: {st['sig_w']:.3f}")

        # 2. Time calculation using your spectral_intensity & get_env
        I_t = spectral_intensity(points=t_pts, period=T, disp=st["tau"],
                                 env_width=st["T_env"], signal_width=st["sig_t"], n_teeth=n_teeth)
        env_t = get_env(points=t_pts, env_width=st["T_env"])

        line_t.set_data(t_pts / T, I_t)
        line_t.set_color(st["color"])
        line_env_t.set_data(t_pts / T, env_t)
        ax_t.set_title(r"Time Domain $|\psi(t)|^2$", fontweight='bold', fontsize=11)
        badge_t.set_text(rf"$\tau / T$: {st['tau']/T:+.2f}" + "\n" + rf"$\sigma_t$: {st['sig_t']:.3f}")

        # 3. Update top title and description
        title_text.set_text(st["stage_name"])
        title_text.set_color(st["color"])
        sub_text.set_text(st["desc"])

        return line_w, line_t, line_env_w, line_env_t

    print("Generating 130 frames of 2-panel QEC stages animation...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    print(f"Saving animation to {output_gif} via Pillow...")
    writer = animation.PillowWriter(fps=fps, metadata=dict(artist='Andriana Misailidi'))
    anim.save(output_gif, writer=writer)
    plt.close(fig)
    print(f"Animation saved successfully: {output_gif} ({os.path.getsize(output_gif) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    create_stages_animation()
