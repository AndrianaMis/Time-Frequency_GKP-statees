import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Parameters of the coherent-like state |α⟩
# ==========================================
omega_1 = 2     # Center carrier frequency (rad/s or relative units)
tau = 0      # Arrival time delay (s or relative units)
Delta_omega = 1 # Spectral width (standard deviation of the amplitude)

# Derived temporal width (Fourier minimum-uncertainty relation)
# ψ(t) envelope ~ exp(-(t - tau)^2 / (2 * Delta_t^2)) with Delta_t = 1 / (Delta_omega)
Delta_t = 1.0 / Delta_omega

# ==========================================
# Grids
# ==========================================
omega = np.linspace(-4, 14, 1000)
t = np.linspace(-6, 12, 1000)

# ==========================================
# Wavefunctions
# ==========================================
# Frequency domain: Psi(omega) = exp(-(omega - omega_1)^2 / (2 * Delta_omega^2)) * exp(i * omega * tau)
psi_omega = np.exp(-(omega - omega_1)**2 / ( Delta_omega**2)) * np.exp(1j * omega * tau)
prob_omega = np.abs(psi_omega)**2
real_om=np.real(psi_omega)
# Time domain: psi(t) ~ exp(-(t - tau)^2 / (2 * Delta_t^2)) * exp(-i * omega_1 * (t - tau))
psi_t = np.exp(-(t - tau)**2 / (2 * Delta_t**2)) * np.exp(-1j * omega_1 * (t - tau))
prob_t = np.abs(psi_t)**2
real_field_t = np.real(psi_t)  # Shows carrier oscillations under the envelope

# ==========================================
# 2D Chronocyclic Wigner Distribution
# W(omega, t) = exp(-(omega - omega_1)^2 / Delta_omega^2 - (t - tau)^2 / Delta_t^2)
# ==========================================
T_grid, OMEGA_grid = np.meshgrid(np.linspace(-6, 12, 300), np.linspace(-4, 14, 300))
W = np.exp(-(OMEGA_grid - omega_1)**2 / Delta_omega**2 - (T_grid - tau)**2 / Delta_t**2)

# ==========================================
# Plotting
# ==========================================
fig = plt.figure(figsize=(13, 8))

# 1. Frequency Domain
ax1 = plt.subplot(2, 2, 1)
ax1.plot(omega, prob_omega, 'b-', lw=2, label=r'$|\tilde{\psi}(\omega)|^2$ (Spectral Intensity)')
ax1.plot(omega, real_om, 'b--', alpha=0.5, label=r'Re$[\tilde{\psi}(\omega)]$ (Phase ramp $e^{i\omega\tau}$)')
ax1.axvline(omega_1, color='k', linestyle=':', label=rf'Center $\omega_1 = {omega_1}$')
ax1.set_title("Frequency Domain Spectrum", fontsize=12, fontweight='bold')
ax1.set_xlabel(r"Frequency $\omega$ [rad/s]")
ax1.set_ylabel("Amplitude / Probability")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)

# 2. Time Domain
ax2 = plt.subplot(2, 2, 2)
ax2.plot(t, prob_t, 'r-', lw=2, label=r'$|\psi(t)|^2$ (Temporal Probability)')
ax2.plot(t, real_field_t, 'r--', alpha=0.5, label=r'Re$[\psi(t)]$ (Carrier $\cos(\omega_1 t)$)')
ax2.axvline(tau, color='k', linestyle=':', label=rf'Delay $\tau = {tau}$')
ax2.set_title("Time Domain Wavepacket", fontsize=12, fontweight='bold')
ax2.set_xlabel("Time $t$ [s]")
ax2.set_ylabel("Amplitude / Probability")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

# 3. Chronocyclic Wigner Distribution (Phase Space)
ax3 = plt.subplot(2, 1, 2)
contour = ax3.contourf(T_grid, OMEGA_grid, W, levels=50, cmap='viridis')
cbar = plt.colorbar(contour, ax=ax3)
cbar.set_label(r'$W(\omega, t)$ Amplitude', rotation=270, labelpad=15)
ax3.axvline(tau, color='w', linestyle='--', alpha=0.7, label=rf'$\tau = {tau}$')
ax3.axhline(omega_1, color='w', linestyle=':', alpha=0.7, label=rf'$\omega_1 = {omega_1}$')
ax3.set_title(r"Chronocyclic Wigner Distribution $W(\omega, t)$ (Joint Phase Space)", fontsize=12, fontweight='bold')
ax3.set_xlabel("Time $t$ [s]")
ax3.set_ylabel(r"Frequency $\omega$ [rad/s]")
ax3.legend(loc='upper right', facecolor='black', framealpha=0.5, labelcolor='white')

plt.tight_layout()
plt.show()
