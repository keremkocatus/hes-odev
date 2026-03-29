"""
MAT374 - Ödev 1, Kısım III: Deformasyon Animasyonu

Hareket:  x1 = X1 + 0.20*sin(t)*X1*X2
          x2 = X2 + 0.10*(1-cos(t))*X1^2
          x3 = X3

Başlangıç bölgesi : birim kare [0,1] x [0,1]
Zaman aralığı     : t ∈ [0, 2*pi]
Dil               : Python (matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Hareket dönüşümü ──────────────────────────────────────────────────────────
def phi(X1, X2, t):
    x1 = X1 + 0.20 * np.sin(t) * X1 * X2
    x2 = X2 + 0.10 * (1 - np.cos(t)) * X1**2
    return x1, x2

# ── Birim kare üzerinde referans ızgara ───────────────────────────────────────
N  = 40                              # kenar başına nokta sayısı (düzgün eğriler)
s  = np.linspace(0, 1, N)
n_lines = 11                         # her yönde ızgara çizgisi sayısı
grid_s  = np.linspace(0, 1, n_lines)

# Birim kare sınırı (kapalı döngü): alt → sağ → üst → sol
bnd_X1 = np.concatenate([s, np.ones(N),  s[::-1], np.zeros(N)])
bnd_X2 = np.concatenate([np.zeros(N), s, np.ones(N),  s[::-1]])

# ── Zaman dizisi ──────────────────────────────────────────────────────────────
T      = 2 * np.pi
frames = 150
t_arr  = np.linspace(0, T, frames)

# ── Şekil ayarları ────────────────────────────────────────────────────────────
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("MAT374 – Deformation of the Unit Square", fontsize=13, fontweight="bold")

# Sol: deformasyon grafiği
ax.set_xlim(-0.05, 1.8)
ax.set_ylim(-0.05, 1.5)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("Deformed region")
ax.grid(True, alpha=0.25, lw=0.5)

# Referans sınırı (sabit, kesikli mavi)
ax.plot(bnd_X1, bnd_X2, "b--", lw=1.2, alpha=0.45, label="Reference ($t=0$)")
ax.fill(bnd_X1, bnd_X2, color="steelblue", alpha=0.08)

# Deforme sınır + dolgu (her karede güncellenir)
def_fill,     = ax.fill([], [], color="tomato", alpha=0.30)
def_border,   = ax.plot([], [], "r-", lw=2,   label="Deformed")

# İç ızgara çizgileri (yatay çizgiler X2 = sabit, dikey X1 = sabit)
h_lines = [ax.plot([], [], color="tomato", lw=0.7, alpha=0.55)[0] for _ in grid_s]
v_lines = [ax.plot([], [], color="tomato", lw=0.7, alpha=0.55)[0] for _ in grid_s]

time_txt = ax.text(0.03, 0.96, "", transform=ax.transAxes, fontsize=10,
                   va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.75))
ax.legend(loc="upper right", fontsize=9)

# Sağ: Merkez noktada Jacobian J = det(F), X=(0.5, 0.5)
# Analitik: F11=1+0.2*sin(t)*X2, F12=0.2*sin(t)*X1,
#           F21=0.2*(1-cos(t))*X1, F22=1  (X=(0.5,0.5) noktasında)
def jacobian(t):
    X1, X2 = 0.5, 0.5
    F11 = 1 + 0.20 * np.sin(t) * X2
    F12 = 0.20 * np.sin(t) * X1
    F21 = 0.20 * (1 - np.cos(t)) * X1
    F22 = 1.0
    return F11 * F22 - F12 * F21

J_all = jacobian(t_arr)

ax2.set_xlim(0, T)
ax2.set_ylim(min(J_all) - 0.05, max(J_all) + 0.05)
ax2.set_xlabel("$t$ (rad)")
ax2.set_ylabel("$J = \\det(F)$")
ax2.set_title("Jacobian at $X = (0.5,\\,0.5)$")
ax2.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6, label="$J=1$ (no volume change)")
ax2.plot(t_arr, J_all, color="lightcoral", lw=1, alpha=0.4)  # tam eğri (soluk)
ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax2.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
ax2.grid(True, alpha=0.25, lw=0.5)
ax2.legend(fontsize=9)

J_line, = ax2.plot([], [], "tomato", lw=2.2)
J_dot,  = ax2.plot([], [], "ro", ms=6)

# ── Animasyon güncelleme ──────────────────────────────────────────────────────
def update(frame):
    t = t_arr[frame]

    # Deforme sınır
    px, py = phi(bnd_X1, bnd_X2, t)
    def_border.set_data(np.append(px, px[0]), np.append(py, py[0]))
    def_fill.set_xy(np.column_stack([np.append(px, px[0]),
                                     np.append(py, py[0])]))

    # İç ızgara: yatay çizgiler (X2 = sabit, X1 değişken)
    for i, x2 in enumerate(grid_s):
        gx, gy = phi(s, np.full_like(s, x2), t)
        h_lines[i].set_data(gx, gy)

    # İç ızgara: dikey çizgiler (X1 = sabit, X2 değişken)
    for j, x1 in enumerate(grid_s):
        gx, gy = phi(np.full_like(s, x1), s, t)
        v_lines[j].set_data(gx, gy)

    # Zaman etiketi
    time_txt.set_text(f"t = {t:.2f} rad  ({np.degrees(t):.1f}°)")

    # Jacobian grafiği
    J_line.set_data(t_arr[:frame+1], J_all[:frame+1])
    J_dot.set_data([t_arr[frame]], [J_all[frame]])

    return [def_fill, def_border, time_txt, J_line, J_dot] + h_lines + v_lines

# ── Çalıştır ──────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(fig, update, frames=frames,
                               interval=30, blit=True)

plt.tight_layout()

# Kaydetmek için:
# ani.save("mat374_animation.gif", writer="pillow", fps=30)
# ani.save("mat374_animation.mp4", writer="ffmpeg", fps=30, dpi=150)

plt.show()
