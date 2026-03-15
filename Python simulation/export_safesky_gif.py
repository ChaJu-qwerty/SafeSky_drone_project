"""
export_safesky_gif.py
─────────────────────
Genera safesky_simulation.gif directamente, sin GUI de Tkinter.
Igual que los scripts orca_sim.py / astar_viz.py — corre y listo.

Uso:
    python export_safesky_gif.py
    python export_safesky_gif.py --drones 4 --obs 4 --out mi_sim.gif

SafeSky MR3001B · Tecnológico de Monterrey 2026
"""

import sys, os, time, warnings, io, argparse
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')   # sin ventanas — igual que orca_sim.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ── Ruta al proyecto ──────────────────────────────────────────────────────────
PROJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)

from physics import QuadParams, DroneState, QuadDynamics, PIDController, WindAR1, \
                   ORCA_RADIUS, R_DRONE_BODY, D_COLLISION, SAFETY_BUFFER
from planner import Obstacle, HybridAStar, ORCAPlanner

# ── Importar Simulation y helpers desde safesky_main_1 ───────────────────────
# Parchamos tkinter antes de importar para evitar el error de display
import unittest.mock as _mock
for _m in ['tkinter','tkinter.ttk','tkinter.messagebox','tkinter.filedialog',
           'matplotlib.backends.backend_tkagg']:
    if _m not in sys.modules:
        sys.modules[_m] = _mock.MagicMock()

# Ahora sí podemos importar sin que explote por falta de display
import importlib, types

# Parchear matplotlib.use para que TkAgg no explote sin display
_orig_mpl_use = matplotlib.use
matplotlib.use = lambda *a, **kw: None   # ignorar todos los use()

# Parchear tkinter completo con atributos reales mínimos para evitar
# AttributeError en cbook._get_running_interactive_framework
import tkinter as _tk_mock_mod
_tk_mock_mod.mainloop = _mock.MagicMock()
_tk_mock_mod.mainloop.__code__ = (lambda: None).__code__
_tk_mock_mod.Misc = _mock.MagicMock()
_tk_mock_mod.Misc.mainloop = _mock.MagicMock()
_tk_mock_mod.Misc.mainloop.__code__ = (lambda: None).__code__
sys.modules['tkinter'] = _tk_mock_mod

_spec = importlib.util.spec_from_file_location(
    "safesky_main", os.path.join(PROJ_DIR, "safesky_main_1.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Restaurar matplotlib.use y forzar Agg
matplotlib.use = _orig_mpl_use
matplotlib.use('Agg')

Simulation   = _mod.Simulation
draw_drone_3d = _mod.draw_drone_3d
DroneState   = _mod.DroneState
DT           = _mod.DT
SPACE        = _mod.SPACE
TRAIL_LEN    = _mod.TRAIL_LEN
CB = _mod.CB; CW = _mod.CW; CR = _mod.CR
CL = _mod.CL; CG = _mod.CG

# ── Argumentos ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Exportar simulación SafeSky como GIF')
parser.add_argument('--drones',  type=int,   default=3,            help='Número de drones (3-6)')
parser.add_argument('--obs',     type=int,   default=3,            help='Número de obstáculos')
parser.add_argument('--alpha',   type=float, default=0.85,         help='Correlación viento AR(1)')
parser.add_argument('--sigma',   type=float, default=0.20,         help='Intensidad viento')
parser.add_argument('--mode',    type=str,   default='ab',         help='Modo misión: ab / ab_cross')
parser.add_argument('--obs_mode',type=str,   default='random',     help='Modo obstáculos: random / fixed')
parser.add_argument('--frames',  type=int,   default=120,          help='Máximo de frames en el GIF')
parser.add_argument('--fps',     type=int,   default=12,           help='FPS del GIF')
parser.add_argument('--dpi',     type=int,   default=85,           help='DPI de cada frame')
parser.add_argument('--out',     type=str,   default='safesky_simulation.gif', help='Archivo de salida')
args = parser.parse_args()

# ── Paso 1: Simular ───────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  SafeSky — Exportación GIF standalone")
print(f"  Drones: {args.drones}  Obstáculos: {args.obs}  Modo: {args.mode}")
print(f"{'─'*55}")

print("⏳ Fase 1/2: Simulando física...", end='', flush=True)
t0_sim = time.perf_counter()

sim = Simulation(args.drones, args.obs, args.alpha, args.sigma,
                 args.obs_mode, args.mode)

MAX_STEPS = int(60.0 / DT)   # máx 60s simulados
for step in range(MAX_STEPS):
    sim.step()
    if step % 100 == 0:
        n_arr = sum(d.arrived for d in sim.drones)
        print(f"\r⏳ Fase 1/2: t={sim.t:.1f}s  ({n_arr}/{args.drones} llegaron)  ", end='', flush=True)
    if sim.all_arrived():
        break

t_sim = time.perf_counter() - t0_sim
n_arrived = sum(d.arrived for d in sim.drones)
print(f"\r✅ Fase 1/2: Simulación completada en {t_sim:.1f}s  "
      f"(t_sim={sim.t:.1f}s · {n_arrived}/{args.drones} llegaron)")

# ── Extraer datos de trayectorias ─────────────────────────────────────────────
all_pos    = [list(d.traj)        for d in sim.drones]
all_angles = [list(d.hist_angles) for d in sim.drones]
all_wind   = [list(d.hist_wind)   for d in sim.drones]

n_steps    = len(all_pos[0])
t_hist     = [i * DT for i in range(n_steps)]
t_sim_total = t_hist[-1]

# ── Paso 2: Renderizar frames ─────────────────────────────────────────────────
SKIP       = max(1, n_steps // args.frames)
frames_idx = list(range(0, n_steps, SKIP))
total_f    = len(frames_idx)

print(f"\n🎨 Fase 2/2: Renderizando {total_f} frames (dpi={args.dpi})...")
t0_render = time.perf_counter()

# Resolución reducida de elipsoides para velocidad
u_s = np.linspace(0, 2*np.pi, 8)
v_s = np.linspace(0, np.pi,   6)

# Crear figura
fig = plt.figure(figsize=(12, 7), facecolor=CB)
fig.suptitle(
    f"SafeSky — Enjambre 6-DOF (Euler-Lagrange + DJI F450)"
    f"  |  t={t_sim_total:.1f}s  |  {args.drones} drones  {args.obs} obs",
    color=CW, fontsize=10, fontweight='bold')
gs = gridspec.GridSpec(2, 3, fig, left=0.04, right=0.98,
                       top=0.94, bottom=0.06, hspace=0.38, wspace=0.30)
ax3  = fig.add_subplot(gs[:, 0:2], projection='3d')
ax_a = fig.add_subplot(gs[0, 2])
ax_k = fig.add_subplot(gs[1, 2])

def render_frame(fi):
    """Renderiza un frame en la figura existente."""
    ax3.cla()
    ax3.set_facecolor('#0D1421')
    ax3.set_xlim(0, SPACE); ax3.set_ylim(0, SPACE); ax3.set_zlim(0, 10)
    ax3.view_init(elev=22, azim=-55 + fi * 0.15)
    ax3.tick_params(colors='gray', labelsize=5)
    for p in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        p.fill = False

    # Obstáculos
    for o in sim.obstacles:
        c = o.center
        xs = c[0] + o.radii[0]*np.outer(np.cos(u_s), np.sin(v_s))
        ys = c[1] + o.radii[1]*np.outer(np.sin(u_s), np.sin(v_s))
        zs = c[2] + o.radii[2]*np.outer(np.ones_like(u_s), np.cos(v_s))
        ax3.plot_surface(xs, ys, zs, alpha=0.28, color=CR, linewidth=0)
        ax3.plot_wireframe(xs, ys, zs, color='#FF8888',
                           linewidth=0.4, alpha=0.35, rstride=3, cstride=3)

    # Drones + trails + viento
    for i, d in enumerate(sim.drones):
        col = d.color
        tr  = np.array(all_pos[i][max(0, fi-TRAIL_LEN):fi+1])
        if len(tr) > 1:
            ax3.plot(tr[:,0], tr[:,1], tr[:,2], color=col, lw=1.5, alpha=0.65)
        pos_fi = np.array(all_pos[i][min(fi, len(all_pos[i])-1)])
        if all_angles[i]:
            aix = min(fi, len(all_angles[i])-1)
            st2 = DroneState(pos=pos_fi, angles=np.array(all_angles[i][aix]))
            draw_drone_3d(ax3, st2, col, scale=0.22)
        if all_wind[i] and fi < len(all_wind[i]):
            w  = np.array(all_wind[i][fi])
            wm = np.linalg.norm(w)
            if wm > 0.05:
                wu, wv, ww2 = w / max(wm, 0.01) * 0.55
                ax3.quiver(pos_fi[0], pos_fi[1], pos_fi[2],
                           wu, wv, ww2, length=max(wm*0.55, 0.25),
                           color='#A8FF78', alpha=0.85,
                           linewidth=1.2, arrow_length_ratio=0.35)

    ax3.set_title(f"t = {t_hist[min(fi, len(t_hist)-1)]:.1f} s", color=CW, fontsize=8)

    # Panel ángulos D1
    ax_a.cla(); ax_a.set_facecolor('#0F1525')
    ax_a.tick_params(colors='gray', labelsize=6)
    ax_a.set_title('Ángulos D1', color=CW, fontsize=7, pad=2)
    if all_angles[0] and fi > 0:
        arr = np.array(all_angles[0][:min(fi+1, len(all_angles[0]))])
        ts  = t_hist[:len(arr)]
        for j, (c2, lb) in enumerate(zip([CR, CL, CG], ['φ','θ','ψ'])):
            ax_a.plot(ts, arr[:,j], color=c2, lw=1.2, label=lb)
        ax_a.legend(fontsize=6, loc='upper right',
                    framealpha=0.3, facecolor='#1A1A2E', labelcolor=CW)
    ax_a.axhline(0, color='white', ls=':', lw=0.5, alpha=0.3)

    # Panel estado
    ax_k.cla(); ax_k.set_facecolor('#0F1525'); ax_k.axis('off')
    ax_k.set_title('Estado', color=CW, fontsize=7, pad=2)
    lines_k = []
    for ii, dd in enumerate(sim.drones):
        p_fi = np.array(all_pos[ii][min(fi, len(all_pos[ii])-1)])
        ref  = dd.goal if dd.goal is not None else np.array(dd.wps[0])
        d2g  = np.linalg.norm(p_fi - ref)
        lines_k.append(f"D{dd.id+1}: {'✓ OK' if dd.arrived else f'{d2g:.1f}m'}")
    lines_k.append(f"t={t_hist[min(fi,len(t_hist)-1)]:.1f}s")
    ax_k.text(0.05, 0.95, '\n'.join(lines_k), transform=ax_k.transAxes,
              color=CG, fontsize=8, va='top', fontfamily='monospace')

# ── Acumular frames PIL ────────────────────────────────────────────────────────
frames_pil = []
for idx, fi in enumerate(frames_idx):
    render_frame(fi)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=args.dpi, facecolor=CB, bbox_inches='tight')
    buf.seek(0)
    frames_pil.append(Image.open(buf).copy())

    # Progreso en consola
    pct     = int((idx+1) / total_f * 100)
    elapsed = time.perf_counter() - t0_render
    eta     = (elapsed / max(idx+1, 1)) * (total_f - idx - 1)
    bar     = '█' * (pct // 5) + '░' * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  frame {idx+1}/{total_f}  "
          f"ETA: {eta:.0f}s   ", end='', flush=True)

plt.close(fig)
t_render = time.perf_counter() - t0_render
print(f"\r✅ Fase 2/2: {total_f} frames renderizados en {t_render:.1f}s{' '*20}")

# ── Guardar GIF ────────────────────────────────────────────────────────────────
print(f"\n💾 Guardando GIF → {args.out} ...", end='', flush=True)
dur_ms = max(40, int(1000 / args.fps))
frames_pil[0].save(
    args.out,
    save_all=True,
    append_images=frames_pil[1:],
    optimize=False,
    loop=0,
    duration=dur_ms
)
size_mb = os.path.getsize(args.out) / 1e6
t_total = time.perf_counter() - t0_sim
print(f"\r✅ GIF guardado: {args.out}  ({size_mb:.1f} MB · {total_f} frames · {args.fps}fps)")
print(f"\n{'─'*55}")
print(f"  Total: {t_total:.1f}s  "
      f"(sim={t_sim:.1f}s · render={t_render:.1f}s)")
print(f"  Drones llegaron: {n_arrived}/{args.drones}")
print(f"{'─'*55}\n")
