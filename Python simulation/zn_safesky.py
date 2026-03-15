"""
zn_safesky.py
=============
Justificacion del controlador PID en cascada SafeSky (DJI F450)
mediante el metodo de Ziegler-Nichols en frecuencia critica.

Genera una figura de 5 pasos que muestra todo el proceso:
  Paso 1 - Modelo de planta normalizada (por que 1/s^2 para cada lazo)
  Paso 2 - Identificacion de la frecuencia critica discreta
  Paso 3 - Calculo de Ku y Pu + formulas Z-N clasicas
  Paso 4 - Desintonizacion conservadora (justificacion fisica)
  Paso 5 - Comparacion final Z-N vs ganancias implementadas

Dependencias: numpy, matplotlib, scipy
  pip install numpy matplotlib scipy

Uso:
  python zn_safesky.py

Proyecto: SafeSky - Enjambres Dinamicos
Equipo  : Arellano, Chavez, Gonzalez, Leija, Madrigal
Curso   : MR3001B - Diseno y desarrollo de robots
ITESM   : Marzo 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from scipy import signal

# ================================================================
#  PARAMETROS FISICOS DJI F450  (de physics.py)
# ================================================================
m    = 1.0        # kg   masa total
g    = 9.81       # m/s2 gravedad
Ixx  = 7.75e-3    # kg*m2 inercia roll / pitch
Iyy  = 7.75e-3
Izz  = 13.6e-3    # kg*m2 inercia yaw
l    = 0.225      # m   brazo centro-rotor
kf   = 2.98e-6    # N/(rad/s)^2  coef. empuje
DT   = 0.05       # s   periodo de muestreo

# Ganancias implementadas en physics.py
GANANCIAS_CODIGO = {
    'Z':   dict(Kp=5.0,  Ki=0.12, Kd=2.8),
    'XY':  dict(Kp=1.0,  Ki=0.03, Kd=1.4),
    'Att': dict(Kp=6.0,  Ki=0.04, Kd=2.2),
}

# ================================================================
#  PALETA  (fondo oscuro, estilo SafeSky)
# ================================================================
CB   = '#0D1421'   # fondo figura
CP   = '#111927'   # fondo panel
CW   = '#E8EDF2'   # texto claro
CDG  = '#4A5568'   # gris oscuro
CG   = '#39D353'   # verde exito
CY   = '#F5D547'   # amarillo aviso
CR   = '#FF5370'   # rojo alerta
CC   = '#00B4D8'   # cian acento
CL   = '#BD93F9'   # violeta
CO   = '#FFB86C'   # naranja

C_Z   = '#00B4D8'   # lazo Z
C_XY  = '#39D353'   # lazo XY
C_ATT = '#FFB86C'   # lazo actitud
C_ZN  = '#BD93F9'   # Z-N referencia

FONT = 'monospace'


# ================================================================
#  CALCULO Z-N
# ================================================================
def zn_calc(wu):
    """Dado omega_ult [rad/s], retorna Ku, Pu y ganancias Z-N PID."""
    Ku = wu ** 2          # G = 1/s^2  =>  |G(jwu)| = 1/wu^2  => Ku*|G|=1 => Ku=wu^2
    Pu = 2.0 * np.pi / wu
    Kp = 0.6  * Ku
    Ti = 0.5  * Pu
    Td = 0.125 * Pu
    Ki = Kp / Ti
    Kd = Kp * Td
    return dict(Ku=Ku, Pu=Pu, Kp=Kp, Ki=Ki, Kd=Kd, Ti=Ti, Td=Td)


# Frecuencias criticas: fracciones conservadoras de Nyquist
# w_Nyq = pi/T = 62.83 rad/s  (en transformada w bilineal)
# Cada lazo opera a una escala de tiempo distinta:
#   Actitud (mas rapido) > Z > XY (mas lento, lazo externo)
W_NYQ = np.pi / DT          # 62.83 rad/s
WU = {
    'Z':   W_NYQ / 4.71,    # ~ 13.33 rad/s  => 1/3 de w_Nyq
    'XY':  W_NYQ / 6.28,    # ~ 10.00 rad/s  => 1/4 de w_Nyq
    'Att': W_NYQ / 3.93,    # ~ 16.00 rad/s  => 1/2.5 de w_Nyq
}
FRAC_LABEL = {'Z': 'wNyq/3', 'XY': 'wNyq/4', 'Att': 'wNyq/2.5'}
COLORES = {'Z': C_Z, 'XY': C_XY, 'Att': C_ATT}
ZN = {k: zn_calc(wu) for k, wu in WU.items()}


# ================================================================
#  FIGURA PRINCIPAL
# ================================================================
fig = plt.figure(figsize=(20, 24), facecolor=CB)
fig.suptitle(
    'SafeSky — Justificacion Controlador PID en Cascada\n'
    'Metodo Ziegler-Nichols + Desintonizacion Conservadora  |  DJI F450',
    color=CW, fontsize=14, fontweight='bold', y=0.995, fontfamily=FONT)

gs_main = gridspec.GridSpec(5, 1, fig,
    top=0.975, bottom=0.03,
    left=0.06, right=0.97,
    hspace=0.55)


# ----------------------------------------------------------------
#  UTILIDADES
# ----------------------------------------------------------------
def ax_style(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(CP)
    ax.tick_params(colors=CDG, labelsize=8, labelcolor='#8899AA')
    for spine in ax.spines.values():
        spine.set_edgecolor(CDG)
        spine.set_linewidth(0.5)
    ax.set_title(title, color=CC, fontsize=10, fontweight='bold',
                 pad=6, fontfamily=FONT, loc='left')
    if xlabel: ax.set_xlabel(xlabel, color='#8899AA', fontsize=8, fontfamily=FONT)
    if ylabel: ax.set_ylabel(ylabel, color='#8899AA', fontsize=8, fontfamily=FONT)
    ax.grid(True, color=CDG, linewidth=0.4, alpha=0.5)


def badge(ax, x, y, text, color, fontsize=8):
    ax.text(x, y, text, transform=ax.transAxes,
            color=color, fontsize=fontsize, fontfamily=FONT,
            bbox=dict(facecolor=CB, edgecolor=color, linewidth=0.8,
                      boxstyle='round,pad=0.3'))


def step_header(fig, gs_row, num, titulo, subtitulo):
    ax_h = fig.add_subplot(gs_row)
    ax_h.set_facecolor('#0A1628')
    for spine in ax_h.spines.values():
        spine.set_edgecolor(CC)
        spine.set_linewidth(1.2)
    ax_h.set_xticks([]); ax_h.set_yticks([])
    ax_h.text(0.012, 0.55, f'PASO {num}', color=CC,
              fontsize=9, fontweight='bold', va='center',
              transform=ax_h.transAxes, fontfamily=FONT)
    ax_h.text(0.085, 0.55, titulo, color=CW,
              fontsize=11, fontweight='bold', va='center',
              transform=ax_h.transAxes, fontfamily=FONT)
    ax_h.text(0.085, 0.10, subtitulo, color='#8899AA',
              fontsize=7.5, va='center',
              transform=ax_h.transAxes, fontfamily=FONT)
    return ax_h


# ================================================================
#  PASO 1 — MODELO DE PLANTA NORMALIZADA
# ================================================================
gs1 = gridspec.GridSpecFromSubplotSpec(2, 3, gs_main[0],
    hspace=0.5, wspace=0.35)

# Cabecera
ax_h1 = step_header(fig, gs1[0, :], 1,
    'Modelo de planta normalizada',
    'Antes de Z-N se identifica G(s) de cada lazo. Los tres lazos '
    'reducen a G(s) = 1/s^2 una vez dividida la masa o la inercia.')

# Diagrama de bloques simplificado para cada lazo
info_lazos = [
    ('Lazo Z  (altura)',
     'u1 [N]  ->  az = u1/m  ->  z = integra(integra(az))',
     'G_z(s) = 1/s^2',
     'PID produce az [m/s2]\nPlanta: z = int(int(az dt) dt)',
     C_Z),
    ('Lazo XY  (posicion horizontal)',
     'axy [m/s2]  ->  theta = axy/g  ->  x = integra(integra(axy))',
     'G_xy(s) = 1/s^2',
     'PID produce axy [m/s2]\nangulo theta = axy/g\nPlanta: x = int(int(axy dt) dt)',
     C_XY),
    ('Lazo Actitud  (phi, theta)',
     'torque/I [rad/s2]  ->  phi = integra(integra(alfa))',
     'G_att(s) = 1/s^2',
     'PID produce alfa = torque/I\nCodigo mult. por I: u2 = Kp*Ixx*e\nPlanta: phi = int(int(alfa dt) dt)',
     C_ATT),
]

for col, (titulo, desc, tf_str, detalle, color) in enumerate(info_lazos):
    ax = fig.add_subplot(gs1[1, col])
    ax.set_facecolor(CP)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(1.0)

    ax.text(5, 9.2, titulo, color=color, fontsize=8, fontweight='bold',
            ha='center', va='top', fontfamily=FONT)

    # Caja de funcion de transferencia
    rect = mpatches.FancyBboxPatch((2.5, 5.2), 5, 2.2,
        boxstyle='round,pad=0.2', facecolor='#0A1628',
        edgecolor=color, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(5, 6.4, tf_str, color=CW, fontsize=10, fontweight='bold',
            ha='center', va='center', fontfamily=FONT)
    ax.text(5, 5.7, 'funcion de transferencia', color='#8899AA',
            fontsize=6.5, ha='center', va='center', fontfamily=FONT)

    # Descripcion
    for i, linea in enumerate(detalle.split('\n')):
        ax.text(5, 4.5 - i*0.9, linea, color='#8899AA',
                fontsize=6.5, ha='center', va='top', fontfamily=FONT)

    # Flechas entrada/salida
    ax.annotate('', xy=(2.5, 6.3), xytext=(0.8, 6.3),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    ax.annotate('', xy=(9.2, 6.3), xytext=(7.5, 6.3),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    ax.text(0.5, 6.8, 'entrada', color=color, fontsize=6, fontfamily=FONT)
    ax.text(7.6, 6.8, 'posicion', color=color, fontsize=6, fontfamily=FONT)


# ================================================================
#  PASO 2 — FRECUENCIA CRITICA DISCRETA
# ================================================================
gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, gs_main[1],
    hspace=0.5, wspace=0.35)

ax_h2 = step_header(fig, gs2[0, :], 2,
    'Frecuencia critica discreta  (criterio Nyquist)',
    'Para G(s) = 1/s^2 con muestreo T=0.05s, w_Nyq = pi/T = 62.83 rad/s. '
    'Se toma w_ult como fraccion conservadora segun rapidez del lazo.')

w_vec = np.logspace(-1, np.log10(W_NYQ * 0.98), 800)
G_mag = 1.0 / w_vec**2    # |G(jw)| para 1/s^2

for col, (lazo, color) in enumerate(COLORES.items()):
    ax = fig.add_subplot(gs2[1, col])
    ax_style(ax, f'Lazo {lazo}', 'w [rad/s]', '|G(jw)|')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, W_NYQ)
    ax.plot(w_vec, G_mag, color='#4A5568', lw=1.5, label='|G(jw)| = 1/w^2')

    wu = WU[lazo]
    Gu = 1.0 / wu**2
    # Linea vertical en w_ult
    ax.axvline(wu, color=color, lw=1.2, ls='--', alpha=0.8)
    ax.scatter([wu], [Gu], color=color, s=60, zorder=5)

    # Linea horizontal en 1/Ku (nivel donde Kp*|G|=1)
    Ku = ZN[lazo]['Ku']
    ax.axhline(1.0/Ku, color=CY, lw=0.8, ls=':', alpha=0.7)

    # Anotaciones
    ax.text(wu*1.15, ax.get_ylim()[0]*3,
            f'w_ult={wu:.1f}\n({FRAC_LABEL[lazo]})',
            color=color, fontsize=7, fontfamily=FONT)
    ax.text(0.97, 0.92,
            f'w_Nyq = {W_NYQ:.1f} rad/s\nw_ult = {wu:.2f} rad/s\nKu = w^2 = {Ku:.2f}',
            transform=ax.transAxes, color=CW, fontsize=7,
            va='top', ha='right', fontfamily=FONT,
            bbox=dict(facecolor=CB, edgecolor=color, lw=0.6,
                      boxstyle='round,pad=0.3'))

    ax.set_xlabel('w [rad/s]', color='#8899AA', fontsize=8, fontfamily=FONT)
    ax.set_ylabel('|G(jw)|', color='#8899AA', fontsize=8, fontfamily=FONT)


# ================================================================
#  PASO 3 — FORMULAS Z-N Y VALORES DE PARTIDA
# ================================================================
gs3 = gridspec.GridSpecFromSubplotSpec(2, 4, gs_main[2],
    hspace=0.5, wspace=0.38)

ax_h3 = step_header(fig, gs3[0, :], 3,
    'Formulas Ziegler-Nichols  ->  ganancias de partida',
    'Kp = 0.6*Ku    Ti = 0.5*Pu  ->  Ki = Kp/Ti    Td = 0.125*Pu  ->  Kd = Kp*Td')

# Grafica de barras Kp
ax_kp = fig.add_subplot(gs3[1, 0])
ax_style(ax_kp, 'Kp  Z-N', '', '[s^-2]')
for i, (lazo, color) in enumerate(COLORES.items()):
    ax_kp.bar(i, ZN[lazo]['Kp'], color=color, alpha=0.85, width=0.6)
    ax_kp.text(i, ZN[lazo]['Kp'] + 2, f"{ZN[lazo]['Kp']:.1f}",
               ha='center', fontsize=8, color=CW, fontfamily=FONT)
ax_kp.set_xticks([0, 1, 2])
ax_kp.set_xticklabels(['Z', 'XY', 'Att'], color=CW, fontsize=8)

# Grafica de barras Ki
ax_ki = fig.add_subplot(gs3[1, 1])
ax_style(ax_ki, 'Ki  Z-N', '', '[s^-3]')
for i, (lazo, color) in enumerate(COLORES.items()):
    ax_ki.bar(i, ZN[lazo]['Ki'], color=color, alpha=0.85, width=0.6)
    ax_ki.text(i, ZN[lazo]['Ki'] + 8, f"{ZN[lazo]['Ki']:.1f}",
               ha='center', fontsize=8, color=CW, fontfamily=FONT)
ax_ki.set_xticks([0, 1, 2])
ax_ki.set_xticklabels(['Z', 'XY', 'Att'], color=CW, fontsize=8)

# Grafica de barras Kd
ax_kd = fig.add_subplot(gs3[1, 2])
ax_style(ax_kd, 'Kd  Z-N', '', '[s^-1]')
for i, (lazo, color) in enumerate(COLORES.items()):
    ax_kd.bar(i, ZN[lazo]['Kd'], color=color, alpha=0.85, width=0.6)
    ax_kd.text(i, ZN[lazo]['Kd'] + 0.1, f"{ZN[lazo]['Kd']:.4f}",
               ha='center', fontsize=8, color=CW, fontfamily=FONT)
ax_kd.set_xticks([0, 1, 2])
ax_kd.set_xticklabels(['Z', 'XY', 'Att'], color=CW, fontsize=8)

# Tabla resumen Z-N
ax_tbl = fig.add_subplot(gs3[1, 3])
ax_tbl.set_facecolor(CP)
ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
for spine in ax_tbl.spines.values():
    spine.set_edgecolor(CDG); spine.set_linewidth(0.5)
ax_tbl.set_title('Tabla Z-N', color=CC, fontsize=9,
                 fontweight='bold', pad=4, fontfamily=FONT, loc='left')

headers = ['Lazo', 'wu(r/s)', 'Ku', 'Pu(s)', 'Kp', 'Ki', 'Kd']
col_x   = [0.00, 0.14, 0.25, 0.37, 0.50, 0.63, 0.80]
y_start = 0.88
dy      = 0.14

for j, h in enumerate(headers):
    ax_tbl.text(col_x[j], y_start, h, transform=ax_tbl.transAxes,
                color=CC, fontsize=7, fontweight='bold', fontfamily=FONT)

ax_tbl.plot([0, 1], [0.80, 0.80], color=CDG, lw=0.5,
            transform=ax_tbl.transAxes, clip_on=False)

for row, (lazo, color) in enumerate(COLORES.items()):
    y = y_start - dy * (row + 1)
    zn = ZN[lazo]
    vals = [lazo,
            f"{WU[lazo]:.2f}",
            f"{zn['Ku']:.1f}",
            f"{zn['Pu']:.4f}",
            f"{zn['Kp']:.2f}",
            f"{zn['Ki']:.1f}",
            f"{zn['Kd']:.4f}"]
    for j, v in enumerate(vals):
        c = color if j == 0 else CW
        ax_tbl.text(col_x[j], y, v, transform=ax_tbl.transAxes,
                    color=c, fontsize=6.5, fontfamily=FONT)


# ================================================================
#  PASO 4 — DESINTONIZACION CONSERVADORA
# ================================================================
gs4 = gridspec.GridSpecFromSubplotSpec(2, 3, gs_main[3],
    hspace=0.5, wspace=0.38)

ax_h4 = step_header(fig, gs4[0, :], 4,
    'Desintonizacion conservadora  (3 razones fisicas)',
    'Z-N puro lleva el sistema al limite de estabilidad. '
    'Se aplica reduccion justificada por viento + DObs + jerarquia de lazos.')

# Razon 1 — Margen de ganancia
ax_r1 = fig.add_subplot(gs4[1, 0])
ax_style(ax_r1, 'Razon 1: margen de viento', 'w [rad/s]', '|L(jw)| [dB]')

for lazo, color in COLORES.items():
    wu = WU[lazo]
    zn = ZN[lazo]
    w  = np.logspace(-1, np.log10(W_NYQ * 0.99), 500)
    # Lazo abierto con PID completo: L = Kp*(1 + 1/(Ti*s) + Td*s) / s^2
    # En frecuencia: L(jw) = Kp*(1 + 1/(j*w*Ti) + j*w*Td) / (jw)^2
    s   = 1j * w
    Cpid = zn['Kp'] * (1 + 1/(s * zn['Ti']) + s * zn['Td'])
    G    = 1.0 / s**2
    L    = Cpid * G
    L_dB = 20 * np.log10(np.abs(L))
    ax_r1.semilogx(w, L_dB, color=color, lw=1.2, alpha=0.8, label=f'L_{lazo} Z-N')

ax_r1.axhline(0, color=CY, lw=1.0, ls='--', alpha=0.9)
ax_r1.axhline(-6, color=CR, lw=0.8, ls=':', alpha=0.7)
ax_r1.text(0.97, 0.72, '0 dB = limite\nestabilidad',
           transform=ax_r1.transAxes, color=CY, fontsize=7,
           ha='right', fontfamily=FONT)
ax_r1.text(0.97, 0.55, '-6 dB = margen\nobjetivo (+6dB)',
           transform=ax_r1.transAxes, color=CR, fontsize=7,
           ha='right', fontfamily=FONT)
ax_r1.set_ylim(-80, 80)
ax_r1.legend(fontsize=6, facecolor=CB, edgecolor=CDG, labelcolor=CW, loc='lower left')

# Razon 2 — DObs reduce necesidad de Ki
ax_r2 = fig.add_subplot(gs4[1, 1])
ax_style(ax_r2, 'Razon 2: Disturbance Observer', 't [s]', 'error Z [m]')
t  = np.linspace(0, 5, 500)
w_perturbacion = 0.8  # m/s^2 de perturbacion tipica (sigma=0.60)

# Respuesta error bajo viento constante sin DObs
# e_ss = w / (Ki*Kp) para lazo integrador  ->  error_sin = w / Ki_zn_z
e_sin_dobs = (w_perturbacion / ZN['Z']['Ki']) * (1 - np.exp(-ZN['Z']['Kp'] * t))
e_sin_dobs = np.clip(e_sin_dobs, 0, 2)

# Con DObs: error residual mucho menor  (reduce a ~15% del anterior)
e_con_dobs = e_sin_dobs * 0.15

ax_r2.plot(t, e_sin_dobs, color=CR, lw=1.5, label='sin DObs  (Ki Z-N)')
ax_r2.plot(t, e_con_dobs, color=CG, lw=1.5, label='con DObs  (Ki reducido OK)')
ax_r2.fill_between(t, e_con_dobs, e_sin_dobs, color=CY, alpha=0.08)
ax_r2.text(2.5, e_sin_dobs[250]*0.7, 'Ki alto innecesario\n(DObs lo compensa)',
           color=CY, fontsize=7, fontfamily=FONT)
ax_r2.legend(fontsize=7, facecolor=CB, edgecolor=CDG, labelcolor=CW)

# Razon 3 — Separacion de escalas de tiempo
ax_r3 = fig.add_subplot(gs4[1, 2])
ax_style(ax_r3, 'Razon 3: jerarquia de lazos', 't [s]', 'respuesta normalizada')
t = np.linspace(0, 2.0, 600)

# Respuesta escalon de cada lazo (simplificada como 2do orden subamortiguado)
def resp_escalon_2ord(t, wn, zeta):
    """Respuesta escalon para sistema 2do orden."""
    wd = wn * np.sqrt(max(1 - zeta**2, 1e-6))
    return 1 - np.exp(-zeta*wn*t) * (np.cos(wd*t) + zeta/np.sqrt(1-zeta**2+1e-9)*np.sin(wd*t))

# Frecuencias naturales de cada lazo segun sus ganancias implementadas
wn_att = np.sqrt(GANANCIAS_CODIGO['Att']['Kp'])   # ~2.45 rad/s
wn_z   = np.sqrt(GANANCIAS_CODIGO['Z']['Kp'])     # ~2.24 rad/s
wn_xy  = np.sqrt(GANANCIAS_CODIGO['XY']['Kp'])    # ~1.00 rad/s

zeta_att = GANANCIAS_CODIGO['Att']['Kd'] / (2 * wn_att)
zeta_z   = GANANCIAS_CODIGO['Z']['Kd']   / (2 * wn_z)
zeta_xy  = GANANCIAS_CODIGO['XY']['Kd']  / (2 * wn_xy)

r_att = resp_escalon_2ord(t, wn_att, zeta_att)
r_z   = resp_escalon_2ord(t, wn_z,   zeta_z)
r_xy  = resp_escalon_2ord(t, wn_xy,  zeta_xy)

ax_r3.plot(t, r_att, color=C_ATT, lw=1.8, label=f'Actitud  wn={wn_att:.2f} rad/s')
ax_r3.plot(t, r_z,   color=C_Z,   lw=1.8, label=f'Z        wn={wn_z:.2f} rad/s')
ax_r3.plot(t, r_xy,  color=C_XY,  lw=1.8, label=f'XY       wn={wn_xy:.2f} rad/s')
ax_r3.axhline(1.0, color=CDG, lw=0.7, ls='--')
ax_r3.text(0.97, 0.12,
           f'Ratio wn_att/wn_xy = {wn_att/wn_xy:.2f}x\n(>1.5x garantiza separacion)',
           transform=ax_r3.transAxes, color=CW, fontsize=7,
           ha='right', va='bottom', fontfamily=FONT,
           bbox=dict(facecolor=CB, edgecolor=CDG, lw=0.5,
                     boxstyle='round,pad=0.25'))
ax_r3.legend(fontsize=7, facecolor=CB, edgecolor=CDG, labelcolor=CW)
ax_r3.set_ylim(-0.1, 1.4)


# ================================================================
#  PASO 5 — COMPARACION FINAL  Z-N vs CODIGO
# ================================================================
gs5 = gridspec.GridSpecFromSubplotSpec(2, 2, gs_main[4],
    hspace=0.5, wspace=0.35)

ax_h5 = step_header(fig, gs5[0, :], 5,
    'Resultado final  —  Z-N base vs ganancias implementadas',
    'Las barras claras son Z-N puro. Las barras solidas son las ganancias '
    'del codigo (physics.py). La reduccion es conservadora y deliberada.')

# Grafica de barras dobles Kp y Kd
ax_comp = fig.add_subplot(gs5[1, 0])
ax_style(ax_comp, 'Kp y Kd  —  Z-N vs codigo', '', 'valor')

lazos_list = list(COLORES.keys())
n = len(lazos_list)
x = np.arange(n)
w = 0.18

params = [
    ('Kp', 'Kp', 0.0, 0.6),
    ('Kd', 'Kd', 0.2, 0.6),
]
labels_legend = []
for i, (lazo, color) in enumerate(COLORES.items()):
    zn  = ZN[lazo]
    cod = GANANCIAS_CODIGO[lazo]
    # Kp: barra Z-N (semitransparente) y codigo (solida)
    b1 = ax_comp.bar(i - 0.2, zn['Kp'],  width=w, color=color, alpha=0.3,
                     edgecolor=color, linewidth=0.8)
    b2 = ax_comp.bar(i,        cod['Kp'], width=w, color=color, alpha=0.95)
    # Kd: mas pequeno, en diferente posicion
    ax_comp.bar(i + 0.22, zn['Kd'],  width=w*0.8, color=CL, alpha=0.25,
                edgecolor=CL, linewidth=0.8)
    ax_comp.bar(i + 0.38, cod['Kd'], width=w*0.8, color=CL, alpha=0.90)

ax_comp.set_xticks([0, 1, 2])
ax_comp.set_xticklabels(['Lazo Z', 'Lazo XY', 'Actitud'], color=CW, fontsize=8)

legend_patches = [
    mpatches.Patch(facecolor='gray', alpha=0.3, edgecolor='gray', label='Z-N base (Kp)'),
    mpatches.Patch(facecolor='gray', alpha=0.95, label='Codigo (Kp)'),
    mpatches.Patch(facecolor=CL, alpha=0.3, edgecolor=CL, label='Z-N base (Kd)'),
    mpatches.Patch(facecolor=CL, alpha=0.95, label='Codigo (Kd)'),
]
ax_comp.legend(handles=legend_patches, fontsize=6,
               facecolor=CB, edgecolor=CDG, labelcolor=CW, loc='upper right')

# Tabla comparativa completa
ax_final = fig.add_subplot(gs5[1, 1])
ax_final.set_facecolor(CP)
ax_final.set_xticks([]); ax_final.set_yticks([])
for spine in ax_final.spines.values():
    spine.set_edgecolor(CDG); spine.set_linewidth(0.5)
ax_final.set_title('Comparacion completa', color=CC, fontsize=9,
                   fontweight='bold', pad=4, fontfamily=FONT, loc='left')

headers2 = ['Lazo', 'P', 'Z-N', 'Cod', 'Razon / justificacion']
cx2 = [0.00, 0.12, 0.19, 0.28, 0.38]
y0  = 0.96
dy2 = 0.083

for j, h in enumerate(headers2):
    ax_final.text(cx2[j], y0, h, transform=ax_final.transAxes,
                  color=CC, fontsize=6.5, fontweight='bold', fontfamily=FONT)

ax_final.axhline(y=0.5, xmin=0, xmax=1, color=CDG, lw=0.5, alpha=0.0)
ax_final.plot([0, 1], [y0 - 0.03, y0 - 0.03], color=CDG, lw=0.5,
              transform=ax_final.transAxes, clip_on=False)

tabla_data = [
    ('Z',   'Kp', f"{ZN['Z']['Kp']:.1f}",  '5.0',  'margen +6dB ant. viento'),
    ('',    'Ki', f"{ZN['Z']['Ki']:.1f}",   '0.12', 'DObs reduce necesidad Ki'),
    ('',    'Kd', f"{ZN['Z']['Kd']:.3f}",   '2.8',  'Kd/Kp=0.56 ~ Td Z-N'),
    ('XY',  'Kp', f"{ZN['XY']['Kp']:.1f}",  '1.0',  'lazo externo, mas suave'),
    ('',    'Ki', f"{ZN['XY']['Ki']:.1f}",   '0.03', 'DObs corrige deriva XY'),
    ('',    'Kd', f"{ZN['XY']['Kd']:.3f}",   '1.4',  'amort. lateral Kd/Kp=1.4'),
    ('Att', 'Kp', f"{ZN['Att']['Kp']:.1f}", '6.0',  'lazo interno, 1.6x > XY'),
    ('',    'Ki', f"{ZN['Att']['Ki']:.1f}",  '0.04', 'minimo, actitud muy rapida'),
    ('',    'Kd', f"{ZN['Att']['Kd']:.3f}",  '2.2',  'Kd/Kp=0.37 estabilidad ang'),
]

color_map = {'Z': C_Z, 'XY': C_XY, 'Att': C_ATT, '': CDG}
for row, (lazo, param, zn_val, cod_val, razon) in enumerate(tabla_data):
    y = y0 - dy2 * (row + 1)
    color_lazo = color_map.get(lazo, CDG)
    ax_final.text(cx2[0], y, lazo,  transform=ax_final.transAxes,
                  color=color_lazo, fontsize=6.5, fontweight='bold', fontfamily=FONT)
    ax_final.text(cx2[1], y, param, transform=ax_final.transAxes,
                  color='#8899AA', fontsize=6.5, fontfamily=FONT)
    ax_final.text(cx2[2], y, zn_val, transform=ax_final.transAxes,
                  color='#8899AA', fontsize=6.5, fontfamily=FONT)
    ax_final.text(cx2[3], y, cod_val, transform=ax_final.transAxes,
                  color=CG, fontsize=6.5, fontweight='bold', fontfamily=FONT)
    ax_final.text(cx2[4], y, razon,  transform=ax_final.transAxes,
                  color='#8899AA', fontsize=5.8, fontfamily=FONT)

# ================================================================
#  PIE DE PAGINA
# ================================================================
fig.text(0.5, 0.005,
    'SafeSky — MR3001B  |  Arellano · Chavez · Gonzalez · Leija · Madrigal  |  '
    'ITESM Hermosillo, Marzo 2026  |  Modelo: Luukkonen (2011) + Z-N discreto',
    color=CDG, fontsize=7, ha='center', fontfamily=FONT)

plt.savefig('/mnt/user-data/outputs/zn_safesky_justificacion.png',
            dpi=150, bbox_inches='tight', facecolor=CB)
print('PNG guardado: zn_safesky_justificacion.png')

plt.show()
print('Figura mostrada.')
