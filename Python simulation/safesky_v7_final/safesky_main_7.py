"""
SafeSky - Simulador de Enjambre de Drones 6-DOF  v9
MR3001B . Tecnológico de Monterrey 2026
=====================================================
NOVEDADES v9:
  * Objeto intrusor externo: cruza la trayectoria del enjambre a velocidad
    configurable; boton "LANZAR INTRUSO" durante la simulacion.
  * Botón " RÁFAGA" para inyectar una ráfaga de viento lateral súbita.
  * Modo misión "A->B línea recta": cada dron vuela directamente a su meta
    sin Hybrid A* (solo ORCA + PID). Útil para ver evasión pura.
  * Tab Rendimiento ampliado: error de velocidad de seguimiento (|v-v_cmd|),
    error de distancia al goal en cada paso, e índice de perturbación de viento.
  * Intruso renderizado en 3D (esfera naranja) con trail; se reporta en KPI.
  * (v8) Obstáculos volumétricos, progreso GIF/MP4, cruzado, orden, leyenda.
"""
import sys, os, time, warnings, io, base64
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
# Usar TkAgg si hay display disponible, Agg si es headless (exportación sin GUI)
try:
    import tkinter as _tk_test; _tk_test.Tk().destroy(); del _tk_test
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')   # modo headless - exportación GIF sin ventana
import matplotlib.pyplot as plt
import matplotlib.animation as mplanim
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics import QuadParams, DroneState, QuadDynamics, PIDController, WindAR1, \
                   ORCA_RADIUS, R_DRONE_BODY, D_COLLISION, SAFETY_BUFFER
from planner import Obstacle, HybridAStar, ORCAPlanner
from kpis    import KPITracker, MissionKPI, D_MIN, D_SAFE, E_B_OK, \
                   DRONE_COLLISION_D, D_COLLISION_PHYSICAL, D_SAFETY_BUFFER, R_DRONE, DRONE_RADIUS_PHYS

# ==========================================================
#  CONSTANTES
# ==========================================================
SPACE     = 20.0
DT        = 0.05
DT_ANIM   = 60
SUCCESS_M      = 0.50   # radio de llegada [m]
# WP_ACCEPT separado por modo:
#   ab    : 1.00m — obs_push puede desviar ~0.8m, necesita margen amplio
#   circle: 0.30m — cuerda entre WPs = 0.87m con n_wp=36, ratio=0.34 -> circulo preciso
WP_ACCEPT      = 1.00   # modo ab
WP_ACCEPT_CIRC = 0.30   # modo circular
V_MAX_NAV      = 2.0
TRAIL_LEN      = 120
HOVER_STEPS    = 120      # 1s consecutivo dentro de SUCCESS_M (era 150 = 7.5s imposible)
# FIX: timeout de waypoint - si el dron lleva más de WP_TIMEOUT_S segundos
# sin llegar al WP actual, lo salta. Evita bloqueo orbital infinito.
WP_TIMEOUT_S   = 8.0    # segundos máximos en un mismo waypoint
HOVER_TOLERANCE= 8      # pasos fuera de SUCCESS_M que no resetean el progreso

# Paleta
CB='#0A0E1A'; CP='#0F1525'; CB2='#141C30'
CW='#E8EEF4'; CG='#2ECC71'; CR='#E74C3C'
CY='#F39C12'; CC='#00E5FF'; CL='#3498DB'
CDG='#7F8C8D'; CO='#E67E22'

DRONE_COLORS   = ['#00E5FF','#FF6B6B','#A8FF78','#FFD700','#B388FF','#FF8A65']
INTRUDER_COLOR = '#FF8C00'   # naranja brillante para el objeto intruso

# Ráfaga de viento súbita: duración y magnitud
GUST_STEPS  = 60    # 3 s a DT=0.05 (más duración = más visible en gráficas)
GUST_MAG    = 8.0   # m/s lateral - genera F~1.9N -> a~1.9m/s2, claramente visible
FONT_H = ('Segoe UI', 10, 'bold')
FONT_B = ('Segoe UI',  9)
FONT_M = ('Consolas',  9)
FONT_S = ('Consolas',  8)
FONT_BIG = ('Consolas', 10)


# ==========================================================
#  GENERADOR DE WAYPOINTS CIRCULAR
# ==========================================================
def circular_waypoints(center, radius, n_points, z_base, drone_idx, n_drones):
    """
    Genera waypoints circulares para un dron.
    Cada dron parte con un desfase de fase para mantener separación.
    """
    phase_offset = 2 * np.pi * drone_idx / n_drones
    angles = np.linspace(phase_offset, phase_offset + 2*np.pi, n_points + 1)[:-1]
    wps = []
    for a in angles:
        x = center[0] + radius * np.cos(a)
        y = center[1] + radius * np.sin(a)
        z = z_base + drone_idx * 0.5   # altitud escalonada
        wps.append(np.array([x, y, z]))
    # Añadir punto de cierre
    wps.append(wps[0].copy())
    return wps


# ==========================================================
#  AGENTE DRON  (soporta modo A->B y CIRCULAR)
# ==========================================================
class DroneAgent:
    """
    Dron con dinámica 6-DOF DJI F450.
    Modo 'ab': navega A -> B con waypoints A*.
    Modo 'circle': sigue una trayectoria circular cerrada.
    """
    def __init__(self, did, start, goal_or_wps, params: QuadParams,
                 wind_alpha, wind_sigma, color, mode='ab'):
        self.id       = did
        self.color    = color
        self.mode     = mode        # 'ab' | 'circle'
        self.params   = params
        self.dyn      = QuadDynamics(params)
        self.pid      = PIDController(params)
        self.wind_gen = WindAR1(wind_alpha, wind_sigma, seed=did*31+7)
        init = np.array(start, float)
        self.state    = DroneState(
            pos=init.copy(), vel=np.zeros(3),
            angles=np.zeros(3), dAngles=np.zeros(3))
        self.wps           = []
        self.wp_idx        = 0
        self.laps          = 0       # vueltas completadas (modo circular)
        self.arrived       = False
        self.traj          = [init.copy()]
        self.L             = 0.0
        self.min_dist_goal = float('inf')
        self._hover_steps  = 0
        self._hover_out    = 0   # pasos consecutivos fuera del radio (tolerancia)
        self.takeoff_done  = False
        # Timeout de waypoint - evita bloqueo orbital infinito
        # Si el dron lleva > WP_TIMEOUT_S segundos sin llegar al WP actual, lo salta
        self._wp_time      = 0.0   # tiempo acumulado en el WP actual [s]
        # Takeoff a la altitud del goal -> cada dron vuela en su banda Z propia
        # garantizando >=1.5m de separación vertical con la configuración actual
        if mode == 'ab':
            goal_z = float(np.array(goal_or_wps, float)[2])
            self.takeoff_alt = max(goal_z, init[2] + 1.5)
        else:
            # Circular: despegar directo a la altitud de los waypoints (z_base ? 9m)
            # para que el ascenso vertical no pase por los obstáculos a z<8m
            first_wp_z = float(np.array(goal_or_wps[0], float)[2])
            self.takeoff_alt = max(first_wp_z, init[2] + 2.0)

        if mode == 'ab':
            self.goal = np.array(goal_or_wps, float)
        else:
            # En modo circular, goal_or_wps es lista de waypoints circulares
            self.goal = None
            self.set_waypoints(goal_or_wps)

        self.hist_angles = []
        self.hist_wind   = []
        self.hist_vel    = []    # velocidad vectorial [m/s]
        self.hist_speed  = []    # rapidez escalar [m/s]
        self.hist_thrust = []    # empuje u1 [N]
        self.hist_vel_err= []    # error de tracking de velocidad |v - v_cmd| [m/s]
        self.hist_dist_goal=[]   # distancia al goal en cada paso [m]
        self.hist_wind_mag = []  # magnitud del viento en cada paso [m/s]

    def set_waypoints(self, wps):
        self.wps    = [np.array(w, float) for w in wps]
        self.wp_idx = 0

    def current_wp(self):
        if not self.wps or self.wp_idx >= len(self.wps):
            return (self.goal.copy() if self.goal is not None
                    else self.wps[0].copy())
        return self.wps[self.wp_idx].copy()

    def step(self, dt, orca_delta=None, obs_push=None, gust=None):
        """
        orca_delta : delta de velocidad ORCA inter-dron [m/s]  -> desvio lateral del target
        obs_push   : vector de empuje de obstaculos [m]        -> desplaza target directamente
        gust       : vector de ráfaga súbita [m/s]             -> se suma al viento AR(1)
                     FIX: antes se hackeaba vel[] directamente; ahora entra en step_rk4
                     y queda correctamente registrado en hist_wind_mag.
        """
        if self.arrived:
            return

        # ?? Takeoff ??????????????????????????????????????
        if not self.takeoff_done:
            target = np.array([self.state.pos[0], self.state.pos[1],
                                self.takeoff_alt])
            if self.state.pos[2] >= self.takeoff_alt - 0.25:
                self.takeoff_done = True
                self.pid.reset()
        else:
            target = self.current_wp().copy()
            dist_to_goal = (np.linalg.norm(self.state.pos - self.goal)
                            if self.goal is not None else 99.)
            # near_goal=2m: no desviar cerca del destino para convergencia limpia
            near_goal = (dist_to_goal < 2.0) if self.mode == 'ab' else False

            if not near_goal:
                # ?? ORCA inter-dron: desvio lateral del target ??
                # Solo componente perpendicular a la dirección de avance
                if orca_delta is not None:
                    d_mag = np.linalg.norm(orca_delta)
                    if d_mag > 0.05:
                        dir_f = target - self.state.pos
                        dn    = np.linalg.norm(dir_f)
                        if dn > 0.1:
                            du   = dir_f / dn
                            perp = orca_delta - np.dot(orca_delta, du) * du
                            # Escalar: velocidad [m/s] -> desplazamiento target [m]
                            target = target + np.clip(perp * 1.5, -2.0, 2.0)

                # ?? Empuje de obstáculos: mover target FUERA del obstaculo ??
                # obs_push viene en metros desde Simulation.step().
                # En modo A->B: hasta 4m (para rodear obstáculos grandes).
                # En modo circular: máx 1.5m (solo corregir deriva; la
                #   trayectoria ya está a z=9m lejos de obstáculos).
                # obs_push: solo en modo A->B donde el dron puede acercarse
                # a obstáculos siguiendo waypoints A*. En modo circular la
                # trayectoria orbita a z=9m (obstáculos a z?4-6m) y ORCA
                # maneja la evasión en velocidad - obs_push interferirÍa
                # con la aceptación de waypoints.
                if obs_push is not None and self.mode == 'ab':
                    pmag = np.linalg.norm(obs_push)
                    if pmag > 0.01:
                        target = target + np.clip(obs_push, -4.0, 4.0)
                        target = np.clip(target, 0.5, SPACE - 0.5)
                        target[2] = np.clip(target[2], 1.0, SPACE - 1.0)

        # ?? Física ???????????????????????????????????????
        w = self.wind_gen.step()
        # FIX: sumar ráfaga al viento AR(1) ANTES del integrador RK4
        # -> la ráfaga pasa por _deriv -> WindAR1.wind_force() correctamente
        # -> queda registrada en hist_wind_mag (antes no aparecía)
        if gust is not None and np.linalg.norm(gust) > 1e-6:
            w = w + gust
        # Guardar velocidad de comando antes de step (para error de tracking)
        v_cmd_mag = np.linalg.norm(target - self.state.pos)
        v_cmd_mag = min(v_cmd_mag, V_MAX_NAV)
        u1, u2, u3, u4 = self.pid.compute(self.state, target, dt=dt)
        self.state = self.dyn.step_rk4(self.state, u1, u2, u3, u4, w, dt)

        # Histórico
        self.traj.append(self.state.pos.copy())
        if len(self.traj) >= 2:
            self.L += np.linalg.norm(self.traj[-1] - self.traj[-2])
        self.hist_angles.append(self.state.angles.copy())
        self.hist_wind.append(w.copy())
        self.hist_vel.append(self.state.vel.copy())
        v_actual = float(np.linalg.norm(self.state.vel))
        self.hist_speed.append(v_actual)
        self.hist_thrust.append(float(u1))
        self.hist_vel_err.append(abs(v_actual - v_cmd_mag))
        if self.goal is not None:
            self.hist_dist_goal.append(float(np.linalg.norm(self.state.pos - self.goal)))
        elif self.wps:
            self.hist_dist_goal.append(float(np.linalg.norm(self.state.pos - self.wps[0])))
        # Magnitud del viento TOTAL (AR1 + ráfaga) - FIX: antes solo registraba AR1
        self.hist_wind_mag.append(float(np.linalg.norm(w)))

        # Distancia mínima al goal (modo A->B)
        if self.goal is not None:
            d = np.linalg.norm(self.state.pos - self.goal)
            if d < self.min_dist_goal:
                self.min_dist_goal = d

        # ?? Avance de waypoint con timeout anti-bloqueo ??????
        if self.takeoff_done and self.wps:
            wp_now   = self.wps[self.wp_idx % len(self.wps)]
            dist_wp  = np.linalg.norm(self.state.pos - wp_now)
            # Umbral de aceptacion diferente segun modo:
            # circular necesita precision fina para que la forma sea un circulo
            wp_thr = WP_ACCEPT_CIRC if self.mode == 'circle' else WP_ACCEPT

            if dist_wp < wp_thr:
                # Llegó al WP -> avanzar
                self.wp_idx += 1
                self._wp_time = 0.0   # resetear timer
                # Modo circular: reiniciar índice = nueva vuelta
                if self.mode == 'circle' and self.wp_idx >= len(self.wps):
                    self.wp_idx = 0
                    self.laps  += 1
                    if self.laps >= 2:
                        self.arrived = True
            else:
                # Acumular tiempo en este WP
                self._wp_time += dt
                # FIX: timeout - si lleva demasiado tiempo sin llegar, saltarlo
                # Evita bloqueo orbital cuando obs_push impide alcanzar el WP
                if self._wp_time > WP_TIMEOUT_S:
                    self.wp_idx  += 1
                    self._wp_time = 0.0

        # ?? Llegada en A->B con hover de convergencia ??????
        if self.mode == 'ab' and self.goal is not None:
            if np.linalg.norm(self.state.pos - self.goal) < SUCCESS_M:
                self._hover_steps += 1
                self._hover_out    = 0
                if self._hover_steps >= HOVER_STEPS:
                    self.arrived = True
            else:
                self._hover_out += 1
                # Solo resetear si lleva MÁS de HOVER_TOLERANCE pasos fuera
                if self._hover_out > HOVER_TOLERANCE:
                    self._hover_steps = 0


# ==========================================================
#  OBJETO INTRUSOR EXTERNO
# ==========================================================
class Intruder:
    """
    Objeto externo (dron/pájaro) que cruza el espacio de simulación
    a velocidad constante en línea recta, disparado manualmente.
    Casos de prueba: 'Un objeto externo cruza la trayectoria del enjambre
    de forma aleatoria a una velocidad de x m/s'.
    """
    RADIUS = 0.40   # m - radio físico del intruso (para detección)

    def __init__(self, speed: float = 3.0, rng=None):
        rng = rng or np.random.RandomState()
        # Elegir un lado de entrada aleatorio (0=X-, 1=X+, 2=Y-, 3=Y+)
        side = rng.randint(4)
        z    = rng.uniform(3.0, 8.0)
        if side == 0:
            self.pos = np.array([0.5, rng.uniform(3, 17), z])
            self.vel = np.array([speed, rng.uniform(-0.3, 0.3)*speed,
                                  rng.uniform(-0.2, 0.2)*speed])
        elif side == 1:
            self.pos = np.array([SPACE-0.5, rng.uniform(3, 17), z])
            self.vel = np.array([-speed, rng.uniform(-0.3, 0.3)*speed,
                                   rng.uniform(-0.2, 0.2)*speed])
        elif side == 2:
            self.pos = np.array([rng.uniform(3, 17), 0.5, z])
            self.vel = np.array([rng.uniform(-0.3, 0.3)*speed, speed,
                                  rng.uniform(-0.2, 0.2)*speed])
        else:
            self.pos = np.array([rng.uniform(3, 17), SPACE-0.5, z])
            self.vel = np.array([rng.uniform(-0.3, 0.3)*speed, -speed,
                                  rng.uniform(-0.2, 0.2)*speed])
        self.speed     = speed
        self.active    = True
        self.traj      = [self.pos.copy()]
        self.min_sep   = float('inf')   # separación mínima con cualquier dron

    def step(self, dt, drones):
        if not self.active:
            return
        self.pos = self.pos + self.vel * dt
        self.traj.append(self.pos.copy())
        # Detectar proximidad a drones
        for d in drones:
            sep = np.linalg.norm(self.pos - d.state.pos)
            if sep < self.min_sep:
                self.min_sep = sep
        # Desactivar si sale del espacio
        if (self.pos[0] < -1 or self.pos[0] > SPACE+1 or
                self.pos[1] < -1 or self.pos[1] > SPACE+1):
            self.active = False


# ==========================================================
#  SIMULACIÓN
# ==========================================================
class Simulation:
    def __init__(self, n_drones, n_obs, wind_alpha, wind_sigma,
                 obs_mode='fixed', mission_mode='ab',
                 drone_order='ltr', goal_side='opposite'):
        self.n_drones     = n_drones
        self.n_obs        = n_obs
        self.wind_alpha   = wind_alpha
        self.wind_sigma   = wind_sigma
        self.obs_mode     = obs_mode      # 'fixed' | 'random' | 'dynamic'
        # ab_straight: A->B en línea recta sin A* (solo ORCA+PID)
        self._straight_mode = (mission_mode == 'ab_straight')
        self.mission_mode = ('ab' if mission_mode in ('ab_cross', 'ab_straight')
                             else mission_mode)
        self._mission_mode_orig = mission_mode   # para labels / KPI
        self._cross_mode  = (mission_mode == 'ab_cross')
        self.drone_order  = drone_order   # 'ltr' | 'rtl'
        self.goal_side    = goal_side     # 'opposite' | 'same'
        self.t            = 0.0
        self.step_count   = 0
        self.params       = QuadParams()
        # FIX: tau reducido 4.0->2.0 - con tau=4 ORCA actuaba a 8m de distancia
        # generando empujes permanentes que competían con obs_push -> bloqueo orbital
        self.orca         = ORCAPlanner(radius=ORCA_RADIUS, tau=2.0, v_max=V_MAX_NAV)
        self.mission_kpi  = None
        self._obs_seed    = 42 if obs_mode == 'fixed' else int(time.time() * 1000) % 100000
        # Intruso y ráfaga
        self.intruder     = None          # se activa con launch_intruder()
        self._gust_steps  = 0             # pasos restantes de ráfaga activa
        self._gust_vec    = np.zeros(3)   # vector de ráfaga lateral
        self._setup()

    def _setup(self):
        rng_seed = 42 if self.obs_mode == 'fixed' else self._obs_seed
        np.random.seed(rng_seed)
        self.obstacles = self._gen_obstacles(rng_seed)
        # Pasar obstáculos al ORCA para que también los evite
        self.orca.obstacles = self.obstacles
        n = self.n_drones

        # Posiciones inicio y meta - drone_order y goal_side configurables
        # drone_order='ltr': D0 en X=2 (izquierda), D(n-1) en X=2+3*(n-1)
        # drone_order='rtl': D0 en X=18 (derecha), D(n-1) en X=18-3*(n-1)
        # goal_side='opposite': metas en el lado contrario en X (cruce)
        # goal_side='same': metas en el mismo lado en X que los inicios
        # _cross_mode: los goals se invierten entre drones (D0?D(n-1))
        if self.drone_order == 'ltr':
            self.starts = [np.array([2.0  + 5*i, 1.5, 0.5]) for i in range(n)]
            goals_base  = [np.array([2.0  + 3.0*i, 18.5, 7.5 + i*0.5]) for i in range(n)]
        else:  # rtl
            self.starts = [np.array([18.0 - 5*i, 1.5, 0.5]) for i in range(n)]
            goals_base  = [np.array([18.0 - 3.0*i, 18.5, 7.5 + i*0.5]) for i in range(n)]

        if self.goal_side == 'opposite':
            # Metas en el lado opuesto en X a los inicios
            if self.drone_order == 'ltr':
                goals_base = [np.array([18.0 - 4*i, 18.5, 7.5 + i*0.5]) for i in range(n)]
            else:
                goals_base = [np.array([2.0  + 4*i, 18.5, 7.5 + i*0.5]) for i in range(n)]

        if self._cross_mode:
            # Cruzado: D0 va a la meta de D(n-1), D1 va a meta de D(n-2), etc.
            self.goals = [goals_base[n - 1 - i] for i in range(n)]
        else:
            self.goals = goals_base

        if self.mission_mode == 'ab':
            self.drones = [
                DroneAgent(i, self.starts[i], self.goals[i], self.params,
                           self.wind_alpha, self.wind_sigma,
                           DRONE_COLORS[i % len(DRONE_COLORS)], mode='ab')
                for i in range(n)
            ]
            if self._straight_mode:
                # Modo línea recta: el waypoint único es la meta directa.
                # No se usa A* - el dron vuela directo y ORCA gestiona evasión.
                for d in self.drones:
                    wps = [d.goal.copy()]
                    d.set_waypoints(wps)
                self._total_nodes = 0
            else:
                # Planificar rutas A*
                astar = HybridAStar(self.obstacles)
                total_nodes = 0
                for d in self.drones:
                    s3       = self.starts[d.id].copy()
                    s3[2]    = d.takeoff_alt
                    wps      = astar.plan(s3, d.goal)
                    d.set_waypoints(wps)
                    total_nodes += astar.nodes_used
                self._total_nodes = total_nodes
        else:
            # Modo circular: orbitan el centro del espacio a z=9m (sobre obstáculos)
            cx, cy = SPACE / 2, SPACE / 2
            radius = 5.0
            # 36 puntos = un WP cada 10 grados -> cuerda=0.87m
            # Con WP_ACCEPT_CIRC=0.30m el dron solo acepta el WP
            # cuando ya llego (~35% de la cuerda) -> circulo bien definido
            n_wp   = 36
            z_base = 9.0   # encima de todos los obstáculos (max obs_z ~7.85m)
            self.drones = []
            for i in range(n):
                circ_wps = circular_waypoints(
                    np.array([cx, cy]), radius, n_wp, z_base, i, n)
                # Punto de inicio XY = primer waypoint del círculo.
                # Si el ascenso vertical desde esa posición pasa por un obstáculo,
                # desplazar radialmente hacia afuera hasta que el ascenso sea libre.
                start_xy = circ_wps[0][:2].copy()
                for _ in range(12):
                    ascent = [np.array([start_xy[0], start_xy[1], z])
                               for z in np.arange(0.5, z_base, 0.4)]
                    if (not self.obstacles or
                            all(min(o.dist_surface(p) for o in self.obstacles) >= 0.35
                                for p in ascent)):
                        break
                    dr = start_xy - np.array([cx, cy])
                    dr_n = np.linalg.norm(dr)
                    start_xy += (dr/dr_n if dr_n>1e-6 else np.array([1.,0.])) * 0.5
                start_c = np.array([start_xy[0], start_xy[1], 0.5])
                d = DroneAgent(
                    i, start_c, circ_wps, self.params,
                    self.wind_alpha, self.wind_sigma,
                    DRONE_COLORS[i % len(DRONE_COLORS)], mode='circle')
                self.drones.append(d)
            self._total_nodes = 0

        self.kpi_tracker = KPITracker(
            n, [d.goal if d.goal is not None else d.wps[0]
                for d in self.drones],
            self.obstacles)

        # Para obstáculos dinámicos: velocidades aleatorias
        if self.obs_mode == 'dynamic':
            self._obs_vel = [
                np.array([np.random.uniform(-0.3, 0.3),
                          np.random.uniform(-0.3, 0.3), 0.0])
                for _ in self.obstacles
            ]
        else:
            self._obs_vel = [np.zeros(3) for _ in self.obstacles]

    def _gen_obstacles(self, seed=42):
        rng  = np.random.RandomState(seed)
        obs, placed = [], []
        tries = 0
        while len(obs) < self.n_obs and tries < 800:
            tries += 1
            c = rng.uniform([4., 4., 3.2], [16., 16., 7.0])
            r = rng.uniform(0.5, 1.2, 3)
            ok = all(np.linalg.norm(c - p) >= 3.0 for p in placed)
            if not ok: continue
            for i in range(6):
                if (np.linalg.norm(c - np.array([2.+3.*i, 1.5, 2.])) < 3.0 or
                        np.linalg.norm(c - np.array([18.-3.*i, 18.5, 4.5])) < 3.0 or
                        np.linalg.norm(c - np.array([SPACE/2, SPACE/2, 4.])) < 4.5):
                    ok = False; break
            if ok:
                obs.append(Obstacle(c, r))
                placed.append(c.copy())
        return obs

    def launch_intruder(self, speed: float = 3.0):
        """Lanza un objeto intrusor externo desde un lado aleatorio."""
        rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
        self.intruder = Intruder(speed=speed, rng=rng)

    def trigger_gust(self):
        """
        Inyecta una ráfaga de viento lateral súbita durante GUST_STEPS pasos.
        El vector de ráfaga es perpendicular al eje Y (lateral puro en X o Z).
        Caso de prueba: 'Aplicación de ráfagas de viento laterales súbitas'.
        """
        rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
        axis = rng.choice([0, 2])   # X o Z
        sign = rng.choice([-1, 1])
        self._gust_vec = np.zeros(3)
        self._gust_vec[axis] = sign * GUST_MAG
        self._gust_steps = GUST_STEPS

    def step(self):
        t0 = time.perf_counter()

        # Mover obstáculos dinámicos
        if self.obs_mode == 'dynamic':
            for i, o in enumerate(self.obstacles):
                o.center += self._obs_vel[i] * DT
                # Rebotar en bordes
                for ax in range(3):
                    if o.center[ax] < 3.0 or o.center[ax] > SPACE - 3.0:
                        self._obs_vel[i][ax] *= -1
                o.center[2] = max(3.2, min(7.0, o.center[2]))

        pos     = [d.state.pos.copy() for d in self.drones]
        
        # ?? Pre-calcular obs_push para cada dron ??????????????
        # FIX: ORCA debe recibir velocidades preferidas que YA incluyen
        # la influencia de obstáculos cercanos. Antes, ORCA recibía la
        # v_pref pura (hacia el WP) mientras obs_push desviaba el target del PID
        # -> ambos sistemas actuaban con información inconsistente -> bloqueo orbital.
        obs_push_pre = []
        for i, d in enumerate(self.drones):
            op = np.zeros(3)
            if not d.arrived and d.takeoff_done:
                near_goal = (d.goal is not None and
                             np.linalg.norm(d.state.pos - d.goal) < SUCCESS_M * 2.0)
                if not near_goal:
                    D_WARN, D_EMERG = 2.5, 0.6
                    for o in self.obstacles:
                        ds = o.dist_surface(d.state.pos)
                        if ds >= D_WARN: continue
                        grad = d.state.pos - o.center
                        gn   = np.linalg.norm(grad)
                        if gn < 1e-6: grad = np.array([0., 0., 1.]); gn = 1.0
                        dirn = grad / gn
                        gain = 4.0 if ds <= D_EMERG else (
                            4.0 * (1-(ds-D_EMERG)/(D_WARN-D_EMERG)) +
                            0.5 * ((ds-D_EMERG)/(D_WARN-D_EMERG)))
                        op += dirn * gain
            obs_push_pre.append(op)

        v_prefs = []
        for i, d in enumerate(self.drones):
            if d.arrived or not d.takeoff_done:
                v_prefs.append(np.zeros(3))
            else:
                wp  = d.current_wp()
                # Ajustar WP destino con obs_push antes de calcular v_pref
                wp_adj = wp + np.clip(obs_push_pre[i], -3.0, 3.0)
                wp_adj = np.clip(wp_adj, 0.5, SPACE - 0.5)
                dv = wp_adj - d.state.pos
                dn = np.linalg.norm(dv)
                # En modo circular limitar a 1.2 m/s: a 2 m/s el PID no
                # tiene tiempo de girar entre WPs de 0.87m (t_wp=0.44s),
                # lo que genera una trayectoria poligonal en lugar de circular.
                v_cap = 1.2 if d.mode == 'circle' else V_MAX_NAV
                v_prefs.append(dv/dn * min(v_cap, dn) if dn > 0.01 else np.zeros(3))

        v_orca = self.orca.compute_velocities(pos, v_prefs)

        # ?? Calcular ráfaga actual ???????????????????????????
        # FIX: antes la ráfaga se aplicaba DESPUÉS del loop hackeando vel[] directamente.
        # Ahora se calcula aquí y se pasa a d.step() para que entre en step_rk4 -> _deriv.
        gust_now = np.zeros(3)
        if self._gust_steps > 0:
            frac     = self._gust_steps / GUST_STEPS
            gust_now = self._gust_vec * frac
            self._gust_steps -= 1

        orca_delta_map = {}
        obs_push_map   = {}

        for i, d in enumerate(self.drones):
            vp = v_prefs[i]

            # ?? Zona de convergencia final ???????????????????????
            near_goal = (d.goal is not None and
                         np.linalg.norm(d.state.pos - d.goal) < SUCCESS_M * 2.0)

            # ?? Delta ORCA inter-dron [m/s] ?????????????????????
            orca_delta = None
            if not near_goal and np.linalg.norm(vp) > 0.05:
                dv = v_orca[i] - vp
                if np.linalg.norm(dv) > 0.05:
                    orca_delta = dv

            # ?? Empuje de obstáculos: mover TARGET del PID ????????
            obs_push = np.zeros(3)
            if not near_goal:
                D_WARN   = 2.5
                D_EMERG  = 0.6
                for o in self.obstacles:
                    ds = o.dist_surface(d.state.pos)
                    if ds >= D_WARN:
                        continue
                    grad = d.state.pos - o.center
                    gn   = np.linalg.norm(grad)
                    if gn < 1e-6:
                        grad = np.array([0., 0., 1.]); gn = 1.0
                    dirn = grad / gn
                    if ds <= D_EMERG:
                        gain = 4.0
                    else:
                        t    = (ds - D_EMERG) / (D_WARN - D_EMERG)
                        gain = 4.0 * (1.0 - t) + 0.5 * t
                    obs_push += dirn * gain

            obs_push_out = obs_push if np.linalg.norm(obs_push) > 0.01 else None

            # ?? Separación de emergencia Z entre drones ?????????
            if not near_goal:
                z_assigned = 3.0 + d.id * 1.5
                for j in range(self.n_drones):
                    if j == i: continue
                    dxy = np.linalg.norm(pos[i][:2] - pos[j][:2])
                    dz  = abs(pos[i][2] - pos[j][2])
                    if dxy < 2.0 and dz < 1.0:
                        z_err = z_assigned - d.state.pos[2]
                        if obs_push_out is None:
                            obs_push_out = np.zeros(3)
                        obs_push_out[2] += np.clip(z_err * 0.4, -0.3, 0.3)

            orca_delta_map[d.id] = orca_delta
            obs_push_map[d.id]   = obs_push_out

        for d in self.drones:
            if not d.arrived:
                d.step(DT,
                       orca_delta = orca_delta_map.get(d.id),
                       obs_push   = obs_push_map.get(d.id),
                       gust       = gust_now)

        # ?? Intruso externo ?????????????????????????????????????
        if self.intruder is not None and self.intruder.active:
            self.intruder.step(DT, self.drones)

        t_ms = (time.perf_counter() - t0) * 1e3
        self.kpi_tracker.update([d.state.pos for d in self.drones],
                                t_ms, self._total_nodes)
        self.t          += DT
        self.step_count += 1

    def all_arrived(self):
        return all(d.arrived for d in self.drones)

    def finalize_kpis(self):
        fps  = [d.state.pos for d in self.drones]
        mdg  = [d.min_dist_goal if d.min_dist_goal < 1e8 else
                np.linalg.norm(d.state.pos - (d.goal if d.goal is not None else d.wps[0]))
                for d in self.drones]
        self.mission_kpi = self.kpi_tracker.compute(fps, self.t, mdg)
        # Añadir wind params
        self.mission_kpi.wind_alpha = self.wind_alpha
        self.mission_kpi.wind_sigma = self.wind_sigma


# ==========================================================
#  GEOMETRÍA 3D
# ==========================================================
def draw_drone_3d(ax, state: DroneState, color, scale=0.22):
    phi, theta, psi = state.angles
    px, py, pz = state.pos
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    R = np.array([
        [cy*ct,  cy*st*sp-sy*cp,  cy*st*cp+sy*sp],
        [sy*ct,  sy*st*sp+cy*cp,  sy*st*cp-cy*sp],
        [-st,    ct*sp,            ct*cp          ]
    ])
    arm_dirs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]], float) * scale
    rcols    = ['#E74C3C','#3498DB','#2ECC71','#F39C12']
    arts     = []
    ctr      = np.array([px, py, pz])
    for i, ad in enumerate(arm_dirs):
        tip = ctr + R @ ad
        ln, = ax.plot([px,tip[0]], [py,tip[1]], [pz,tip[2]],
                      color=rcols[i], lw=2.5, solid_capstyle='round')
        dt, = ax.plot([tip[0]], [tip[1]], [tip[2]],
                      'o', color=rcols[i], ms=5, zorder=9)
        arts += [ln, dt]
    body, = ax.plot([px], [py], [pz], 's', color=color, ms=9,
                    markeredgecolor='white', markeredgewidth=0.7, zorder=10)
    arts.append(body)
    AL = scale * 0.85
    for axd, c in zip([np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])],
                      ['#FF4444', '#44FF44', '#4488FF']):
        end = ctr + R @ (axd * AL)
        ln, = ax.plot([px,end[0]], [py,end[1]], [pz,end[2]],
                      '-', color=c, lw=1.2, alpha=0.65)
        arts.append(ln)
    return arts


# ==========================================================
#  HELPER - Figura matplotlib -> tk.PhotoImage
#  FIX BUG 1: FigureCanvasTkAgg queda en blanco cuando el tab
#  Notebook nunca fue seleccionado (winfo_width=1).
#  Solución: renderizar con backend Agg puro -> PNG en memoria ->
#  tk.PhotoImage.  No depende del tamaño del widget padre.
# ==========================================================
def _fig_to_photoimage(fig, dpi=100):
    """
    Convierte figura matplotlib -> tk.PhotoImage.
    CRITICO: tk.PhotoImage(data=...) REQUIERE base64, no bytes PNG crudos.
    Pasar bytes crudos crea imagen vacia silenciosamente (pestana en blanco).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)
    photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
    buf.close()
    return photo


# ==========================================================
#  APLICACIÓN PRINCIPAL  (ventana única con Notebook)
# ==========================================================
class SafeSkyApp:
    def __init__(self, root):
        self.root  = root
        self.root.title("SafeSky - Simulador de Enjambre de Drones | MR3001B")
        self.root.configure(bg=CB)
        self.root.minsize(1350, 820)
        self.sim          = None
        self.ani          = None
        self._running     = False
        self._paused      = False
        self._wind_q_arts = []
        self._mission_done = False
        self._build_ui()

    # ?????????????????????????????????????????????
    #  CONSTRUCCIÓN UI COMPLETA
    # ?????????????????????????????????????????????
    def _build_ui(self):
        # Panel izquierdo - scrollable
        # Estructura: outer_frame -> canvas + scrollbar -> inner_frame (P)
        outer = tk.Frame(self.root, bg=CP, width=278)
        outer.pack(side=tk.LEFT, fill=tk.Y, padx=(6,0), pady=6)
        outer.pack_propagate(False)

        # Scrollbar vertical
        _sb = tk.Scrollbar(outer, orient=tk.VERTICAL, bg=CP,
                           troughcolor='#0A0E1A', width=8)
        _sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Canvas que contiene el frame de controles
        _cv = tk.Canvas(outer, bg=CP, highlightthickness=0,
                        yscrollcommand=_sb.set, width=265)
        _cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _sb.config(command=_cv.yview)

        # Frame interior donde van todos los widgets
        self.panel_left = tk.Frame(_cv, bg=CP)
        _win = _cv.create_window((0, 0), window=self.panel_left, anchor='nw')

        # Actualizar scrollregion cuando cambia el tamaño del frame interior
        def _on_frame_configure(e):
            _cv.configure(scrollregion=_cv.bbox('all'))
        self.panel_left.bind('<Configure>', _on_frame_configure)

        # Ajustar ancho del frame interior al canvas
        def _on_canvas_configure(e):
            _cv.itemconfig(_win, width=e.width)
        _cv.bind('<Configure>', _on_canvas_configure)

        # Scroll con rueda del mouse sobre el panel
        def _on_mousewheel(e):
            _cv.yview_scroll(int(-1*(e.delta/120)), 'units')
        def _on_mousewheel_linux(e):
            _cv.yview_scroll(-1 if e.num==4 else 1, 'units')
        outer.bind_all('<MouseWheel>', _on_mousewheel)
        outer.bind_all('<Button-4>',   _on_mousewheel_linux)
        outer.bind_all('<Button-5>',   _on_mousewheel_linux)

        # Panel derecho: Notebook con pestañas
        self.panel_right = tk.Frame(self.root, bg=CB)
        self.panel_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_controls()
        self._build_notebook()

    def _build_notebook(self):
        """Notebook: pestaña 3D de simulación + pestaña KPIs."""
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook',        background=CB, borderwidth=0)
        style.configure('TNotebook.Tab',    background='#1A2940', foreground=CW,
                        padding=[14, 6], font=FONT_H)
        style.map('TNotebook.Tab',
                  background=[('selected', CB2)],
                  foreground=[('selected', CC)])

        self.notebook = ttk.Notebook(self.panel_right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ?? Pestaña 1: Simulación 3D ??????????????????
        self.tab_sim = tk.Frame(self.notebook, bg=CB)
        self.notebook.add(self.tab_sim, text="  ?  Simulación 3D  ")

        # ?? Pestaña 2: KPIs ???????????????????????????
        self.tab_kpi = tk.Frame(self.notebook, bg=CP)
        self.notebook.add(self.tab_kpi, text="  KPIs Finales  ")

        # ?? Pestaña 3: Planos 2D ??????????????????????
        self.tab_planes = tk.Frame(self.notebook, bg=CB)
        self.notebook.add(self.tab_planes, text="  ?  Planos 2D  ")

        # ?? Pestaña 4: Rendimiento ????????????????????
        self.tab_perf = tk.Frame(self.notebook, bg=CB)
        self.notebook.add(self.tab_perf, text="  Rendimiento  ")

        self._build_sim_canvas()
        self._build_kpi_tab()
        self._build_planes_tab()
        self._build_perf_tab()

    def _build_drone_legend(self):
        """Crea las etiquetas de color de cada dron en el panel izquierdo."""
        for w in self._legend_frame.winfo_children():
            w.destroy()
        n = int(self._var_drones.get()) if hasattr(self, '_var_drones') else 3
        for i in range(n):
            col = DRONE_COLORS[i % len(DRONE_COLORS)]
            fr = tk.Frame(self._legend_frame, bg=CP)
            fr.pack(fill=tk.X, pady=1)
            # Bloque de color
            canvas_dot = tk.Canvas(fr, width=18, height=18, bg=CP,
                                   highlightthickness=0)
            canvas_dot.pack(side=tk.LEFT, padx=(0,4))
            canvas_dot.create_oval(2, 2, 16, 16, fill=col, outline='white',
                                   width=0.8)
            tk.Label(fr, text=f"Dron {i+1}", bg=CP, fg=col,
                     font=('Consolas', 9, 'bold')).pack(side=tk.LEFT)

    def _build_sim_canvas(self):
        """Canvas matplotlib en la pestaña de simulación."""
        self.fig = plt.figure(figsize=(13.0, 8.0), facecolor=CB)
        self.fig.suptitle(
            "SafeSky - Enjambre 6-DOF  (Euler-Lagrange + DJI F450 + AR(1) Wind)",
            color=CW, fontsize=11, fontweight='bold', y=0.99)
        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               left=0.04, right=0.98,
                               top=0.95, bottom=0.06,
                               hspace=0.38, wspace=0.32)
        self.ax3d   = self.fig.add_subplot(gs[:, 0:2], projection='3d')
        self.ax_ang = self.fig.add_subplot(gs[0, 2])
        self.ax_kpi = self.fig.add_subplot(gs[1, 2])
        self._style_3d(); self._style_2d()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_sim)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def _build_kpi_tab(self):
        """Pestaña de KPIs - tabla detallada (se rellena al terminar la misión)."""
        self.kpi_tab_frame = tk.Frame(self.tab_kpi, bg=CP)
        self.kpi_tab_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(self.kpi_tab_frame,
                 text="REPORTE FINAL DE MISION",
                 bg=CP, fg=CC, font=('Segoe UI',14,'bold')).pack(pady=(10,2))
        tk.Label(self.kpi_tab_frame,
                 text="SafeSky MR3001B . Tecnológico de Monterrey - Los resultados\n"
                      "aparecerán aquí al completar la misión.",
                 bg=CP, fg=CDG, font=FONT_B).pack(pady=(0,10))

        self._kpi_content = tk.Frame(self.kpi_tab_frame, bg=CP)
        self._kpi_content.pack(fill=tk.BOTH, expand=True)

        # Placeholder
        self._kpi_placeholder = tk.Label(
            self._kpi_content,
            text="\n\n?  Ejecuta una simulación para ver los KPIs.",
            bg=CP, fg=CDG, font=('Segoe UI', 12))
        self._kpi_placeholder.pack(expand=True)

    def _populate_kpi_tab(self, kpi: MissionKPI, mission_mode: str):
        """Llena la pestaña KPI con la tabla real."""
        # Limpiar contenido anterior
        for w in self._kpi_content.winfo_children():
            w.destroy()

        pad = dict(padx=0, pady=3)
        n_ok  = sum(d.success for d in kpi.drones)
        c_mis = CG if kpi.success_rate==1.0 else (CY if kpi.success_rate>0 else CR)

        # ?? Banner de colisiones ???????????????????????????????
        # Usar constantes calculadas: D_COLLISION_PHYSICAL = 2xR_DRONE = 0.690m
        #                             DRONE_COLLISION_D = 0.690 + buffer = 0.840m
        coll_drone    = [d for d in kpi.drones if d.colision_drone]
        coll_obs_phys = [d for d in kpi.drones if d.colision_obs]
        obs_viol      = [d for d in kpi.drones if d.d_obs_min < D_SAFE and not d.colision_obs]
        any_alert = coll_drone or coll_obs_phys or obs_viol
        if any_alert:
            fr_alert = tk.Frame(self._kpi_content, bg='#2A0000')
            fr_alert.pack(fill=tk.X, pady=(0, 8))
            tk.Label(fr_alert, text="  ALERTAS DE SEGURIDAD DETECTADAS",
                     bg='#2A0000', fg='#FF4444',
                     font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=10, pady=(6,2))
            if coll_obs_phys:
                msg = "  COLISION CON OBSTACULO: " + ", ".join(
                    f"D{d.did+1} (d_obs={d.d_obs_min:.3f}m, {d.n_obs_violations} pasos violados)"
                    for d in coll_obs_phys)
                tk.Label(fr_alert, text=msg, bg='#2A0000', fg='#FF3333',
                         font=('Consolas', 9)).pack(anchor='w', padx=14, pady=2)
            if coll_drone:
                msg = "  ? COLISIÓN ENTRE DRONES: " + ", ".join(
                    f"D{d.did+1} (d_swarm={d.d_swarm_min:.3f}m < {D_COLLISION_PHYSICAL:.3f}m "
                    f"[brazo {R_DRONE_BODY:.3f}m x 2 = hélices se tocan])"
                    for d in coll_drone)
                tk.Label(fr_alert, text=msg, bg='#2A0000', fg='#FF6666',
                         font=('Consolas', 9)).pack(anchor='w', padx=14, pady=2)
            if obs_viol:
                msg = "  ZONA SEGURA VIOLADA: " + ", ".join(
                    f"D{d.did+1} (d_obs={d.d_obs_min:.3f}m < {D_SAFE:.2f}m, {d.n_obs_violations} pasos)"
                    for d in obs_viol)
                tk.Label(fr_alert, text=msg, bg='#2A0000', fg='#FF8888',
                         font=('Consolas', 9)).pack(anchor='w', padx=14, pady=2)
            tk.Label(fr_alert, text="  Ver pestaña Planos 2D para ubicación exacta.",
                     bg='#2A0000', fg='#FF9999',
                     font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=14, pady=(2,6))
        else:
            fr_ok = tk.Frame(self._kpi_content, bg='#002A00')
            fr_ok.pack(fill=tk.X, pady=(0, 8))
            tk.Label(fr_ok,
                     text="  \u2713  SIN COLISIONES - Todos los drones mantuvieron separacion segura",
                     bg='#002A00', fg=CG,
                     font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=10, pady=6)

        # ?? Resumen global ????????????????????????????????
        fr_g = tk.Frame(self._kpi_content, bg=CB2, relief='flat')
        fr_g.pack(fill=tk.X, pady=(0,10))

        tk.Label(fr_g, text="  RESUMEN GLOBAL", bg=CB2, fg=CC,
                 font=FONT_H).grid(row=0, column=0, columnspan=6,
                                   sticky='w', padx=10, pady=(8,4))

        mode_lbl = ("A->B Paralelo"   if mission_mode == 'ab' else
                    "A->B Línea Recta" if mission_mode == 'ab_straight' else
                    "A->B Cruzado"   if mission_mode == 'ab_cross' else
                    "Trayectoria Circular")
        items_g = [
            ("Modo misión",      mode_lbl,                       CC),
            ("Éxito misión",     f"{kpi.success_rate*100:.0f}%", c_mis),
            ("Drones OK",        f"{n_ok}/{len(kpi.drones)}",    c_mis),
            ("t simulación",     f"{kpi.t_sim:.1f} s",           CW),
            ("t cómputo medio",  f"{kpi.t_comp_mean:.2f} ms",    CW),
            ("Nodos A*",         f"{kpi.n_nodes}",               CW),
            ("Viento ?",         f"{kpi.wind_alpha:.2f}",        CC),
            ("Viento ?",         f"{kpi.wind_sigma:.2f}",        CC),
        ]
        for idx, (lbl, val, col) in enumerate(items_g):
            c = (idx % 3) * 2
            r = 1 + idx // 3
            tk.Label(fr_g, text=lbl+":", bg=CB2, fg=CDG,
                     font=FONT_B).grid(row=r, column=c, sticky='e',
                                       padx=(14,3), pady=3)
            tk.Label(fr_g, text=val, bg=CB2, fg=col,
                     font=('Consolas',10,'bold')).grid(
                         row=r, column=c+1, sticky='w', padx=(3,16), pady=3)
        tk.Label(fr_g, text="", bg=CB2).grid(row=r+1, column=0, pady=6)

        # ?? Tabla por dron ????????????????????????????????
        tk.Label(self._kpi_content, text="  DETALLE POR DRON",
                 bg=CP, fg=CC, font=FONT_H).pack(anchor='w', pady=(6,4))

        fr_t = tk.Frame(self._kpi_content, bg=CB2)
        fr_t.pack(fill=tk.X)

        hdrs  = ["Dron", "Estado", "e_B (m)", "d_obs (m)",
                 "d_swarm (m)", "L total (m)", "Violac.Obs"]
        widths= [8, 12, 11, 11, 13, 13, 11]
        for c, (h, w) in enumerate(zip(hdrs, widths)):
            bg = '#1A2940' if c%2==0 else '#182438'
            tk.Label(fr_t, text=h, bg=bg, fg=CC,
                     font=('Segoe UI',10,'bold'),
                     width=w, anchor='center'
                     ).grid(row=0, column=c, sticky='nsew', padx=1, pady=1)

        for r, d in enumerate(kpi.drones):
            rbg = '#0F1A2E' if r%2==0 else '#0D1725'
            # Estado: distinguir colision con obstáculo, con dron, o exitoso
            if d.colision_obs:
                ok_txt = "COLISION OBS"
                ok_col = '#FF2222'
            elif d.success:
                ok_txt = "OK LLEGO"
                ok_col = CG
            else:
                ok_txt = "NO LLEGO"
                ok_col = CR
            def chk(v, thr, lower):
                return CG if (v<=thr if lower else v>=thr) else CR
            viol_col = CR if d.n_obs_violations > 0 else CG
            viol_txt = f"{d.n_obs_violations} pasos" if d.n_obs_violations > 0 else "0  OK"
            vals = [
                (f"D{d.did+1}",               CW,    widths[0]),
                (ok_txt,                       ok_col,widths[1]),
                (f"{d.e_b:.4f}",              chk(d.e_b,E_B_OK,True),    widths[2]),
                (f"{d.d_obs_min:.3f}",        chk(d.d_obs_min,D_SAFE,False),widths[3]),
                (f"{d.d_swarm_min:.3f}",      chk(d.d_swarm_min,D_MIN,False),widths[4]),
                (f"{d.L_total:.2f}",          CW,    widths[5]),
                (viol_txt,                    viol_col, widths[6]),
            ]
            for c, (txt, col, w) in enumerate(vals):
                tk.Label(fr_t, text=txt, bg=rbg, fg=col,
                         font=FONT_BIG, width=w,
                         anchor='center').grid(
                             row=r+1, column=c, sticky='nsew', padx=1, pady=2)

        # ?? Umbrales ??????????????????????????????????????
        tk.Label(self._kpi_content, text="", bg=CP).pack()
        fr_leg = tk.Frame(self._kpi_content, bg=CP)
        fr_leg.pack(anchor='w', pady=4)
        tk.Label(fr_leg, text="Umbrales: ", bg=CP, fg=CDG,
                 font=FONT_B).pack(side=tk.LEFT)
        for txt, col in [
            (f"e_B < {E_B_OK} m", CG),
            (f"   d_obs >= {D_SAFE} m", CG),
            (f"   d_swarm >= {D_MIN} m", CG),
            ("    ? OK", CG), ("   ? FUERA", CR)]:
            tk.Label(fr_leg, text=txt, bg=CP, fg=col,
                     font=FONT_S).pack(side=tk.LEFT)

        # Cambiar a la pestaña KPI automáticamente
        self.notebook.select(self.tab_kpi)

        # Cambiar a la pestaña KPI automáticamente
        self.notebook.select(self.tab_kpi)

    # ?????????????????????????????????????????????
    #  PESTAÑA PLANOS 2D - placeholder
    # ?????????????????????????????????????????????
    def _build_planes_tab(self):
        """Pestaña de proyecciones 2D - se rellena tras la misión."""
        self._planes_content = tk.Frame(self.tab_planes, bg=CB)
        self._planes_content.pack(fill=tk.BOTH, expand=True)
        tk.Label(self._planes_content,
                 text="\n\n?  Las proyecciones 2D aparecerán al completar la misión.",
                 bg=CB, fg=CDG, font=('Segoe UI', 12)).pack(expand=True)
        self._planes_fig    = None
        self._planes_canvas = None

    def _populate_planes_tab(self, sim):
        """
        Genera 6 subplots en la pestaña de Planos 2D:
          Row 0: XY planta  |  XZ lateral  |  YZ frontal
          Row 1: distancia mínima a obstáculos  |  distancia inter-dron  |  leyenda
        """
        self.root.update_idletasks()
        self.root.update_idletasks()
        for w in self._planes_content.winfo_children():
            w.destroy()
        if self._planes_fig:
                plt.close(self._planes_fig)

        fig = plt.figure(figsize=(13.5, 8.0), facecolor=CB)
        fig.suptitle("SafeSky - Proyecciones 2D de Trayectorias y Separaciones",
                     color=CW, fontsize=11, fontweight='bold', y=0.99)
        gs  = gridspec.GridSpec(2, 3, fig,
                                left=0.07, right=0.97,
                                top=0.94, bottom=0.08,
                                hspace=0.42, wspace=0.36)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[0, 2])
        ax_do = fig.add_subplot(gs[1, 0])   # dist a obstáculos vs tiempo
        ax_dd = fig.add_subplot(gs[1, 1])   # dist inter-dron vs tiempo
        ax_lg = fig.add_subplot(gs[1, 2])   # leyenda / tabla resumen

        # Estilos
        for ax in [ax_xy, ax_xz, ax_yz, ax_do, ax_dd, ax_lg]:
            ax.set_facecolor('#0D1421')
            ax.tick_params(colors='gray', labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#1E2E45')

        # Títulos de proyecciones
        ax_xy.set_title("Planta  XY", color=CW, fontsize=9)
        ax_xy.set_xlabel("X [m]", color='#AAB7B8', fontsize=7)
        ax_xy.set_ylabel("Y [m]", color='#AAB7B8', fontsize=7)
        ax_xz.set_title("Lateral  XZ", color=CW, fontsize=9)
        ax_xz.set_xlabel("X [m]", color='#AAB7B8', fontsize=7)
        ax_xz.set_ylabel("Z [m]", color='#AAB7B8', fontsize=7)
        ax_yz.set_title("Frontal  YZ", color=CW, fontsize=9)
        ax_yz.set_xlabel("Y [m]", color='#AAB7B8', fontsize=7)
        ax_yz.set_ylabel("Z [m]", color='#AAB7B8', fontsize=7)

        import matplotlib.patches as mpatches

        # Dibujar obstáculos proyectados en cada plano
        for o in sim.obstacles:
            cx, cy, cz = o.center
            rx, ry, rz = o.radii

            for ax, (h, v, rh, rv, lbl) in [
                (ax_xy, (cx, cy, rx, ry, "XY")),
                (ax_xz, (cx, cz, rx, rz, "XZ")),
                (ax_yz, (cy, cz, ry, rz, "YZ")),
            ]:
                # Elipse del obstáculo
                el = mpatches.Ellipse((h, v), 2*rh, 2*rv,
                                      color=CR, alpha=0.45, zorder=3)
                ax.add_patch(el)
                # Zona de seguridad D_SAFE
                el_s = mpatches.Ellipse((h, v),
                                        2*(rh+D_SAFE), 2*(rv+D_SAFE),
                                        fill=False, edgecolor='#FF8888',
                                        linestyle='--', linewidth=0.9,
                                        alpha=0.60, zorder=3)
                ax.add_patch(el_s)

        # Trayectorias de cada dron
        for d in sim.drones:
            traj = np.array(d.traj)
            col  = d.color
            lw   = 1.8

            # XY
            ax_xy.plot(traj[:,0], traj[:,1], color=col, lw=lw, alpha=0.85, zorder=5)
            ax_xy.plot(traj[0,0], traj[0,1], 'D', color=col, ms=7, zorder=8)
            if d.goal is not None:
                ax_xy.plot(d.goal[0], d.goal[1], '*', color=col, ms=12, zorder=8)
            # XZ
            ax_xz.plot(traj[:,0], traj[:,2], color=col, lw=lw, alpha=0.85, zorder=5)
            ax_xz.plot(traj[0,0], traj[0,2], 'D', color=col, ms=7, zorder=8)
            if d.goal is not None:
                ax_xz.plot(d.goal[0], d.goal[2], '*', color=col, ms=12, zorder=8)
            # YZ
            ax_yz.plot(traj[:,1], traj[:,2], color=col, lw=lw, alpha=0.85, zorder=5)
            ax_yz.plot(traj[0,1], traj[0,2], 'D', color=col, ms=7, zorder=8)
            if d.goal is not None:
                ax_yz.plot(d.goal[1], d.goal[2], '*', color=col, ms=12, zorder=8)

            # Punto más cercano a obstáculos (?)
            if sim.obstacles:
                for pt in traj:
                    dists = [o.dist_surface(pt) for o in sim.obstacles]
                if min(dists) < D_SAFE * 2:
                    idx_close = np.argmin([
                        min(o.dist_surface(p) for o in sim.obstacles)
                        for p in traj])
                    pc = traj[idx_close]
                    ax_xy.plot(pc[0], pc[1], 'x', color='#FF4444', ms=9,
                               mew=2, zorder=10)
                    ax_xz.plot(pc[0], pc[2], 'x', color='#FF4444', ms=9,
                               mew=2, zorder=10)
                    ax_yz.plot(pc[1], pc[2], 'x', color='#FF4444', ms=9,
                               mew=2, zorder=10)

        # Líneas de referencia D_MIN entre drones (zona de exclusión)
        # - solo en XY para no saturar los otros planos
        n = len(sim.drones)
        for i in range(n):
            for j in range(i+1, n):
                t1 = np.array(sim.drones[i].traj)
                t2 = np.array(sim.drones[j].traj)
                npts = min(len(t1), len(t2))
                dists_ij = np.linalg.norm(t1[:npts] - t2[:npts], axis=1)
                idx_min  = np.argmin(dists_ij)
                p1 = t1[idx_min]; p2 = t2[idx_min]
                # Marcar punto de mínima separación
                mid = (p1 + p2) / 2
                ax_xy.annotate('',
                    xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
                    arrowprops=dict(arrowstyle='<->', color='#FFD700',
                                   lw=1.2, alpha=0.7))
                ax_xy.text(mid[0], mid[1],
                           f" {dists_ij[idx_min]:.2f}m",
                           color='#FFD700', fontsize=6.5, zorder=12)

        # Límites de ejes
        m = 1.0
        for ax in [ax_xy]:
            ax.set_xlim(-m, SPACE+m); ax.set_ylim(-m, SPACE+m)
            ax.set_aspect('equal', adjustable='box')
        for ax in [ax_xz]:
            ax.set_xlim(-m, SPACE+m); ax.set_ylim(-0.5, 10.5)
        for ax in [ax_yz]:
            ax.set_xlim(-m, SPACE+m); ax.set_ylim(-0.5, 10.5)

        # Líneas de suelo y techo
        for ax in [ax_xz, ax_yz]:
            ax.axhline(0, color='#3A4A5A', lw=0.8, ls=':')
            ax.axhline(10, color='#3A4A5A', lw=0.8, ls=':')

        # Grids suaves
        for ax in [ax_xy, ax_xz, ax_yz]:
            ax.grid(True, color='#1E2E45', lw=0.4, alpha=0.7)

        # ?? Distancia a obstáculos vs tiempo ??????????????
        # t_axis por dron: cada dron tiene su propia longitud de trayectoria
        # (los drones que llegan primero tienen traj más corta).
        # Usar t_axis[0] global solo para el eje X máximo del subplot.
        t_max = max(len(d.traj) for d in sim.drones) * DT
        for d in sim.drones:
            traj = np.array(d.traj)
            t_drone = np.arange(len(traj)) * DT   # eje t propio de este dron
            if sim.obstacles:
                obs_d = np.array([min(o.dist_surface(p) for o in sim.obstacles)
                                  for p in traj])
                ax_do.plot(t_drone, obs_d,
                           color=d.color, lw=1.4, alpha=0.85,
                           label=f"D{d.id+1}")
        ax_do.axhline(D_SAFE, color=CR, lw=1.2, ls='--', alpha=0.8)
        ax_do.text(0.02, D_SAFE + 0.05,
                   f"  D_SAFE = {D_SAFE} m", color=CR,
                   fontsize=7, transform=ax_do.get_yaxis_transform())
        ax_do.set_title("Distancia mínima a obstáculos", color=CW, fontsize=9)
        ax_do.set_xlabel("Tiempo [s]", color='#AAB7B8', fontsize=7)
        ax_do.set_ylabel("Distancia [m]", color='#AAB7B8', fontsize=7)
        ax_do.legend(fontsize=7, loc='upper right',
                     framealpha=0.3, facecolor='#0A1020', labelcolor=CW)
        ax_do.grid(True, color='#1E2E45', lw=0.4, alpha=0.7)
        ax_do.set_ylim(bottom=0)

        # ?? Distancia inter-dron vs tiempo ????????????????
        n = len(sim.drones)
        # Umbrales calculados desde geometría real (importados de physics/kpis):
        #   D_COLLISION_PHYSICAL = 2 x R_DRONE = 2 x (l_arm + r_prop) = 0.690 m
        #   DRONE_COLLISION_D    = D_COLLISION_PHYSICAL + SAFETY_BUFFER = 0.840 m
        collision_events = []      # lista de (t, di, dj, dist)
        for i in range(n):
            for j in range(i+1, n):
                t1 = np.array(sim.drones[i].traj)
                t2 = np.array(sim.drones[j].traj)
                npts = min(len(t1), len(t2))
                dd   = np.linalg.norm(t1[:npts] - t2[:npts], axis=1)
                col_mix = sim.drones[i].color
                ax_dd.plot(np.arange(npts)*DT, dd, color=col_mix, lw=1.4,
                           alpha=0.85, label=f"D{i+1}?D{j+1}")
                # Zona naranja: violación D_MIN (separación deseada)
                viol_mask = dd < D_MIN
                # Zona roja: colisión física real (hélices se tocan)
                coll_mask = dd < D_COLLISION_PHYSICAL
                if np.any(viol_mask):
                    ax_dd.fill_between(np.arange(npts)*DT, 0, dd,
                                       where=viol_mask,
                                       color='#FF8C00', alpha=0.25)
                if np.any(coll_mask):
                    ax_dd.fill_between(np.arange(npts)*DT, 0, dd,
                                       where=coll_mask,
                                       color=CR, alpha=0.45)
                    idx_c = np.where(coll_mask)[0]
                    t_c   = np.arange(npts)[idx_c[len(idx_c)//2]] * DT
                    d_c   = dd[idx_c].min()
                    collision_events.append((t_c, i+1, j+1, d_c))
                    ax_dd.annotate(
                        f"COLISION\nD{i+1}-D{j+1}\n{d_c:.2f}m",
                        xy=(t_c, d_c),
                        xytext=(t_c + 0.5, d_c + 0.8),
                        color=CR, fontsize=7, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=CR, lw=1.2),
                        bbox=dict(boxstyle='round,pad=0.2', fc='#200000', alpha=0.8))

        # ?? Líneas de referencia con desglose explícito ???
        # 1) Colisión física: 2 x (brazo + hélice)
        ax_dd.axhline(D_COLLISION_PHYSICAL, color=CR, lw=1.2, ls=':', alpha=0.9)
        ax_dd.text(0.01, D_COLLISION_PHYSICAL + 0.04,
                   f"  COLISIÓN FÍSICA = 2x(l_arm+r_prop) = 2x({R_DRONE_BODY:.3f}) = {D_COLLISION_PHYSICAL:.3f}m",
                   color=CR, fontsize=6,
                   transform=ax_dd.get_yaxis_transform())
        # 2) Colisión + buffer de seguridad
        ax_dd.axhline(DRONE_COLLISION_D, color='#FF8C00', lw=1.0, ls='-.', alpha=0.85)
        ax_dd.text(0.01, DRONE_COLLISION_D + 0.04,
                   f"  VIOLACIÓN = colisión + buffer({D_SAFETY_BUFFER:.3f}m) = {DRONE_COLLISION_D:.3f}m",
                   color='#FF8C00', fontsize=6,
                   transform=ax_dd.get_yaxis_transform())
        # 3) D_MIN deseado (criterio de éxito)
        ax_dd.axhline(D_MIN, color='#FFD700', lw=1.2, ls='--', alpha=0.8)
        ax_dd.text(0.01, D_MIN + 0.04,
                   f"  D_MIN deseado = {D_MIN}m",
                   color='#FFD700', fontsize=6.5,
                   transform=ax_dd.get_yaxis_transform())
        ax_dd.set_title("Separación entre drones", color=CW, fontsize=9)
        ax_dd.set_xlabel("Tiempo [s]", color='#AAB7B8', fontsize=7)
        ax_dd.set_ylabel("Distancia [m]", color='#AAB7B8', fontsize=7)
        ax_dd.legend(fontsize=7, loc='upper right',
                     framealpha=0.3, facecolor='#0A1020', labelcolor=CW)
        ax_dd.grid(True, color='#1E2E45', lw=0.4, alpha=0.7)
        ax_dd.set_ylim(bottom=0)

        # Marcar también en XY las posiciones de colisión
        for t_c, di, dj, d_c in collision_events:
            idx_c = int(t_c / DT)
            t1 = np.array(sim.drones[di-1].traj)
            t2 = np.array(sim.drones[dj-1].traj)
            if idx_c < len(t1) and idx_c < len(t2):
                mid_xy = (t1[idx_c] + t2[idx_c]) / 2
                # Círculo de colisión en XY
                import matplotlib.patches as mpatches2
                circ = mpatches2.Circle((mid_xy[0], mid_xy[1]),
                                        DRONE_RADIUS_PHYS, fill=True,
                                        facecolor=CR, alpha=0.5,
                                        edgecolor='white', lw=1.5, zorder=12)
                ax_xy.add_patch(circ)
                ax_xy.text(mid_xy[0], mid_xy[1]+0.6,
                           f"{d_c:.2f}m",
                           color=CR, fontsize=7.5, fontweight='bold',
                           ha='center', zorder=13)
                # En XZ
                mid_xz = (np.array([t1[idx_c][0], t1[idx_c][2]]) +
                          np.array([t2[idx_c][0], t2[idx_c][2]])) / 2
                ax_xz.plot(mid_xz[0], mid_xz[1], 'x',
                           color=CR, ms=12, mew=2.5, zorder=12)
                ax_xz.text(mid_xz[0], mid_xz[1]+0.3,
                           "!", color=CR, fontsize=9, ha='center', zorder=13)

        # ?? Leyenda / Tabla resumen ????????????????????????
        ax_lg.axis('off')
        ax_lg.set_title("Leyenda", color=CW, fontsize=9)
        items_lg = [
            ("*  Inicio de trayectoria",             '#AAAAAA'),
            ("*  Meta (A->B)",                         '#AAAAAA'),
            ("?  Obstáculo (proyección)",              CR),
            ("--  Zona seguridad (D_SAFE)",            '#FF8888'),
            ("?  Sep. mínima entre drones",            '#FFD700'),
            ("?  Punto más cercano a obstáculo",       '#FF4444'),
            ("COLISION FISICA (< 2x radio)",        CR),
            ("?  Zona D_MIN violada",                  '#FF8C00'),
            ("",                                       CW),
        ]
        for i, d in enumerate(sim.drones):
            items_lg.append((f"?  Dron {d.id+1}", d.color))
        # Resumen de colisiones
        if collision_events:
            items_lg.append(("", CW))
            items_lg.append(("EVENTOS DE COLISION:", CR))
            for t_c, di, dj, d_c in collision_events:
                items_lg.append((f"  D{di}-D{dj}: {d_c:.3f}m @ t={t_c:.1f}s", CR))
        else:
            items_lg.append(("", CW))
            items_lg.append(("Sin colisiones fisicas", CG))
        y = 0.98; dy = 0.075
        for txt, col in items_lg:
            ax_lg.text(0.04, y, txt, transform=ax_lg.transAxes,
                       color=col, fontsize=8, va='top')
            y -= dy

        # Agg backend: renderiza figura sin necesitar widget Tk visible
        photo = _fig_to_photoimage(fig, dpi=100)
        self._planes_photo = photo   # retener referencia - Tk no hace copia
        plt.close(fig)
        self._planes_fig = None      # ya cerrada, reset para evitar doble-close

        fr = tk.Frame(self._planes_content, bg=CB)
        fr.pack(fill=tk.BOTH, expand=True)
        sy = tk.Scrollbar(fr, orient=tk.VERTICAL)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        sx = tk.Scrollbar(fr, orient=tk.HORIZONTAL)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        cv = tk.Canvas(fr, bg=CB, yscrollcommand=sy.set, xscrollcommand=sx.set,
                       highlightthickness=0)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sy.config(command=cv.yview); sx.config(command=cv.xview)
        cv.create_image(0, 0, anchor=tk.NW, image=photo)
        cv.config(scrollregion=cv.bbox(tk.ALL))
        self._planes_canvas = cv
    # ?????????????????????????????????????????????
    #  PESTAÑA RENDIMIENTO - placeholder
    # ?????????????????????????????????????????????
    def _build_perf_tab(self):
        """Pestaña de rendimiento - se rellena tras la misión."""
        self._perf_content = tk.Frame(self.tab_perf, bg=CB)
        self._perf_content.pack(fill=tk.BOTH, expand=True)
        tk.Label(self._perf_content,
                 text="\n\nLas graficas de rendimiento apareceran al completar la mision.",
                 bg=CB, fg=CDG, font=('Segoe UI', 12)).pack(expand=True)
        self._perf_fig    = None
        self._perf_canvas = None

    def _populate_perf_tab(self, sim):
        """
        Genera 9 subplots de rendimiento por dron:
          Row 0: Velocidad escalar  |  Error vel. tracking  |  Empuje u1
          Row 1: Dist. al goal      |  Viento AR(1) mag.    |  Dist. a obstáculos
          Row 2: Barras KPI         |  Eficiencia tray.     |  Tabla resumen
        Métricas de desempeño: error de velocidad, error de distancias.
        """
        self.root.update_idletasks()
        for w in self._perf_content.winfo_children():
            w.destroy()
        if self._perf_fig:
            plt.close(self._perf_fig)

        fig = plt.figure(figsize=(13.5, 9.5), facecolor=CB)
        fig.suptitle("SafeSky - Rendimiento y Métricas por Dron  (v9)",
                     color=CW, fontsize=11, fontweight='bold', y=0.99)
        gs = gridspec.GridSpec(3, 3, fig,
                               left=0.07, right=0.97,
                               top=0.94, bottom=0.07,
                               hspace=0.52, wspace=0.36)
        ax_spd  = fig.add_subplot(gs[0, 0])   # velocidad escalar
        ax_verr = fig.add_subplot(gs[0, 1])   # error de tracking de velocidad
        ax_thr  = fig.add_subplot(gs[0, 2])   # empuje u1
        ax_dg   = fig.add_subplot(gs[1, 0])   # distancia al goal (error de distancia)
        ax_wnd  = fig.add_subplot(gs[1, 1])   # magnitud del viento
        ax_dobs = fig.add_subplot(gs[1, 2])   # distancia a obs (todos drones)
        ax_bar  = fig.add_subplot(gs[2, 0])   # barras KPI
        ax_eff  = fig.add_subplot(gs[2, 1])   # eficiencia trayectoria
        ax_tbl  = fig.add_subplot(gs[2, 2])   # tabla resumen

        # Estilo base
        def _sty(ax, title, xl, yl):
            ax.set_facecolor('#0D1421')
            ax.tick_params(colors='gray', labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#1E2E45')
            ax.set_title(title, color=CW, fontsize=8.5)
            ax.set_xlabel(xl,   color='#AAB7B8', fontsize=7)
            ax.set_ylabel(yl,   color='#AAB7B8', fontsize=7)
            ax.grid(True, color='#1E2E45', lw=0.4, alpha=0.7)

        _sty(ax_spd,  "Velocidad escalar  |v|",         "t [s]", "v [m/s]")
        _sty(ax_verr, "Error vel. tracking  |v-v_cmd|", "t [s]", "err [m/s]")
        _sty(ax_thr,  "Empuje total  u?",                "t [s]", "u? [N]")
        _sty(ax_dg,   "Error distancia al goal  d(t)",  "t [s]", "d [m]")
        _sty(ax_wnd,  "Perturbación viento  |w(t)|",    "t [s]", "|w| [m/s]")
        _sty(ax_dobs, "Dist. mínima a obstáculos",       "t [s]", "d [m]")

        # Referencia de hover
        p_hover = sim.params.thrust_hover()
        ax_thr.axhline(p_hover, color='#88AAFF', lw=0.9, ls=':')
        ax_thr.text(0.01, p_hover+0.15, f"hover={p_hover:.1f}N",
                    color='#88AAFF', fontsize=6.5,
                    transform=ax_thr.get_yaxis_transform())

        for d in sim.drones:
            traj   = np.array(d.traj)
            npts   = len(d.hist_speed)
            t_ax   = np.arange(npts) * DT
            col    = d.color
            lbl    = f"D{d.id+1}"

            # Velocidad escalar
            ax_spd.plot(t_ax, d.hist_speed, color=col, lw=1.4, alpha=0.9, label=lbl)

            # Error de tracking de velocidad |v - v_cmd|
            if d.hist_vel_err:
                t_ve = np.arange(len(d.hist_vel_err)) * DT
                ax_verr.plot(t_ve, d.hist_vel_err, color=col, lw=1.2, alpha=0.85, label=lbl)
                # Sombrear área bajo la curva (integral del error)
                ax_verr.fill_between(t_ve, d.hist_vel_err, alpha=0.12, color=col)

            # Empuje u1
            if d.hist_thrust:
                t_thr = np.arange(len(d.hist_thrust)) * DT
                ax_thr.plot(t_thr, d.hist_thrust, color=col, lw=1.3,
                            alpha=0.85, label=lbl)

            # Distancia al goal (error de distancia) - solo modo A->B
            if d.hist_dist_goal:
                t_dg = np.arange(len(d.hist_dist_goal)) * DT
                ax_dg.plot(t_dg, d.hist_dist_goal, color=col, lw=1.3,
                           alpha=0.85, label=lbl)

            # Viento magnitud
            if d.hist_wind_mag:
                t_wnd = np.arange(len(d.hist_wind_mag)) * DT
                ax_wnd.plot(t_wnd, d.hist_wind_mag, color=col, lw=1.2, alpha=0.80,
                            label=lbl)

            # Distancia a obstáculos - usar t propio de traj
            if sim.obstacles:
                obs_d = np.array([min(o.dist_surface(p) for o in sim.obstacles)
                                  for p in traj])
                t_obs = np.arange(len(obs_d)) * DT
                ax_dobs.plot(t_obs, obs_d,
                             color=col, lw=1.3, alpha=0.85, label=lbl)

        # Referencias
        ax_spd.axhline(V_MAX_NAV, color=CY, lw=0.9, ls='--', alpha=0.6)
        ax_spd.text(0.01, V_MAX_NAV+0.05, f"v_max={V_MAX_NAV}m/s",
                    color=CY, fontsize=6.5, transform=ax_spd.get_yaxis_transform())
        ax_verr.axhline(0.3, color=CY, lw=0.7, ls=':', alpha=0.5)
        ax_verr.text(0.01, 0.32, "tol=0.3", color=CY, fontsize=6,
                     transform=ax_verr.get_yaxis_transform())
        ax_dg.axhline(SUCCESS_M, color=CG, lw=0.8, ls='--', alpha=0.7)
        ax_dg.text(0.01, SUCCESS_M+0.1, f"r_ok={SUCCESS_M}m",
                   color=CG, fontsize=6.5, transform=ax_dg.get_yaxis_transform())

        # Línea D_SAFE en distancia a obstáculos
        ax_dobs.axhline(D_SAFE, color=CR, lw=1.0, ls='--', alpha=0.8)
        ax_dobs.text(0.01, D_SAFE+0.05, f"D_SAFE={D_SAFE}m",
                     color=CR, fontsize=6.5,
                     transform=ax_dobs.get_yaxis_transform())

        # Leyendas comunes
        for ax in [ax_spd, ax_verr, ax_thr, ax_dg, ax_wnd, ax_dobs]:
            ax.legend(fontsize=7, loc='upper right',
                      framealpha=0.3, facecolor='#0A1020', labelcolor=CW)

        # ?? Barras KPI por dron ???????????????????????????
        ax_bar.set_facecolor('#0D1421')
        ax_bar.tick_params(colors='gray', labelsize=7)
        for sp in ax_bar.spines.values(): sp.set_edgecolor('#1E2E45')
        ax_bar.set_title("KPIs por dron", color=CW, fontsize=8.5)

        n   = len(sim.drones)
        xp  = np.arange(n)
        w3  = 0.25
        kpi = sim.mission_kpi

        eb_vals  = [d.e_b       for d in kpi.drones]
        dob_vals = [d.d_obs_min for d in kpi.drones]
        dsw_vals = [d.d_swarm_min for d in kpi.drones]

        bars1 = ax_bar.bar(xp - w3, eb_vals,  w3, label='e_B',      color='#00BFFF', alpha=0.85)
        bars2 = ax_bar.bar(xp,      dob_vals, w3, label='d_obs_min', color='#2ECC71', alpha=0.85)
        bars3 = ax_bar.bar(xp + w3, dsw_vals, w3, label='d_sw_min',  color='#F39C12', alpha=0.85)

        # Líneas de umbral
        ax_bar.axhline(E_B_OK, color='#00BFFF', lw=0.9, ls='--', alpha=0.7)
        ax_bar.axhline(D_SAFE, color='#2ECC71', lw=0.9, ls='--', alpha=0.7)
        ax_bar.axhline(D_MIN,  color='#F39C12', lw=0.9, ls='--', alpha=0.7)
        ax_bar.set_xticks(xp)
        ax_bar.set_xticklabels([f"D{d.id+1}" for d in sim.drones],
                               color=CW, fontsize=8)
        ax_bar.set_ylabel("Distancia [m]", color='#AAB7B8', fontsize=7)
        ax_bar.legend(fontsize=7, loc='upper right',
                      framealpha=0.3, facecolor='#0A1020', labelcolor=CW)
        ax_bar.grid(True, axis='y', color='#1E2E45', lw=0.4, alpha=0.7)

        # ?? Eficiencia de trayectoria ??????????????????????
        ax_eff.set_facecolor('#0D1421')
        ax_eff.tick_params(colors='gray', labelsize=7)
        for sp in ax_eff.spines.values(): sp.set_edgecolor('#1E2E45')
        ax_eff.set_title("Eficiencia de trayectoria", color=CW, fontsize=8.5)
        ax_eff.set_xlabel("Dron", color='#AAB7B8', fontsize=7)

        eff_cols   = []
        L_reals    = []
        L_directs  = []
        v_avgs     = []
        v_maxs     = []

        for d in sim.drones:
            traj   = np.array(d.traj)
            L_dir  = np.linalg.norm(traj[-1] - traj[0])
            eff    = (L_dir / max(d.L, 0.01)) * 100
            eff_cols.append(CG if eff > 80 else (CY if eff > 60 else CR))
            L_reals.append(d.L)
            L_directs.append(L_dir)
            v_avgs.append(float(np.mean(d.hist_speed)) if d.hist_speed else 0.)
            v_maxs.append(float(np.max(d.hist_speed))  if d.hist_speed else 0.)

        x_dr = np.arange(n)
        ax_eff.bar(x_dr - 0.2, L_reals,   0.4,
                   label='L total',    color='#3498DB', alpha=0.85)
        ax_eff.bar(x_dr + 0.2, L_directs, 0.4,
                   label='L directa',  color='#A8FF78', alpha=0.85)
        ax_eff.set_xticks(x_dr)
        ax_eff.set_xticklabels([f"D{d.id+1}" for d in sim.drones],
                               color=CW, fontsize=8)
        ax_eff.set_ylabel("Longitud [m]", color='#AAB7B8', fontsize=7)
        ax_eff.legend(fontsize=7, loc='upper right',
                      framealpha=0.3, facecolor='#0A1020', labelcolor=CW)
        ax_eff.grid(True, axis='y', color='#1E2E45', lw=0.4, alpha=0.7)

        # ?? Tabla de rendimiento ???????????????????????????
        ax_tbl.axis('off')
        ax_tbl.set_facecolor('#0D1421')
        ax_tbl.set_title("Tabla de Rendimiento", color=CW, fontsize=8.5)

        hdr_row = f"{'Dr':<3} {'v_avg':>6} {'ve_avg':>7} {'L':>7} {'eff':>6} {'OK'}"
        sep_row = "?" * 45
        rows    = [hdr_row, sep_row]
        for i, d in enumerate(sim.drones):
            eff  = (L_directs[i] / max(L_reals[i], 0.01)) * 100
            ve   = float(np.mean(d.hist_vel_err)) if d.hist_vel_err else 0.
            ok   = "OK" if kpi.drones[i].success else "NO"
            rows.append(
                f"D{d.id+1}  {v_avgs[i]:>5.2f}  {ve:>6.3f}"
                f"  {L_reals[i]:>6.1f}  {eff:>5.1f}%  {ok}")
        rows += [sep_row,
                 "ve_avg = err vel tracking",
                 f"?_wind = {sim.wind_alpha:.2f}",
                 f"?_wind = {sim.wind_sigma:.2f}",
                 f"dt = {DT} s",
                 f"N_pasos = {sim.step_count}"]
        if sim.intruder is not None:
            rows.append(f"Intruso: {sim.intruder.speed:.1f}m/s "
                        f"sep_min={sim.intruder.min_sep:.2f}m")

        ax_tbl.text(0.03, 0.97, '\n'.join(rows),
                    transform=ax_tbl.transAxes,
                    color=CW, fontsize=8.2, va='top',
                    fontfamily='monospace', linespacing=1.60,
                    bbox=dict(boxstyle='round,pad=0.4',
                              fc='#0A1020', alpha=0.7))

        # Agg backend: renderiza figura sin necesitar widget Tk visible
        photo = _fig_to_photoimage(fig, dpi=105)
        self._perf_photo = photo     # retener referencia - Tk no hace copia
        plt.close(fig)
        self._perf_fig = None        # ya cerrada, reset para evitar doble-close

        fr = tk.Frame(self._perf_content, bg=CB)
        fr.pack(fill=tk.BOTH, expand=True)
        sy = tk.Scrollbar(fr, orient=tk.VERTICAL)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        sx = tk.Scrollbar(fr, orient=tk.HORIZONTAL)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        cv = tk.Canvas(fr, bg=CB, yscrollcommand=sy.set, xscrollcommand=sx.set,
                       highlightthickness=0)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sy.config(command=cv.yview); sx.config(command=cv.xview)
        cv.create_image(0, 0, anchor=tk.NW, image=photo)
        cv.config(scrollregion=cv.bbox(tk.ALL))
        self._perf_canvas = cv
    # ?????????????????????????????????????????????
    #  CONTROLES IZQUIERDO
    # ?????????????????????????????????????????????
    def _build_controls(self):
        P   = self.panel_left
        pad = dict(padx=10, pady=3)

        tk.Label(P, text="? SafeSky", bg=CP, fg=CC,
                 font=('Segoe UI',14,'bold')).pack(padx=10, pady=(10,2))
        tk.Label(P, text="Enjambre Dinámico de Drones",
                 bg=CP, fg='#5A7A9A', font=FONT_B).pack(pady=0)
        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Modo de misión ????????????????????????????
        tk.Label(P, text="MODO DE MISION", bg=CP, fg=CC, font=FONT_H).pack(**pad)
        self._var_mission = tk.StringVar(value='ab')
        fr_m = tk.Frame(P, bg=CP)
        fr_m.pack(fill=tk.X, padx=10)
        for txt, val in [(">  A -> B (paralelo)",  'ab'),
                        ("?  A -> B (línea recta)", 'ab_straight'),
                        ("?  A -> B (cruzado)",  'ab_cross'),
                        ("?  Circular",          'circle')]:
            tk.Radiobutton(fr_m, text=txt, variable=self._var_mission, value=val,
                           bg=CP, fg=CW, selectcolor='#1A2940',
                           activebackground=CP, activeforeground=CC,
                           font=FONT_B).pack(anchor='w', pady=1)

        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Configuración ????????????????????????????
        tk.Label(P, text="?  CONFIGURACIÓN", bg=CP, fg=CC, font=FONT_H).pack(**pad)

        def int_slider(label, var, lo, hi):
            tk.Label(P, text=label, bg=CP, fg=CW, font=FONT_B).pack(anchor='w', **pad)
            sl = ttk.Scale(P, from_=lo, to=hi, orient='horizontal',
                           variable=var, length=220)
            sl.pack(padx=10)
            sl.configure(command=lambda v, v2=var: v2.set(int(round(float(v)))))
            tk.Label(P, textvariable=var, bg=CP, fg=CY,
                     font=('Consolas',11,'bold')).pack(anchor='e', padx=14)

        self._var_drones = tk.IntVar(value=3)
        self._var_obs    = tk.IntVar(value=4)
        int_slider("Drones (3?6):",     self._var_drones, 3, 6)
        int_slider("Obstáculos (0?8):", self._var_obs,    0, 8)
        # Actualizar leyenda cuando cambia el número de drones
        self._var_drones.trace_add('write', lambda *a: (
            self._build_drone_legend() if hasattr(self, '_legend_frame') else None))

        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Obstáculos: modo ??????????????????????????
        tk.Label(P, text="?  MODO OBSTÁCULOS", bg=CP, fg=CC, font=FONT_H).pack(**pad)
        self._var_obs_mode = tk.StringVar(value='fixed')
        fr_ob = tk.Frame(P, bg=CP)
        fr_ob.pack(fill=tk.X, padx=10)
        for txt, val, tip in [
            ("Fijo (misma posición)",  'fixed',   ''),
            ("Aleatorio (cada run)",   'random',  ''),
            ("Dinámico (se mueven)",   'dynamic', ''),
        ]:
            tk.Radiobutton(fr_ob, text=txt, variable=self._var_obs_mode, value=val,
                           bg=CP, fg=CW, selectcolor='#1A2940',
                           activebackground=CP, activeforeground=CC,
                           font=FONT_B).pack(anchor='w', pady=1)

        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Viento AR(1) ??????????????????????????????
        tk.Label(P, text="VIENTO AR(1)", bg=CP, fg=CC, font=FONT_H).pack(**pad)
        tk.Label(P, text="?=0.60->brisa (?1.1 m/s)  ?=1.0->viento (?1.9 m/s)  ?=1.8->ráfaga",
                 bg=CP, fg='#5A7A9A', font=('Consolas',7)).pack(pady=0)
        self._var_alpha = tk.DoubleVar(value=0.85)
        self._var_sigma = tk.DoubleVar(value=0.60)
        for lbl, var, lo, hi in [
            ("? - persistencia:", self._var_alpha, 0.0, 0.98),
            ("? - intensidad:",   self._var_sigma, 0.0, 2.00)]:
            tk.Label(P, text=lbl, bg=CP, fg=CW, font=FONT_B).pack(anchor='w', **pad)
            ttk.Scale(P, from_=lo, to=hi, orient='horizontal',
                      variable=var, length=220).pack(padx=10)
            sv = tk.StringVar(value=f"{var.get():.2f}")
            var.trace_add('write', lambda *a, sv=sv, v=var: sv.set(f"{v.get():.2f}"))
            tk.Label(P, textvariable=sv, bg=CP, fg=CY,
                     font=FONT_M).pack(anchor='e', padx=14)

        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Velocidad ?????????????????????????????????
        tk.Label(P, text="VELOCIDAD", bg=CP, fg=CC, font=FONT_H).pack(**pad)
        self._var_speed = tk.IntVar(value=2)
        def int_slider_s(label, var, lo, hi):
            tk.Label(P, text=label, bg=CP, fg=CW, font=FONT_B).pack(anchor='w', **pad)
            sl = ttk.Scale(P, from_=lo, to=hi, orient='horizontal',
                           variable=var, length=220)
            sl.pack(padx=10)
            sl.configure(command=lambda v, v2=var: v2.set(int(round(float(v)))))
            tk.Label(P, textvariable=var, bg=CP, fg=CY,
                     font=('Consolas',11,'bold')).pack(anchor='e', padx=14)
        int_slider_s("Pasos/frame:", self._var_speed, 1, 8)

        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Orden de drones ???????????????????????????
        tk.Label(P, text="?  ORDEN DE INICIO", bg=CP, fg=CC, font=FONT_H).pack(**pad)
        tk.Label(P, text="Dron que inicia más a la izquierda (ID más bajo = posición X menor):",
                 bg=CP, fg='#5A7A9A', font=('Segoe UI',7), wraplength=240,
                 justify='left').pack(anchor='w', padx=10)
        self._var_order = tk.StringVar(value='ltr')
        fr_ord = tk.Frame(P, bg=CP)
        fr_ord.pack(fill=tk.X, padx=10)
        for txt, val in [("?>  Izq->Der  (D1 en X=2)", 'ltr'),
                         ("?>  Der->Izq  (D1 en X=18)", 'rtl')]:
            tk.Radiobutton(fr_ord, text=txt, variable=self._var_order, value=val,
                           bg=CP, fg=CW, selectcolor='#1A2940',
                           activebackground=CP, activeforeground=CC,
                           font=FONT_B).pack(anchor='w', pady=1)
        tk.Label(P, text="Metas (modo A->B):",
                 bg=CP, fg='#5A7A9A', font=('Segoe UI',7)).pack(anchor='w', padx=10, pady=(4,0))
        self._var_goal_side = tk.StringVar(value='opposite')
        fr_gs = tk.Frame(P, bg=CP)
        fr_gs.pack(fill=tk.X, padx=10)
        for txt, val in [("?  Lado opuesto (cruzado)", 'opposite'),
                         ("?  Mismo lado  (paralelo)", 'same')]:
            tk.Radiobutton(fr_gs, text=txt, variable=self._var_goal_side, value=val,
                           bg=CP, fg=CW, selectcolor='#1A2940',
                           activebackground=CP, activeforeground=CC,
                           font=FONT_B).pack(anchor='w', pady=1)

        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Botones ???????????????????????????????????
        tk.Label(P, text=">  CONTROLES", bg=CP, fg=CC, font=FONT_H).pack(**pad)
        bst = dict(bg='#1A2940', fg=CW, font=FONT_H, relief='flat',
                   cursor='hand2', activebackground='#253A55',
                   bd=0, width=24, pady=6)
        self._btn_start = tk.Button(P, text=">  INICIAR",      **bst, command=self._start)
        self._btn_pause = tk.Button(P, text="PAUSAR",       **bst, command=self._pause)
        self._btn_reset = tk.Button(P, text="REINICIAR",    **bst, command=self._reset)
        self._btn_gust  = tk.Button(P, text="RAFAGA SUBITA",**bst, command=self._trigger_gust)
        self._btn_intr  = tk.Button(P, text="LANZAR INTRUSO",**bst, command=self._launch_intruder)
        for b in [self._btn_start, self._btn_pause, self._btn_reset,
                  self._btn_gust, self._btn_intr]:
            b.pack(padx=10, pady=2)
        self._btn_pause['state'] = 'disabled'
        self._btn_gust['state']  = 'disabled'
        self._btn_intr['state']  = 'disabled'

        # Velocidad del intruso
        tk.Label(P, text="Vel. intruso (m/s):", bg=CP, fg='#5A7A9A',
                 font=('Segoe UI', 7)).pack(anchor='w', padx=10, pady=(2,0))
        self._var_intr_spd = tk.DoubleVar(value=4.0)
        ttk.Scale(P, from_=1.0, to=10.0, orient='horizontal',
                  variable=self._var_intr_spd, length=220).pack(padx=10)
        sv_is = tk.StringVar(value="4.0")
        self._var_intr_spd.trace_add('write', lambda *a: sv_is.set(f"{self._var_intr_spd.get():.1f}"))
        tk.Label(P, textvariable=sv_is, bg=CP, fg=CY, font=FONT_M).pack(anchor='e', padx=14)

        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Estado ????????????????????????????????????
        tk.Label(P, text="ESTADO", bg=CP, fg=CC, font=FONT_H).pack(**pad)
        self._status_var = tk.StringVar(value="Esperando inicio...")
        tk.Label(P, textvariable=self._status_var, bg=CP, fg=CG,
                 font=FONT_M, wraplength=250, justify='left').pack(**pad)
        self._live_var = tk.StringVar(value="")
        tk.Label(P, textvariable=self._live_var, bg=CP, fg=CY,
                 font=('Consolas',8), wraplength=250, justify='left').pack(**pad)

        ttk.Separator(P, orient='horizontal').pack(fill=tk.X, padx=8, pady=5)

        # ?? Leyenda de colores de drones ??????????????
        tk.Label(P, text="LEYENDA DRONES", bg=CP, fg=CC, font=FONT_H).pack(**pad)
        self._legend_frame = tk.Frame(P, bg=CP)
        self._legend_frame.pack(fill=tk.X, padx=10, pady=(0,6))
        self._build_drone_legend()

    # ?????????????????????????????????????????????
    #  ESTILOS MATPLOTLIB
    # ?????????????????????????????????????????????
    def _style_3d(self):
        ax = self.ax3d
        ax.set_facecolor('#0D1421')
        for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            p.fill = False; p.set_edgecolor('#1E2A3A')
        ax.tick_params(colors='gray', labelsize=6)
        ax.set_xlabel('X [m]', color='#AAB7B8', fontsize=7)
        ax.set_ylabel('Y [m]', color='#AAB7B8', fontsize=7)
        ax.set_zlabel('Z [m]', color='#AAB7B8', fontsize=7)
        ax.set_title('Vista 3D - Enjambre + Obstáculos + Viento',
                     color=CW, fontsize=9, pad=4)
        ax.set_xlim(0, SPACE); ax.set_ylim(0, SPACE); ax.set_zlim(0, 10)
        ax.view_init(elev=22, azim=-55)
        xx, yy = np.meshgrid([0, SPACE], [0, SPACE])
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color='#2A3A50')

    def _style_2d(self):
        for ax in [self.ax_ang, self.ax_kpi]:
            ax.set_facecolor('#0F1525')
            ax.tick_params(colors='gray', labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#1E2E45')
        self.ax_ang.set_title('Ángulos de Euler - Drone 1',
                              color=CW, fontsize=8, pad=3)
        self.ax_ang.set_ylabel('rad', color='#AAB7B8', fontsize=7)
        self.ax_ang.set_xlabel('t [s]', color='#AAB7B8', fontsize=7)
        self.ax_ang.axhline(0, color='white', ls=':', lw=0.5, alpha=0.3)
        self.ax_kpi.set_title('KPIs en Tiempo Real', color=CW, fontsize=8, pad=3)
        self.ax_kpi.axis('off')

    # ?????????????????????????????????????????????
    #  INICIO / PAUSA / RESET
    # ?????????????????????????????????????????????
    def _start(self):
        if self._running and not self._paused: return
        if self._paused:
            self._paused = False
            self._btn_pause['text'] = "PAUSAR"
            return

        n_dr      = int(self._var_drones.get())
        n_obs     = int(self._var_obs.get())
        alpha     = round(float(self._var_alpha.get()), 2)
        sigma     = round(float(self._var_sigma.get()), 2)
        obs_mode  = self._var_obs_mode.get()
        miss_mode = self._var_mission.get()

        self._status_var.set(
            "Planificando rutas A->B paralelas..." if miss_mode == 'ab' else
            "Modo A->B línea recta (sin A*)..." if miss_mode == 'ab_straight' else
            "Planificando rutas A->B cruzadas..." if miss_mode == 'ab_cross' else
            "Generando trayectorias circulares...")
        self.root.update()

        drone_order = self._var_order.get()
        goal_side   = self._var_goal_side.get()
        self.sim = Simulation(n_dr, n_obs, alpha, sigma, obs_mode, miss_mode,
                              drone_order=drone_order, goal_side=goal_side)
        self._running      = True
        self._paused       = False
        self._mission_done = False
        self._btn_start['text']  = '>  CORRIENDO'
        self._btn_pause['state'] = 'normal'
        self._btn_gust['state']  = 'normal'
        self._btn_intr['state']  = 'normal'

        self._init_artists()
        mode_txt = ("A->B-Paralelo" if miss_mode=='ab' else
                    "A->B-Recto"   if miss_mode=='ab_straight' else
                    "A->B-Cruzado" if miss_mode=='ab_cross' else "Circular")
        self._status_var.set(
            f"[{mode_txt}] {n_dr} drones, {n_obs} obs, {obs_mode}\n"
            f"?={alpha:.2f}  ?={sigma:.2f}")
        # Volver a pestaña simulación
        self.notebook.select(self.tab_sim)
        self._anim_loop()

    def _pause(self):
        if not self._running: return
        self._paused = not self._paused
        self._btn_pause['text'] = ">  CONTINUAR" if self._paused else "PAUSAR"

    def _trigger_gust(self):
        """Inyecta una ráfaga de viento lateral súbita en la simulación activa."""
        if self.sim and self._running and not self._paused:
            self.sim.trigger_gust()
            self._status_var.set(" ¡Ráfaga súbita lanzada!")

    def _launch_intruder(self):
        """Lanza un objeto intrusor externo a la velocidad configurada."""
        if self.sim and self._running and not self._paused:
            spd = float(self._var_intr_spd.get())
            self.sim.launch_intruder(speed=spd)
            self._status_var.set(f"Intruso lanzado ({spd:.1f} m/s)")

    def _reset(self):
        self._running = False; self._paused = False
        if self.ani:
            try: self.ani.event_source.stop()
            except: pass
            self.ani = None
        self.sim = None
        self._mission_done = False
        self._btn_start['text']  = '>  INICIAR'
        self._btn_pause['text']  = '||  PAUSAR'
        self._btn_pause['state'] = 'disabled'
        self._btn_gust['state']  = 'disabled'
        self._btn_intr['state']  = 'disabled'
        self._status_var.set("Esperando inicio...")
        self._live_var.set("")
        self._wind_q_arts = []
        self.ax3d.cla(); self.ax_ang.cla(); self.ax_kpi.cla()
        self._style_3d(); self._style_2d()
        self.canvas.draw()
        # Limpiar pestañas de análisis
        for w in self._planes_content.winfo_children(): w.destroy()
        if self._planes_fig: plt.close(self._planes_fig); self._planes_fig = None
        tk.Label(self._planes_content,
                 text="\n\n?  Las proyecciones 2D aparecerán al completar la misión.",
                 bg=CB, fg=CDG, font=('Segoe UI', 12)).pack(expand=True)
        for w in self._perf_content.winfo_children(): w.destroy()
        if self._perf_fig: plt.close(self._perf_fig); self._perf_fig = None
        tk.Label(self._perf_content,
                 text="\n\nLas graficas de rendimiento apareceran al completar la mision.",
                 bg=CB, fg=CDG, font=('Segoe UI', 12)).pack(expand=True)
        self.notebook.select(self.tab_sim)

    # ?????????????????????????????????????????????
    #  ARTISTAS INICIALES
    # ?????????????????????????????????????????????
    def _init_artists(self):
        ax = self.ax3d; ax.cla(); self._style_3d()
        sim = self.sim

        # Obstáculos
        u = np.linspace(0, 2*np.pi, 16)
        v = np.linspace(0, np.pi, 12)
        self._obs_surfs = []
        for o in sim.obstacles:
            xs = o.center[0] + o.radii[0]*np.outer(np.cos(u), np.sin(v))
            ys = o.center[1] + o.radii[1]*np.outer(np.sin(u), np.sin(v))
            zs = o.center[2] + o.radii[2]*np.outer(np.ones_like(u), np.cos(v))
            sf = ax.plot_surface(xs, ys, zs, alpha=0.28, color=CR,
                                 linewidth=0, antialiased=False)
            # Wireframe encima para que los bordes sean visibles (volumen sólido)
            wf = ax.plot_wireframe(xs, ys, zs, color='#FF8888', linewidth=0.4,
                                   alpha=0.35, rstride=3, cstride=3)
            self._obs_surfs.append(sf)
            self._obs_surfs.append(wf)

        # Metas / waypoints circulares
        if sim.mission_mode == 'ab':
            for i, d in enumerate(sim.drones):
                ax.plot(*d.goal, '*', color=d.color, ms=14, zorder=12)
                ax.text(d.goal[0], d.goal[1], d.goal[2]+0.5,
                        f"G{i+1}", color=d.color, fontsize=7, zorder=11)
                if d.wps:
                    pts = np.array(d.wps)
                    ax.plot(pts[:,0], pts[:,1], pts[:,2],
                            '--', color=d.color, lw=0.7, alpha=0.22)
        else:
            # Trazar el círculo guía
            cx, cy = SPACE/2, SPACE/2
            ang = np.linspace(0, 2*np.pi, 80)
            for i, d in enumerate(sim.drones):
                r = 5.0; z = 4.0 + i*0.5
                ax.plot(cx + r*np.cos(ang), cy + r*np.sin(ang),
                        np.full_like(ang, z),
                        '--', color=d.color, lw=0.8, alpha=0.30)

        # Estelas
        self._trail_lines  = []
        self._drone_bodies = [[] for _ in sim.drones]
        for d in sim.drones:
            ln, = ax.plot([], [], [], color=d.color, lw=1.8, alpha=0.70, zorder=5)
            self._trail_lines.append(ln)

        # Plano suelo
        xx, yy = np.meshgrid([0, SPACE], [0, SPACE])
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color='#2A3A50')

        # Ángulos
        self.ax_ang.cla(); self._style_2d()
        self._ang_lines = []
        for col, lbl in zip([CR, CL, CG], ['? roll', '? pitch', '? yaw']):
            ln, = self.ax_ang.plot([], [], color=col, lw=1.5, label=lbl)
            self._ang_lines.append(ln)
        self.ax_ang.legend(fontsize=6, loc='upper right',
                           framealpha=0.3, facecolor='#1A1A2E', labelcolor=CW)
        self.ax_ang.set_xlim(0, 60); self.ax_ang.set_ylim(-0.55, 0.55)

        # Texto viento (siempre visible)
        self._wind_text = ax.text2D(
            0.02, 0.97, "Viento AR(1): calculando...",
            transform=ax.transAxes,
            color='#A8FF78', fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='#0A1A0A', alpha=0.6))
        self._wind_q_arts = []
        self.canvas.draw()

    # ?????????????????????????????????????????????
    #  LOOP DE ANIMACIÓN
    # ?????????????????????????????????????????????
    def _anim_loop(self):
        def animate(frame):
            if not self._running or self._paused or self.sim is None:
                return []
            sim   = self.sim
            steps = max(1, int(self._var_speed.get()))
            for _ in range(steps):
                if not sim.all_arrived():
                    sim.step()
            self._update_artists()
            if sim.all_arrived() and not self._mission_done:
                self._mission_done = True
                sim.finalize_kpis()
                self._show_kpi_panel(sim.mission_kpi)
                self._populate_kpi_tab(sim.mission_kpi, sim._mission_mode_orig)
                # Usar after() para renderizar pestañas análisis DESPUÉS de que
                # Tkinter termine de dibujar la pestaña KPI (evita blank canvas)
                self.root.after(150, lambda s=sim: self._populate_planes_tab(s))
                self.root.after(300, lambda s=sim: self._populate_perf_tab(s))
                n_ok = sum(d.success for d in sim.mission_kpi.drones)
                self._status_var.set(
                    f"Mision completada!\nt={sim.t:.1f}s  "
                    f"{n_ok}/{sim.n_drones} OK")
            return []

        self.ani = mplanim.FuncAnimation(
            self.fig, animate, interval=DT_ANIM,
            blit=False, cache_frame_data=False)
        self.canvas.draw()

    # ?????????????????????????????????????????????
    #  UPDATE ARTISTAS
    # ?????????????????????????????????????????????
    def _update_artists(self):
        sim = self.sim
        ax  = self.ax3d
        t_now = sim.t

        # Obstáculos dinámicos: redraw
        if sim.obs_mode == 'dynamic':
            for sf in self._obs_surfs:
                try: sf.remove()
                except: pass
            self._obs_surfs = []
            u = np.linspace(0, 2*np.pi, 14)
            v = np.linspace(0, np.pi, 10)
            for o in sim.obstacles:
                xs = o.center[0] + o.radii[0]*np.outer(np.cos(u), np.sin(v))
                ys = o.center[1] + o.radii[1]*np.outer(np.sin(u), np.sin(v))
                zs = o.center[2] + o.radii[2]*np.outer(np.ones_like(u), np.cos(v))
                sf = ax.plot_surface(xs, ys, zs, alpha=0.28, color=CO,
                                     linewidth=0, antialiased=False)
                wf = ax.plot_wireframe(xs, ys, zs, color='#FF9944', linewidth=0.4,
                                       alpha=0.35, rstride=3, cstride=3)
                self._obs_surfs.append(sf)
                self._obs_surfs.append(wf)

        # Estelas y geometría drones
        for i, d in enumerate(sim.drones):
            traj = np.array(d.traj[-TRAIL_LEN:])
            if len(traj) > 1:
                self._trail_lines[i].set_data(traj[:,0], traj[:,1])
                self._trail_lines[i].set_3d_properties(traj[:,2])
            for art in self._drone_bodies[i]:
                try: art.remove()
                except: pass
            self._drone_bodies[i] = draw_drone_3d(ax, d.state, d.color)

        # Ángulos dron 0
        d0 = sim.drones[0]
        if d0.hist_angles:
            arr  = np.array(d0.hist_angles)
            t_ax = np.linspace(0, t_now, len(arr))
            self.ax_ang.set_xlim(max(0, t_now-30), max(30, t_now+2))
            for j in range(3):
                self._ang_lines[j].set_data(t_ax, arr[:,j])

        # ?? Campo de viento 3D (cuadrícula en el espacio) ??
        # El viento AR(1) es un campo global uniforme - se visualiza
        # como cuadrícula 3x3x2 de flechas en el volumen de simulación,
        # NO pegadas a los drones. Escala: 2.5 m de flecha por (m/s).
        for arts in self._wind_q_arts:
            for a in arts:
                try: a.remove()
                except: pass
        self._wind_q_arts = []

        WS = 2.8  # m de flecha / (m/s) de viento
        d0 = sim.drones[0]
        if d0.hist_wind:
            w  = np.array(d0.hist_wind[-1])
            wm = np.linalg.norm(w)
            # Cuadrícula 3x3x2 = 18 puntos en el interior del volumen
            grid_pts = [(x, y, z)
                        for x in [4., 10., 16.]
                        for y in [4., 10., 16.]
                        for z in [4., 12.]]
            arts = []
            if wm > 0.05:
                xs = [p[0] for p in grid_pts]
                ys = [p[1] for p in grid_pts]
                zs = [p[2] for p in grid_pts]
                # Cuadrícula completa - flechas semitransparentes
                q = ax.quiver(xs, ys, zs,
                              [w[0]*WS]*18, [w[1]*WS]*18, [w[2]*WS]*18,
                              normalize=False,
                              color='#A8FF78', alpha=0.40,
                              linewidth=1.0, arrow_length_ratio=0.28)
                arts.append(q)
                # Flecha central grande - muy visible
                q2 = ax.quiver(10., 10., 8.,
                               w[0]*WS*2.0, w[1]*WS*2.0, w[2]*WS*2.0,
                               normalize=False,
                               color='#A8FF78', alpha=0.92,
                               linewidth=2.8, arrow_length_ratio=0.22)
                arts.append(q2)
            else:
                xs = [p[0] for p in grid_pts]
                ys = [p[1] for p in grid_pts]
                zs = [p[2] for p in grid_pts]
                pt, = ax.plot(xs, ys, zs, 's',
                              color='#3A5A3A', ms=3, alpha=0.4, zorder=8)
                arts.append(pt)
            self._wind_q_arts.append(arts)

        # Texto de viento (siempre actualizado)
        if d0.hist_wind:
            w = d0.hist_wind[-1]
            wm = np.linalg.norm(w)
            # Indicar si hay ráfaga activa
            gust_lbl = f"   RÁFAGA ({sim._gust_steps}p)" if sim._gust_steps > 0 else ""
            tipo = "laminar" if self._var_alpha.get() > 0.7 else "turbulento"
            self._wind_text.set_text(
                f"Viento AR(1) [{tipo}]{gust_lbl}\n"
                f"[{w[0]:+.2f}, {w[1]:+.2f}, {w[2]:+.2f}] m/s  "
                f"|w|={wm:.2f}")

        # ?? Renderizar intruso externo ?????????????????????????
        # Eliminar render previo del intruso
        if hasattr(self, '_intruder_arts'):
            for a in self._intruder_arts:
                try: a.remove()
                except: pass
        self._intruder_arts = []
        if sim.intruder is not None and sim.intruder.active:
            ix, iy, iz = sim.intruder.pos
            # Esfera naranja simple como punto grande
            pt, = ax.plot([ix], [iy], [iz], 'o',
                          color=INTRUDER_COLOR, ms=12, zorder=15,
                          markeredgecolor='white', markeredgewidth=0.8)
            self._intruder_arts.append(pt)
            # Trail del intruso
            traj_i = np.array(sim.intruder.traj[-60:])
            if len(traj_i) > 1:
                ln, = ax.plot(traj_i[:,0], traj_i[:,1], traj_i[:,2],
                              '-', color=INTRUDER_COLOR, lw=1.5, alpha=0.55)
                self._intruder_arts.append(ln)
            # Label
            txt = ax.text(ix, iy, iz+0.6, "INTRUSO",
                          color=INTRUDER_COLOR, fontsize=7, zorder=16)
            self._intruder_arts.append(txt)

        self._update_kpi_live()
        self.canvas.draw_idle()

    def _update_kpi_live(self):
        sim = self.sim
        # Texto izquierdo
        lines = []
        for d in sim.drones:
            if d.goal is not None:
                dist = np.linalg.norm(d.state.pos - d.goal)
            else:
                dist = np.linalg.norm(d.state.pos - d.current_wp())
            st   = "LLEGO" if d.arrived else f"{dist:.2f}m"
            if sim.mission_mode == 'circle':
                st += f" [{d.laps}v]"
            lines.append(f"D{d.id+1}: {st}  L={d.L:.1f}m")
        lines.append(f"t={sim.t:.1f}s  n={sim.step_count}")
        self._live_var.set('\n'.join(lines))

        # Panel KPI derecho en tiempo real
        ax = self.ax_kpi
        ax.cla(); ax.set_facecolor('#0F1525'); ax.axis('off')
        ax.set_title('KPIs en Tiempo Real', color=CW, fontsize=8, pad=3)
        rows = [f"{'Dr':<3}  {'dist':>6}  {'L':>7}   estado", "?"*34]
        for d in sim.drones:
            if d.goal is not None:
                dist = np.linalg.norm(d.state.pos - d.goal)
            else:
                dist = np.linalg.norm(d.state.pos - d.current_wp())
            st   = "OK" if d.arrived else f"{dist:.2f}m"
            if sim.mission_mode == 'circle':
                st += f"[{d.laps}v]"
            rows.append(f"D{d.id+1}   {dist:>6.2f}  {d.L:>7.1f}   {st}")
        rows += ["?"*34,
                 f"t  = {sim.t:.1f} s",
                 f"n  = {sim.step_count} pasos",
                 f"?={sim.wind_alpha:.2f}  ?={sim.wind_sigma:.2f}",
                 f"obs: {sim.obs_mode}  [{sim.mission_mode}]"]
        ax.text(0.04, 0.97, '\n'.join(rows), transform=ax.transAxes,
                color=CW, fontsize=7.5, va='top',
                fontfamily='monospace', linespacing=1.55)
        # Leyenda de colores en el panel 3D
        for i, d in enumerate(sim.drones):
            ax.text(0.04, 0.10 - i*0.055,
                    f"? D{d.id+1}", transform=ax.transAxes,
                    color=d.color, fontsize=8, va='top',
                    fontfamily='monospace', fontweight='bold')

    def _show_kpi_panel(self, kpi: MissionKPI):
        """Panel KPI final compacto en el subplot matplotlib."""
        ax = self.ax_kpi
        ax.cla(); ax.set_facecolor('#0F1525'); ax.axis('off')
        ax.set_title('KPIs - Resultado Final', color=CW, fontsize=8, pad=3)
        n_ok  = sum(d.success for d in kpi.drones)
        c_mis = CG if kpi.success_rate==1.0 else (CY if kpi.success_rate>0 else CR)
        rows  = [
            (f"ÉXITO: {kpi.success_rate*100:.0f}%  ({n_ok}/{len(kpi.drones)} OK)", c_mis),
            (f"t={kpi.t_sim:.1f}s  t_cmp={kpi.t_comp_mean:.1f}ms", CW),
            (f"Nodos A*: {kpi.n_nodes}", CW),
            ("", CW),
            (f"{'Dr':<4}{'e_B':>6}{'d_obs':>7}{'d_sw':>7}{'L':>8}", CC),
            ("?"*35, CDG),
        ]
        for d in kpi.drones:
            ok = "OK" if d.success else "NO"
            dc = CG if d.success else CR
            rows.append((
                f"D{d.did+1}{ok}  {d.e_b:.3f} {d.d_obs_min:.3f}  "
                f"{d.d_swarm_min:.3f} {d.L_total:.1f}", dc))
        y = 0.97; dy = 0.092
        for ln, col in rows:
            ax.text(0.04, y, ln, transform=ax.transAxes,
                    color=col, fontsize=7.5, va='top', fontfamily='monospace')
            y -= dy
        # Reportar intruso si fue lanzado
        if self.sim and self.sim.intruder is not None:
            intr = self.sim.intruder
            intr_col = CY if intr.min_sep > 1.0 else CR
            ax.text(0.04, y, f"Intruso {intr.speed:.1f}m/s  sep_min={intr.min_sep:.2f}m",
                    transform=ax.transAxes, color=intr_col,
                    fontsize=7.5, va='top', fontfamily='monospace')
        self.canvas.draw_idle()


# ==========================================================
#  MAIN
# ==========================================================
def main():
    root = tk.Tk()
    app  = SafeSkyApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (
        setattr(app, '_running', False),
        root.destroy()
    ))
    root.mainloop()

if __name__ == '__main__':
    main()
