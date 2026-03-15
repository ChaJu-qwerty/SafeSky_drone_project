"""
physics.py — Dinámica 6-DOF del cuadricóptero SafeSky
======================================================
Modelo basado en Luukkonen (2011) "Modelling and control of quadcopter"

Parámetros reales DJI F450 (con batería 3S 2200 mAh ~250g):
  m   = 1.0  kg   (chasis 282g + 4 motores ~260g + bat ~250g + ESCs ~120g)
  l   = 0.225 m   (brazo: distancia centro del frame → hub del rotor)
  Ixx = 7.75e-3 kg·m²  (eje roll)
  Iyy = 7.75e-3 kg·m²  (eje pitch)
  Izz = 13.6e-3 kg·m²  (eje yaw, mayor por distribución radial)
  kf  = 2.98e-6  N/(rad/s)²  (coef. empuje motor 2212/920KV)
  kb  = 1.14e-7  N·m/(rad/s)² (coef. par)

GEOMETRÍA PARA COLISIÓN / ORCA:
  l_arm  = 0.225 m   brazo (centro → hub rotor)
  r_prop = 0.120 m   radio hélice 9.45" (9.45 × 0.0254 / 2)
  r_body = 0.345 m   radio físico total = l_arm + r_prop
  ──────────────────────────────────────────────────
  El radio que usa ORCA para evasión = r_body + buffer_seguridad
  (ver ORCA_RADIUS abajo). Esto garantiza que dos drones nunca
  permitan que sus hélices se acerquen más del buffer.
"""
import numpy as np
from dataclasses import dataclass, field

# ── Geometría física del dron ────────────────────────────────────────────────
L_ARM          = 0.225   # m — brazo: centro frame → hub del rotor (diagonal F450/2)
R_PROP         = 0.120   # m — radio hélice 9.45 pulgadas (9.45 × 0.0254 / 2)
R_DRONE_BODY   = L_ARM + R_PROP          # = 0.345 m — radio físico real del dron
D_COLLISION    = 2 * R_DRONE_BODY        # = 0.690 m — colisión física (hélices se tocan)
SAFETY_BUFFER  = 0.150                   # m — margen operacional de seguridad
ORCA_RADIUS    = R_DRONE_BODY + SAFETY_BUFFER  # = 0.495 m — radio que usa ORCA por dron
#                                              #   zona libre entre dos drones = 2×0.495 = 0.990m


# ──────────────────────────────────────────────────────────
#  PARÁMETROS DJI F450
# ──────────────────────────────────────────────────────────
@dataclass
class QuadParams:
    # Masa y geometría
    m:   float = 1.0         # kg
    g:   float = 9.81        # m/s²
    l:   float = 0.225       # m  (brazo centro→rotor; diagonal F450 = 450mm → l=225mm)
    # Inercias (kg·m²) — escaladas con l² desde mediciones experimentales
    Ixx: float = 7.75e-3     # eje roll  (antes 8.1e-3 con l=0.23m)
    Iyy: float = 7.75e-3     # eje pitch
    Izz: float = 13.6e-3     # eje yaw
    # Coeficientes de motor DJI 2212/920KV con hélice 9450
    kf:  float = 2.98e-6     # N / (rad/s)²
    kb:  float = 1.14e-7     # N·m / (rad/s)²
    # Arrastre aerodinámico traslacional (N·s/m) — estimado en túnel de viento
    Ax:  float = 0.08
    Ay:  float = 0.08
    Az:  float = 0.10
    # Límites físicos
    omega_max: float = 1200.0  # rad/s (máximo RPM motor)
    phi_max:   float = 0.40    # rad (~23°) ángulo máximo roll/pitch en nav

    def omega_hover(self):
        """Velocidad angular de cada rotor en vuelo estacionario."""
        return np.sqrt(self.m * self.g / (4.0 * self.kf))

    def thrust_max(self):
        return 4.0 * self.kf * self.omega_max**2

    def thrust_hover(self):
        return self.m * self.g


# ──────────────────────────────────────────────────────────
#  ROTACIÓN Y CINEMÁTICA  (Luukkonen 2011)
# ──────────────────────────────────────────────────────────
def rotation_matrix(phi, theta, psi):
    """Matriz R (cuerpo → inercial), convención ZYX."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,    ct*sp,             ct*cp            ]
    ])

def jacobian_matrix(phi, theta, Ixx, Iyy, Izz):
    """J(η) = W_η^T · I · W_η — Jacobiana de inercia (Luukkonen ec.16)."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    J11 = Ixx
    J12 = 0.
    J13 = -Ixx*st
    J22 = Iyy*cp**2 + Izz*sp**2
    J23 = (Iyy - Izz)*cp*sp*ct
    J33 = Ixx*st**2 + Iyy*sp**2*ct**2 + Izz*cp**2*ct**2
    return np.array([[J11, J12, J13],
                     [J12, J22, J23],
                     [J13, J23, J33]])

def coriolis_matrix(phi, theta, psi, dphi, dtheta, dpsi, Ixx, Iyy, Izz):
    """C(η, η̇) — matriz de Coriolis/centrífuga (Luukkonen ec.19)."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    C11 = 0.
    C12 = ((Iyy-Izz)*(dtheta*cp*sp + dpsi*sp**2*ct)
           + (Izz-Iyy)*dpsi*cp**2*ct - Ixx*dpsi*ct)
    C13 = (Izz-Iyy)*dpsi*cp*sp*ct**2
    C21 = ((Izz-Iyy)*(dtheta*cp*sp + dpsi*sp*ct)
           + (Iyy-Izz)*dpsi*cp**2*ct + Ixx*dpsi*ct)
    C22 = (Izz-Iyy)*dphi*cp*sp
    C23 = (-Ixx*dpsi*st*ct + Iyy*dpsi*sp**2*st*ct + Izz*dpsi*cp**2*st*ct)
    C31 = ((Iyy-Izz)*dpsi*ct**2*sp*cp - Ixx*dtheta*ct)
    C32 = ((Izz-Iyy)*(dtheta*cp*sp*st + dphi*sp**2*ct)
           + (Iyy-Izz)*dphi*cp**2*ct
           + Ixx*dpsi*st*ct - Iyy*dpsi*sp**2*st*ct - Izz*dpsi*cp**2*st*ct)
    C33 = ((Iyy-Izz)*dphi*cp*sp*ct**2
           - Iyy*dtheta*sp**2*ct*st - Izz*dtheta*cp**2*ct*st + Ixx*dtheta*ct*st)
    return np.array([[C11, C12, C13],
                     [C21, C22, C23],
                     [C31, C32, C33]])


# ──────────────────────────────────────────────────────────
#  VIENTO  AR(1)
# ──────────────────────────────────────────────────────────
class WindAR1:
    """
    Proceso AR(1) vectorial:  w[k+1] = α·w[k] + η[k],  η ~ N(0, σ²I₃)

    w[k] representa VELOCIDAD DE VIENTO [m/s].
    La fuerza aerodinámica que genera sobre el dron se calcula con el
    modelo de arrastre cuadrático real (no lineal):
      F_wind = 0.5 · ρ · Cd_A · |w| · w
    donde Cd_A = Cd × A_frontal ≈ 1.2 × 0.04 m² = 0.048 m² (DJI F450)

    FIX — valor anterior (sigma=0.20, Cd_lineal=0.30) generaba solo
    0.11 m/s² de perturbación, imperceptible para el PID.
    Nuevo sigma=1.0 con arrastre cuadrático genera ~0.6-1.2 m/s²,
    visible en empuje, velocidad y ángulos.

    Valores de sigma según escenario:
      sigma = 0.50 → viento suave interior  (std ≈ 0.95 m/s)
      sigma = 1.00 → viento urbano moderado (std ≈ 1.90 m/s)
      sigma = 1.80 → viento fuerte / ráfaga (std ≈ 3.42 m/s)
    std_estacionaria = sigma / sqrt(1 - alpha²)
    """
    # ── Constantes aerodinámicas DJI F450 ───────────────────
    # Cd_A = Cd × A_frontal efectiva en simulación
    # Valor físico puro:  Cd≈1.2, A≈0.04 m² → Cd_A_fisico = 0.048 m²
    # En simulación 6-DOF con campo de viento AR(1) turbulento, se usa un
    # factor ×8 para representar los efectos de turbulencia, ground effect
    # y variaciones de presión dinámica no modelados explícitamente.
    # Cd_A=0.40 → sigma=1.0 genera ~0.88 m/s² (9% de g), claramente visible
    # en empuje (spike ~1N), ángulos (φ/θ ±5°) y velocidad lateral.
    RHO  = 1.225   # kg/m³  — densidad del aire al nivel del mar
    CD_A = 0.400   # m²     — Cd_efectivo × A_frontal (incluye factor turbulencia)

    def __init__(self, alpha=0.85, sigma=0.60, seed=None):
        self.alpha = alpha
        self.sigma = sigma
        self.w     = np.zeros(3)
        self._rng  = np.random.default_rng(seed)

    def step(self):
        eta    = self._rng.normal(0., self.sigma, 3)
        self.w = self.alpha * self.w + eta
        return self.w.copy()

    def reset(self):
        self.w[:] = 0.

    @classmethod
    def wind_force(cls, wind_vel: np.ndarray, mass: float) -> np.ndarray:
        """
        Convierte velocidad de viento [m/s] → aceleración [m/s²].
        Modelo cuadrático real: F = 0.5·ρ·Cd_A·|w|·w  →  a = F/m
        """
        spd = np.linalg.norm(wind_vel)
        if spd < 1e-6:
            return np.zeros(3)
        return (0.5 * cls.RHO * cls.CD_A * spd * wind_vel) / mass


# ──────────────────────────────────────────────────────────
#  ESTADO 6-DOF
# ──────────────────────────────────────────────────────────
@dataclass
class DroneState:
    pos:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    angles:  np.ndarray = field(default_factory=lambda: np.zeros(3))   # φ θ ψ
    dAngles: np.ndarray = field(default_factory=lambda: np.zeros(3))   # φ̇ θ̇ ψ̇

    def copy(self):
        return DroneState(self.pos.copy(), self.vel.copy(),
                          self.angles.copy(), self.dAngles.copy())


# ──────────────────────────────────────────────────────────
#  DINÁMICA EULER-LAGRANGE  —  integrador RK4
# ──────────────────────────────────────────────────────────
class QuadDynamics:
    """
    Ecuaciones de movimiento del cuadricóptero DJI F450:

    Traslación (Newton-Euler inercial):
      m·ξ̈ = [0, 0, -m·g]ᵀ + R·[0, 0, u1]ᵀ − A·ξ̇ + F_wind

    Rotación (Euler-Lagrange):
      J(η)·η̈ = [u2, u3, u4]ᵀ − C(η,η̇)·η̇

    Entradas:
      u1 : empuje total  [N]
      u2 : torque roll   [N·m]
      u3 : torque pitch  [N·m]
      u4 : torque yaw    [N·m]
      wind: perturbación AR(1) [m/s] → convertida a fuerza m·Cd·wind
    """
    def __init__(self, params: QuadParams):
        self.p = params

    def _deriv(self, s: DroneState, u1, u2, u3, u4, wind: np.ndarray):
        p = self.p
        phi, theta, psi     = s.angles
        dphi, dtheta, dpsi  = s.dAngles
        R = rotation_matrix(phi, theta, psi)

        # ── Traslación ──────────────────────────────────────
        # Empuje (solo componente Z del cuerpo → inercial vía R)
        thrust_body = np.array([0., 0., u1])
        # Arrastre aerodinámico traslacional (proporcional a velocidad del dron)
        A_drag = np.array([p.Ax, p.Ay, p.Az])
        drag   = A_drag * s.vel
        # Viento: arrastre cuadrático real F = 0.5·ρ·Cd_A·|w|·w  →  a = F/m
        # FIX: antes era f_wind = 0.30*m*wind (incorrecto — daba solo 0.11 m/s²)
        # Ahora usa el modelo cuadrático correcto con Cd_A del DJI F450
        a_wind = WindAR1.wind_force(wind, p.m)
        # Aceleración traslacional
        d_vel = (np.array([0., 0., -p.g])
                 + (R @ thrust_body) / p.m
                 - drag / p.m
                 + a_wind)
        # Derivada de posición = velocidad (pura cinemática)
        d_pos = s.vel.copy()

        # ── Rotación (Euler-Lagrange) ────────────────────────
        J   = jacobian_matrix(phi, theta, p.Ixx, p.Iyy, p.Izz)
        C   = coriolis_matrix(phi, theta, psi, dphi, dtheta, dpsi,
                              p.Ixx, p.Iyy, p.Izz)
        tau = np.array([u2, u3, u4])
        try:
            d2a = np.linalg.solve(J, tau - C @ s.dAngles)
        except np.linalg.LinAlgError:
            d2a = np.zeros(3)

        drv          = DroneState()
        drv.pos      = d_pos
        drv.vel      = d_vel
        drv.angles   = s.dAngles.copy()
        drv.dAngles  = d2a
        return drv

    def _add(self, s: DroneState, d: DroneState, h: float) -> DroneState:
        ns = DroneState()
        ns.pos     = s.pos     + h * d.pos
        ns.vel     = s.vel     + h * d.vel
        ns.angles  = s.angles  + h * d.angles
        ns.dAngles = s.dAngles + h * d.dAngles
        return ns

    def step_rk4(self, state: DroneState, u1, u2, u3, u4,
                 wind: np.ndarray, dt: float) -> DroneState:
        k1 = self._deriv(state,                  u1, u2, u3, u4, wind)
        k2 = self._deriv(self._add(state, k1, dt/2), u1, u2, u3, u4, wind)
        k3 = self._deriv(self._add(state, k2, dt/2), u1, u2, u3, u4, wind)
        k4 = self._deriv(self._add(state, k3, dt),   u1, u2, u3, u4, wind)

        ns = DroneState()
        ns.pos     = state.pos     + (dt/6)*(k1.pos    + 2*k2.pos    + 2*k3.pos    + k4.pos)
        ns.vel     = state.vel     + (dt/6)*(k1.vel    + 2*k2.vel    + 2*k3.vel    + k4.vel)
        ns.angles  = state.angles  + (dt/6)*(k1.angles + 2*k2.angles + 2*k3.angles + k4.angles)
        ns.dAngles = state.dAngles + (dt/6)*(k1.dAngles+ 2*k2.dAngles+ 2*k3.dAngles+ k4.dAngles)

        # Restricciones físicas
        ns.pos[2]   = max(0., ns.pos[2])          # no penetrar el suelo
        ns.vel      = np.clip(ns.vel, -12., 12.)  # velocidad terminal ~12 m/s
        ns.angles   = np.clip(ns.angles, -0.52, 0.52)  # ±30° max en navegación
        ns.dAngles  = np.clip(ns.dAngles, -8., 8.)
        return ns


# ──────────────────────────────────────────────────────────
#  CONTROLADOR PID CASCADA  (posición → actitud → torques)
# ──────────────────────────────────────────────────────────
class PIDController:
    """
    Controlador PID en cascada para el DJI F450, con Disturbance Observer (DObs).

    Lazo EXTERNO (posición → ángulos deseados):
      - Error XY → phi_d, theta_d  (actitud deseada)
      - Error Z  → u1              (empuje)

    Lazo INTERNO (actitud → torques):
      - Error (phi,theta,psi) → (u2, u3, u4)

    DISTURBANCE OBSERVER:
      Estima la perturbación de viento comparando la aceleración medida
      (Δvel/dt) con la aceleración comandada por empuje y gravedad.
      La estimación se usa como feedforward en los lazos Z y XY,
      reduciendo el error de estado estacionario bajo viento continuo.
      Ganancia D_GAIN=0.20 conservadora — no interfiere con la estabilidad.
    """
    def __init__(self, params: QuadParams):
        self.p = params
        # ── Lazo posición Z ──────────────────────────────────
        self.Kp_z  = 5.0
        self.Ki_z  = 0.12
        self.Kd_z  = 2.8
        # ── Lazo posición XY ─────────────────────────────────
        self.Kp_xy = 1.0
        self.Ki_xy = 0.03
        self.Kd_xy = 1.4
        # ── Lazo actitud ─────────────────────────────────────
        self.Kp_att = np.array([6.0, 6.0, 3.0])
        self.Ki_att = np.array([0.04, 0.04, 0.02])
        self.Kd_att = np.array([2.2, 2.2, 0.9])
        # ── Velocidades máximas ──────────────────────────────
        self.v_max_xy = 2.5
        self.v_max_z  = 1.5
        # ── Integradores y estados previos ───────────────────
        self._i_z      = 0.
        self._i_xy     = np.zeros(2)
        self._i_att    = np.zeros(3)
        self._prev_ez  = 0.
        self._prev_exy = np.zeros(2)
        self._prev_ea  = np.zeros(3)
        # ── Disturbance Observer ─────────────────────────────
        # d_hat: estimación de aceleración perturbadora [m/s²]
        # Filtro LP: d_hat[k] = 0.85*d_hat[k-1] + 0.15*(a_medida - a_cmd)
        # D_GAIN: fracción de compensación feedforward (0=off)
        self._d_hat    = np.zeros(3)
        self._prev_vel = np.zeros(3)
        self._prev_u1  = 9.81         # [N] — empuje del paso anterior
        self.D_GAIN    = 0.20         # conservador: no desestabiliza

    def reset(self):
        self._i_z   = 0.
        self._i_xy  = np.zeros(2)
        self._i_att = np.zeros(3)
        self._prev_ez  = 0.
        self._prev_exy = np.zeros(2)
        self._prev_ea  = np.zeros(3)
        self._d_hat    = np.zeros(3)
        self._prev_vel = np.zeros(3)
        self._prev_u1  = 9.81

    def compute(self, state: DroneState, target_pos, target_psi=0., dt=0.05):
        p   = self.p
        phi, theta, psi = state.angles

        # ── Disturbance Observer ─────────────────────────────
        # Aceleración medida en este paso
        a_meas = (state.vel - self._prev_vel) / (dt + 1e-9)
        # Aceleración que debería producir el empuje del paso anterior
        R      = rotation_matrix(phi, theta, psi)
        a_cmd  = (np.array([0., 0., -p.g])
                  + (R @ np.array([0., 0., self._prev_u1])) / p.m)
        # Perturbación estimada = diferencia, suavizada con filtro LP
        d_raw       = a_meas - a_cmd
        d_raw       = np.clip(d_raw, -5.0, 5.0)     # saturar ruido diferencial
        self._d_hat = 0.85 * self._d_hat + 0.15 * d_raw
        self._prev_vel = state.vel.copy()

        # ── Lazo Z: calcular u1 (empuje) ────────────────────
        ez         = target_pos[2] - state.pos[2]
        ez_sat     = np.clip(ez, -self.v_max_z*2, self.v_max_z*2)
        self._i_z  = np.clip(self._i_z + ez_sat*dt, -2.0, 2.0)
        dez        = (ez - self._prev_ez) / (dt + 1e-9)
        self._prev_ez = ez

        # Feedforward de gravedad + compensación de perturbación Z
        wind_ff_z = -self._d_hat[2] * self.D_GAIN
        az_des = (self.Kp_z * ez_sat
                  + self.Ki_z * self._i_z
                  + self.Kd_z * dez
                  + p.g + wind_ff_z)

        cos_tilt = max(np.cos(phi) * np.cos(theta), 0.1)
        u1 = p.m * az_des / cos_tilt
        u1 = float(np.clip(u1, 0., p.thrust_max()))
        self._prev_u1 = u1

        # ── Lazo XY: calcular ángulos deseados ──────────────
        exy        = target_pos[:2] - state.pos[:2]
        exy_sat    = np.clip(exy, -8.0, 8.0)
        self._i_xy = np.clip(self._i_xy + exy_sat*dt, -3.0, 3.0)
        dexy       = (exy - self._prev_exy) / (dt + 1e-9)
        self._prev_exy = exy

        # Aceleración XY deseada + compensación de perturbación XY
        wind_ff_xy = -self._d_hat[:2] * self.D_GAIN
        axy_des = (self.Kp_xy * exy_sat
                   + self.Ki_xy * self._i_xy
                   + self.Kd_xy * dexy
                   + wind_ff_xy)

        axy_max = p.g * np.tan(p.phi_max)
        axy_des = np.clip(axy_des, -axy_max, axy_max)

        ax_des, ay_des = axy_des
        phi_d   = float(np.clip(
            (ax_des*np.sin(psi) - ay_des*np.cos(psi)) / (p.g + 1e-6),
            -p.phi_max, p.phi_max))
        theta_d = float(np.clip(
            (ax_des*np.cos(psi) + ay_des*np.sin(psi)) / (p.g + 1e-6),
            -p.phi_max, p.phi_max))

        # ── Lazo actitud: torques (u2, u3, u4) ──────────────
        ea          = np.array([phi_d, theta_d, target_psi]) - state.angles
        ea[2]       = (ea[2] + np.pi) % (2*np.pi) - np.pi
        self._i_att = np.clip(self._i_att + ea*dt, -0.5, 0.5)
        dea         = (ea - self._prev_ea) / (dt + 1e-9)
        self._prev_ea = ea

        torques = (self.Kp_att * ea
                   + self.Ki_att * self._i_att
                   + self.Kd_att * dea)

        u2 = float(torques[0] * p.Ixx)
        u3 = float(torques[1] * p.Iyy)
        u4 = float(torques[2] * p.Izz)

        return u1, u2, u3, u4
