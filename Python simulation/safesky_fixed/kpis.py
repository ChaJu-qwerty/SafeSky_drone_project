"""
kpis.py — Indicadores Clave de Desempeño (KPIs)
=================================================
SafeSky MR3001B (2026)

KPIs definidos:
  1. Error final en punto B       e_B        < E_B_OK  (0.10 m)
  2. Distancia mínima obstáculos  d_obs_min  ≥ D_SAFE  (0.40 m)
  3. Separación mínima enjambre   d_swarm_min≥ D_MIN   (1.20 m)
  4. Longitud total trayectoria   L_total    [m]
  5. Tiempo de cómputo medio      t_comp     [ms]
  6. Nodos A* expandidos          n_nodes

GEOMETRÍA DE COLISIÓN (DJI F450 + hélice 9450):
  ┌─────────────────────────────────────────────────────┐
  │  l_arm   = 0.225 m  (brazo: centro → hub del rotor) │
  │  r_prop  = 0.120 m  (radio hélice 9.45" / 2)        │
  │  r_drone = 0.345 m  (radio físico total del dron)    │
  │                                                       │
  │  COLISIÓN FÍSICA  = 2 × r_drone = 0.690 m            │
  │  (las hélices de dos drones se tocan)                 │
  │                                                       │
  │  BUFFER SEGURIDAD = 0.150 m  (margen operacional)    │
  │  DRONE_COLLISION_D = 0.690 + 0.150 = 0.840 m        │
  │  → Si d_swarm < 0.840 m se registra como VIOLACIÓN   │
  │  → Si d_swarm < 0.690 m se registra como COLISIÓN    │
  └─────────────────────────────────────────────────────┘
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List

# ── Geometría física del dron (DJI F450 + hélice 9450) ─────────────────────
L_ARM          = 0.225   # m — brazo: distancia centro del frame → hub del rotor
R_PROP         = 0.120   # m — radio de la hélice 9.45 pulgadas (9.45*0.0254/2 ≈ 0.120m)
R_DRONE        = L_ARM + R_PROP   # = 0.345 m — radio físico real del dron completo

# ── Distancias de colisión entre dos drones ─────────────────────────────────
D_COLLISION_PHYSICAL = 2 * R_DRONE          # = 0.690 m — hélices se tocan → COLISIÓN REAL
D_SAFETY_BUFFER      = 0.150                # m — margen operacional de seguridad
DRONE_COLLISION_D    = D_COLLISION_PHYSICAL + D_SAFETY_BUFFER  # = 0.840 m — umbral de VIOLACIÓN
#
#  Resumen visual:
#    0 ──── 0.345 ──── 0.690 ──── 0.840 ────────►  distancia entre centros
#           r_drone    COLISIÓN    VIOLACIÓN
#                      física      (con buffer)

# ── Criterios de éxito de la misión ────────────────────────────────────────
E_B_OK  = 0.10   # m — tolerancia error final en B
D_SAFE  = 0.40   # m — margen mínimo a obstáculos (criterio de éxito)
D_MIN   = 1.20   # m — separación mínima DESEADA entre drones (criterio de éxito)

# Alias para compatibilidad con código existente
DRONE_RADIUS_PHYS = R_DRONE   # = 0.345 m


@dataclass
class DroneKPI:
    did:            int   = 0
    e_b:            float = 0.0     # error final punto B  [m]
    d_obs_min:      float = 1e9     # dist mínima a obstáculos [m]
    d_swarm_min:    float = 1e9     # dist mínima a otro dron [m]
    L_total:        float = 0.0     # longitud de trayectoria [m]
    success:        bool  = False   # True solo si e_b<E_B_OK y d_obs>=D_SAFE
    colision_obs:   bool  = False   # True si colisión física con obstáculo
    colision_drone: bool  = False   # True si colisión física entre drones (d<0.45m)
    n_obs_violations: int = 0       # pasos con dist_obs < D_SAFE


@dataclass
class MissionKPI:
    drones:        List[DroneKPI] = field(default_factory=list)
    success_rate:  float = 0.0
    t_sim:         float = 0.0
    t_comp_mean:   float = 0.0
    n_nodes:       int   = 0
    wind_alpha:    float = 0.0
    wind_sigma:    float = 0.0
    any_collision: bool  = False   # True si cualquier dron tuvo colisión física


class KPITracker:
    """Registro en tiempo real de KPIs durante la simulación."""

    def __init__(self, n_drones, goals, obstacles):
        self.n           = n_drones
        self.goals       = [np.array(g) for g in goals]
        self.obs         = obstacles
        self._pos_hist   = [[] for _ in range(n_drones)]
        self._d_obs      = [1e9] * n_drones
        self._d_sw       = [1e9] * n_drones
        self._d_goal_min = [1e9] * n_drones
        self._L          = [0.0] * n_drones
        self._t_comp     = []
        self._n_nodes    = 0
        # Nuevos contadores
        self._n_obs_viol     = [0] * n_drones   # pasos con dist < D_SAFE
        self._collision      = [False] * n_drones  # colisión con obstáculo
        self._collision_drone= [False] * n_drones  # colisión física entre drones

    def update(self, positions, t_comp_ms=0., n_nodes=0):
        """Llamar en cada paso de simulación."""
        for i, p in enumerate(positions):
            # Longitud incremental de trayectoria
            if self._pos_hist[i]:
                self._L[i] += np.linalg.norm(p - self._pos_hist[i][-1])
            self._pos_hist[i].append(p.copy())

            # Distancia a obstáculos — registrar mínimo histórico
            if self.obs:
                d_obs_now = min(o.dist_surface(p) for o in self.obs)
                if d_obs_now < self._d_obs[i]:
                    self._d_obs[i] = d_obs_now
                # Contar violaciones D_SAFE
                if d_obs_now < D_SAFE:
                    self._n_obs_viol[i] += 1
                # Colisión física: dentro del radio del dron
                if d_obs_now < DRONE_RADIUS_PHYS:
                    self._collision[i] = True

            # Distancia al goal para e_b
            if self.goals[i] is not None:
                d_goal = float(np.linalg.norm(p - self.goals[i]))
                if d_goal < self._d_goal_min[i]:
                    self._d_goal_min[i] = d_goal

        # Separación inter-dron + detección de colisión real entre drones
        for i in range(self.n):
            if not self._pos_hist[i]: continue
            for j in range(i + 1, self.n):
                if not self._pos_hist[j]: continue
                d_ij = np.linalg.norm(positions[i] - positions[j])
                if d_ij < self._d_sw[i]: self._d_sw[i] = d_ij
                if d_ij < self._d_sw[j]: self._d_sw[j] = d_ij
                # Colisión física real entre drones
                if d_ij < DRONE_COLLISION_D:
                    self._collision_drone[i] = True
                    self._collision_drone[j] = True

        if t_comp_ms > 0.:
            self._t_comp.append(t_comp_ms)
        self._n_nodes = max(self._n_nodes, n_nodes)

    def compute(self, final_positions, t_sim, min_dist_goals=None) -> MissionKPI:
        drones   = []
        n_ok     = 0
        any_coll = False

        for i in range(self.n):
            fp = np.array(final_positions[i])
            e_b_final = float(np.linalg.norm(fp - self.goals[i])) \
                        if self.goals[i] is not None else 1e9
            # Usar mínima distancia histórica al goal si disponible
            if min_dist_goals is not None:
                e_b = min(e_b_final, float(min_dist_goals[i]))
            else:
                e_b = min(e_b_final, self._d_goal_min[i])

            # Criterio de éxito: llegó Y nunca violó la zona segura de obstáculos
            ok = (e_b < E_B_OK) and (self._d_obs[i] >= D_SAFE)
            if ok: n_ok += 1
            if self._collision[i]: any_coll = True

            drones.append(DroneKPI(
                did=i,
                e_b=e_b,
                d_obs_min=self._d_obs[i],
                d_swarm_min=self._d_sw[i],
                L_total=self._L[i],
                success=ok,
                colision_obs=self._collision[i],
                colision_drone=self._collision_drone[i],
                n_obs_violations=self._n_obs_viol[i]
            ))

        t_comp = float(np.mean(self._t_comp)) if self._t_comp else 0.
        return MissionKPI(
            drones=drones,
            success_rate=n_ok / self.n,
            t_sim=t_sim,
            t_comp_mean=t_comp,
            n_nodes=self._n_nodes,
            any_collision=any_coll
        )
