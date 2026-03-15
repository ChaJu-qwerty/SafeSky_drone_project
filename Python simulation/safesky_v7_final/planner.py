"""
planner.py — Planificadores de trayectoria SafeSky
===================================================
• HybridAStar    — Planificador global (ruta base)
• ORCAPlanner    — Planificador local (evasión inter-agente Y obstáculos)
• Obstacle       — Región de exclusión elíptica/esférica

CORRECCIONES v5:
  1. _free() usa dist_surface() en metros reales, NO is_inside() con
     métrica elipsoidal escalada.  El margen anterior (EPSILON+D_SAFE en
     espacio elipsoidal) equivalía a << D_SAFE metros para obstáculos
     de radio distinto de 1m — dejando waypoints demasiado cercanos.

  2. ORCAPlanner ahora recibe lista de obstáculos y genera semiplanos
     ORCA contra cada obstáculo estático, de modo que la corrección de
     velocidad también evita colisiones con paredes/objetos.

  3. D_SAFE y GRID_RES ajustados para garantizar separación suficiente.
"""
import numpy as np, heapq, time
from dataclasses import dataclass, field
from typing import List

SPACE    = 20.0
GRID_RES = 0.40   # resolución fina para encontrar rutas entre obstáculos
EPSILON  = 0.10   # tolerancia numérica libre
# FIX: D_SAFE aumentado de 0.50 → 0.65m para ser consistente con ORCA_RADIUS=0.495m
# Antes, A* generaba waypoints a 0.50m de superficie, pero ORCA necesita 0.495m libres
# → el dron entraba en zona de acción ORCA sobre WPs propios del A*, causando oscilación
D_SAFE   = 0.65   # m — debe ser ≥ ORCA_RADIUS(0.495) + margen(0.10) + EPSILON


# ──────────────────────────────────────────────────────────
#  OBSTÁCULO  (modelo elipsoidal)
# ──────────────────────────────────────────────────────────
class Obstacle:
    """
    X_obs = { p ∈ ℝ³ : (p-c)ᵀ Q (p-c) ≤ 1 }
    Q = diag(1/r²)
    """
    def __init__(self, center, radii):
        self.center = np.array(center, float)
        self.radii  = np.array(radii,  float)
        self.Q      = np.diag(1.0 / self.radii**2)

    def sdf(self, p):
        """
        Distancia con signo aproximada (metros reales):
          < 0 dentro del obstáculo
          > 0 fuera
        Se calcula como: (||Q^{1/2}(p-c)||₂ - 1) * r_min
        que da metros reales para el eje más pequeño.
        """
        d = p - self.center
        xi = np.sqrt(max(float(d @ self.Q @ d), 1e-12))  # norma elipsoidal
        return (xi - 1.0) * float(np.min(self.radii))

    def is_inside(self, p, margin=0.0):
        d = p - self.center
        return float(d @ self.Q @ d) < (1.0 + margin)

    def dist_surface(self, p):
        """Distancia positiva a la superficie en metros reales (0 si dentro)."""
        return max(0.0, self.sdf(p))

    def normal_outward(self, p):
        """Vector unitario apuntando hacia afuera desde el obstáculo."""
        d = p - self.center
        n = d / (np.linalg.norm(d) + 1e-9)
        return n


# ──────────────────────────────────────────────────────────
#  HYBRID A*
# ──────────────────────────────────────────────────────────
class HybridAStar:
    """
    A* sobre grilla 3-D con suavizado de trayectoria.
    f(n) = g(n) + h(n),  h = distancia euclidiana al goal.

    FIX: _free() usa dist_surface() en METROS REALES,
    garantizando que ningún waypoint quede a menos de D_SAFE metros
    de la superficie de cualquier obstáculo, independientemente de
    su forma o tamaño.
    """
    def __init__(self, obstacles: List[Obstacle]):
        self.obs = obstacles
        self.nodes_used = 0
        self.t_plan = 0.0
        # 26 vecinos (dx,dy,dz ∈ {-1,0,1} \ {0,0,0})
        self.moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0: continue
                    self.moves.append(np.array([dx, dy, dz], float) * GRID_RES)

    def _h(self, p, g):
        return np.linalg.norm(p - g)

    def _free(self, p, margin=0.):
        """
        Comprueba si el punto p está a al menos (D_SAFE + EPSILON + margin)
        metros de la superficie de todos los obstáculos — en METROS REALES.
        """
        if np.any(p < 0) or np.any(p > SPACE):
            return False
        clearance = D_SAFE + EPSILON + margin
        for o in self.obs:
            if o.dist_surface(p) < clearance:
                return False
        return True

    def plan(self, start, goal):
        t0   = time.perf_counter()
        heap = [(self._h(start, goal), 0, start.tolist(), [start.copy()])]
        visited, best = {}, {}
        self.nodes_used = 0
        itr = 0
        while heap and itr < 80000:
            itr += 1
            f, _, cl, path = heapq.heappop(heap)
            curr = np.array(cl)
            gk   = tuple(np.round(curr / GRID_RES).astype(int))
            if gk in visited: continue
            visited[gk] = True
            self.nodes_used += 1
            if self._h(curr, goal) < GRID_RES * 1.5:
                path.append(goal.copy())
                self.t_plan = time.perf_counter() - t0
                return self._smooth(path)
            gc = f - self._h(curr, goal)
            for mv in self.moves:
                nxt = curr + mv
                if not self._free(nxt): continue
                nk = tuple(np.round(nxt / GRID_RES).astype(int))
                if nk in visited: continue
                gn = gc + np.linalg.norm(mv)
                if nk in best and best[nk] <= gn: continue
                best[nk] = gn
                heapq.heappush(heap, (gn + self._h(nxt, goal), itr,
                                      nxt.tolist(), path + [nxt.copy()]))
        # Fallback: línea recta
        self.t_plan = time.perf_counter() - t0
        return [start + t * (goal - start) for t in np.linspace(0, 1, 30)]

    def _smooth(self, path, iters=5):
        s = path[:]
        for _ in range(iters):
            ns, i = [s[0]], 0
            while i < len(s) - 1:
                j = len(s) - 1
                while j > i + 1:
                    if all(self._free(s[i] + t * (s[j] - s[i]))
                           for t in np.linspace(0, 1, 12)[1:-1]):
                        break
                    j -= 1
                ns.append(s[j])
                i = j
            s = ns
        return s


# ──────────────────────────────────────────────────────────
#  ORCA  (Optimal Reciprocal Collision Avoidance)
#  v5: también evita obstáculos estáticos
# ──────────────────────────────────────────────────────────
class ORCAPlanner:
    """
    ORCA para N agentes + evasión de obstáculos estáticos.

    Para cada par de drones genera el semiplano ORCA estándar
    (van den Berg et al. 2011).

    Para cada obstáculo, modela el obstáculo como un dron
    estático de radio igual al radio mínimo del elipsoide +
    el radio del dron, y aplica la misma lógica ORCA pero
    con velocidad del obstáculo = 0 (el dron asume toda la
    responsabilidad de evasión → factor 1.0 en vez de 0.5).

    Referencia: van den Berg et al. (2011) ISRR.
    """
    def __init__(self, radius=0.35, tau=2.0, v_max=2.5, obstacles=None):
        self.r     = radius    # radio del dron [m]
        # FIX: tau reducido de 3.0→2.0s. tau grande hace que ORCA empiece a
        # actuar 6m antes del obstáculo (v_max*tau=2*3=6m) generando empujes
        # permanentes que compiten con obs_push del PID → oscilación orbital.
        # tau=2.0 → zona de acción = 4m, suficiente para reaccionar a 2 m/s.
        self.tau   = tau
        self.v_max = v_max
        self.obstacles = obstacles or []   # lista de Obstacle

    def _orca_halfplane_agent(self, pa, va, pb, vb):
        """
        Semiplano ORCA entre dos agentes móviles.
        Responsabilidad compartida (0.5 cada uno).
        Devuelve (u_point, normal) o None.
        """
        rp   = pb - pa
        rv   = va - vb
        d    = np.linalg.norm(rp)
        cr   = 2.0 * self.r          # suma de radios
        if d < 1e-6: return None

        # Vector de velocidad relativa respecto al VO truncado
        w  = rv - rp / self.tau
        wl = np.linalg.norm(w)

        leg2 = d**2 - cr**2
        if leg2 <= 0:
            # Dentro del VO truncado — proyectar sobre el círculo
            if wl < 1e-9: return None
            n = w / wl
            u = (cr / self.tau - wl) * n
            return (va + 0.5 * u, n)

        # Proyectar sobre las piernas del cono
        sl    = np.sqrt(leg2)
        # Leg izquierda
        leg_l = np.array([rp[0]*sl - rp[1]*cr,
                           rp[0]*cr + rp[1]*sl]) / (d**2)
        n_l   = np.array([ leg_l[1], -leg_l[0], 0.])
        leg_l = np.append(leg_l, 0.)
        # Leg derecha
        leg_r = np.array([rp[0]*sl + rp[1]*cr,
                          -rp[0]*cr + rp[1]*sl]) / (d**2)
        n_r   = np.array([-leg_r[1],  leg_r[0], 0.])
        leg_r = np.append(leg_r, 0.)

        w3 = rv
        # Decidir qué leg usar con el producto cruzado Z
        cross_l = leg_l[0]*w3[1] - leg_l[1]*w3[0]
        if cross_l >= 0:
            n3 = n_l
        else:
            n3 = n_r

        n3l = np.linalg.norm(n3) + 1e-9
        n3  = n3 / n3l
        u3  = (np.dot(rv, n3) - (1.0 / self.tau) *
               (np.dot(rp, n3) - cr)) * n3
        if np.linalg.norm(u3) < 1e-9: return None
        return (va + 0.5 * u3, n3)

    def _orca_halfplane_obstacle(self, pa, va, obs):
        """
        Semiplano ORCA entre un dron y un obstáculo ESTÁTICO.

        Modelamos el obstáculo como un agente estático de radio
        r_obs = min(radii) + r_drone en la dirección más cercana.
        El dron toma toda la responsabilidad (factor 1.0).

        Si el dron ya está en zona peligrosa (dist < D_SAFE),
        la corrección es perpendicular hacia afuera del obstáculo.
        """
        # Punto más cercano de la superficie del obstáculo
        # Aproximamos con el vector de la dirección del dron al centro
        d_vec = pa - obs.center
        d_mag = np.linalg.norm(d_vec)
        if d_mag < 1e-6:
            # Dentro del obstáculo — empujar hacia arriba
            return (va + np.array([0., 0., 1.]) * self.v_max, np.array([0., 0., 1.]))

        # Radio efectivo del obstáculo en la dirección del dron
        d_norm = d_vec / d_mag
        # Proyectar en los ejes del obstáculo para obtener radio efectivo
        r_eff  = 1.0 / np.sqrt(max(float(d_norm @ obs.Q @ d_norm), 1e-9))
        cr     = r_eff + self.r    # suma de radios: obstáculo + dron

        # Velocidad relativa (obstáculo estático → vb = 0)
        rv = va
        rp = obs.center - pa       # de dron al obstáculo

        d_dist = d_mag
        w  = rv - rp / self.tau
        wl = np.linalg.norm(w)

        leg2 = d_dist**2 - cr**2
        if leg2 <= 0:
            # Dron demasiado cerca — evasión de emergencia
            if wl < 1e-9:
                n = d_norm
            else:
                n = w / wl
            u = (cr / self.tau - wl) * n
            # Dron toma toda la responsabilidad (factor 1.0)
            return (va + u, n)

        # Calcular semiplano en el plano 2D (x,y) ignorando Z
        # para el componente horizontal, luego corregir Z
        rp2 = rp[:2]
        d2  = np.linalg.norm(rp2)
        if d2 < 1e-6:
            # Obstáculo directamente arriba/abajo — solo componente Z
            sign = 1.0 if pa[2] > obs.center[2] else -1.0
            n = np.array([0., 0., sign])
            u = max(0., cr / self.tau - abs(va[2])) * n
            return (va + u, n)

        sl   = np.sqrt(max(d_dist**2 - cr**2, 0.))
        leg_l = np.array([rp2[0]*sl - rp2[1]*cr,
                           rp2[0]*cr + rp2[1]*sl]) / (d_dist**2)
        n_l   = np.array([ leg_l[1], -leg_l[0], 0.])
        leg_r = np.array([rp2[0]*sl + rp2[1]*cr,
                          -rp2[0]*cr + rp2[1]*sl]) / (d_dist**2)
        n_r   = np.array([-leg_r[1],  leg_r[0], 0.])

        cross_l = leg_l[0]*rv[1] - leg_l[1]*rv[0]
        n3 = n_l if cross_l >= 0 else n_r

        n3l = np.linalg.norm(n3) + 1e-9
        n3  = n3 / n3l
        u3  = (np.dot(rv, n3) -
               (1.0 / self.tau) * (np.dot(rp, n3) - cr)) * n3
        if np.linalg.norm(u3) < 1e-9: return None

        # Dron toma TODA la responsabilidad de evasión (factor 1.0)
        return (va + u3, n3)

    def compute_velocities(self, positions, v_prefs):
        """
        Calcula velocidades ajustadas por ORCA para N drones,
        considerando colisiones con otros drones Y con obstáculos.
        """
        N   = len(positions)
        out = [v.copy() for v in v_prefs]

        for i in range(N):
            v = v_prefs[i].copy()

            # ── Semiplanos ORCA vs otros drones ───────────────
            for j in range(N):
                if i == j: continue
                hp = self._orca_halfplane_agent(
                    positions[i], v_prefs[i],
                    positions[j], v_prefs[j])
                if hp is None: continue
                pt, nv = hp
                if np.dot(v - pt, nv) < 0.:
                    v = v - np.dot(v - pt, nv) * nv

            # ── Semiplanos ORCA vs obstáculos estáticos ───────
            for obs in self.obstacles:
                # Solo actuar si el dron está dentro del horizonte
                dist = obs.dist_surface(positions[i])
                if dist > self.v_max * self.tau + obs.radii.max():
                    continue   # demasiado lejos — sin efecto
                hp = self._orca_halfplane_obstacle(
                    positions[i], v, obs)
                if hp is None: continue
                pt, nv = hp
                if np.dot(v - pt, nv) < 0.:
                    v = v - np.dot(v - pt, nv) * nv

            spd = np.linalg.norm(v)
            if spd > self.v_max:
                v = v / spd * self.v_max
            out[i] = v

        return out
