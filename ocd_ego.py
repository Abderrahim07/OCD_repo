"""
ocd_ego.py — pilote le véhicule ego d'un test LDW avec OCD au lieu de la
physique CARLA.  CARLA ne sert plus qu'à (1) la géométrie de la route,
(2) le rendu caméra.  Tout ce qui est plain Python (pas d'import carla au
niveau module) pour pouvoir être testé sans CARLA.

Conventions
-----------
* OCD  : repère ISO  -> x devant, y à GAUCHE, lacet psi positif = vers la gauche.
* CARLA: x devant, y à DROITE, yaw positif = vers la droite (degrés).
  => la conversion inverse y et le lacet (c'est fait ici, une seule fois).
"""
import math
from ocd_vehicle_dynamics import OCDVehicleModel, EngineGearbox

# ----------------------------------------------------------------------
# Profils de véhicule prêts à l'emploi (mêmes valeurs que test_3_vehicules.py)
# Dans test_case.json :  "ocd_vehicle": "berline"   ou un dict complet
# {"params": {...}, "engine": {...}, "brake_n": ...}
# ----------------------------------------------------------------------
PRESETS = {
    "citadine": dict(
        params=dict(mass_kg=1000.0, wheelbase_m=2.5, cog_distance_from_front_axle_m=1.3,
                    moment_of_inertia_z_kgpm2=1200.0, drag_coefficient=0.32, frontal_area_m2=2.0,
                    tire_rolling_resistance=0.012, wheel_radius_m=0.29),
        engine=dict(torque_curve=[(900, 70), (2000, 110), (3500, 130), (5000, 120), (6000, 100)],
                    gear_ratios=(3.7, 2.0, 1.3, 0.95, 0.76), final_drive=4.1, efficiency=0.90,
                    idle_rpm=800, max_rpm=6000, upshift_rpm=5200, downshift_rpm=1800,
                    engine_brake_nm_at_max_rpm=35),
        brake_n=8000.0),
    "berline": dict(
        params=dict(mass_kg=1500.0, wheelbase_m=2.7, cog_distance_from_front_axle_m=1.2,
                    moment_of_inertia_z_kgpm2=2500.0, drag_coefficient=0.28, frontal_area_m2=2.2,
                    tire_rolling_resistance=0.011, wheel_radius_m=0.32),
        engine=dict(torque_curve=[(900, 120), (1500, 190), (2500, 250), (3500, 280), (4500, 275),
                                  (5500, 250), (6500, 210)],
                    gear_ratios=(3.6, 2.1, 1.4, 1.0, 0.8, 0.65), final_drive=3.7),
        brake_n=11000.0),
    "suv": dict(
        params=dict(mass_kg=2300.0, wheelbase_m=3.0, cog_distance_from_front_axle_m=1.4,
                    moment_of_inertia_z_kgpm2=4200.0, drag_coefficient=0.38, frontal_area_m2=2.9,
                    tire_rolling_resistance=0.013, wheel_radius_m=0.36),
        engine=dict(torque_curve=[(800, 250), (1500, 420), (2000, 450), (3000, 420), (4000, 340),
                                  (4800, 260)],
                    gear_ratios=(4.2, 2.5, 1.6, 1.2, 0.9, 0.7), final_drive=3.9, efficiency=0.90,
                    idle_rpm=800, max_rpm=4800, upshift_rpm=4000, downshift_rpm=1500,
                    engine_brake_nm_at_max_rpm=70),
        brake_n=14000.0),
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class OCDEgo:
    """Véhicule ego simulé par OCD, placé dans le repère monde CARLA."""

    def __init__(self, spawn_xy_yaw_deg, vehicle="berline", fps=20, kp_speed=0.4, ki_speed=0.004):
        cfg = PRESETS[vehicle.lower()] if isinstance(vehicle, str) else vehicle
        self.cfg = cfg
        self.dt = 1.0 / fps
        self.kp = kp_speed
        self.ki = ki_speed
        self.wheelbase = float(cfg["params"].get("wheelbase_m", 2.7))
        self.model = OCDVehicleModel(
            dynamics_params=cfg["params"],
            engine=EngineGearbox(**{"wheel_radius_m": cfg["params"].get("wheel_radius_m", 0.3), **cfg["engine"]}),
            max_brake_force_n=cfg.get("brake_n", 10000.0),
        )
        self.spawn = spawn_xy_yaw_deg       # (x, y, yaw_deg) CARLA — pose au début de l'enregistrement
        self._ref = (0.0, 0.0, 0.0)         # pose OCD (x, y, psi) prise comme origine
        self._state = self.model.step(0.0, 0.0, 0.0, self.dt)
        self._path_idx = 0
        self._i_speed = 0.0                 # terme intégral du régulateur de vitesse

    # ---------------------------------------------------------------- vitesse
    def _speed_cmd(self, target_kmh):
        """Régulateur PI : le P seul laisserait un écart permanent (traînée,
        frottement) et la voiture n'atteindrait jamais la vitesse du test."""
        err = target_kmh - 3.6 * self._state["v_x_mps"]
        u = self.kp * err / 10.0 + self._i_speed
        if abs(u) < 1.0 and abs(err) < 12.0:  # anti-emballement : on n'intègre que près de la cible
            self._i_speed = _clamp(self._i_speed + self.ki * err * self.dt, -0.3, 0.6)
        if u >= 0:
            return _clamp(u, 0.0, 1.0), 0.0
        return 0.0, _clamp(-u, 0.0, 1.0)

    def preroll(self, target_kmh, max_s=60.0, tol_kmh=1.0):
        """Accélère en ligne droite jusqu'à la vitesse du test (rien n'est
        rendu).  La pose atteinte devient l'origine : la voiture apparaît
        dans CARLA au point de spawn, déjà à vitesse."""
        for _ in range(int(max_s / self.dt)):
            if abs(3.6 * self._state["v_x_mps"] - target_kmh) < tol_kmh:
                break
            th, br = self._speed_cmd(target_kmh)
            self._state = self.model.step(th, 0.0, br, self.dt)
        s = self._state
        self._ref = (s["x_m"], s["y_m"], s["psi_rad"])
        return 3.6 * s["v_x_mps"]

    # ------------------------------------------------------------------ pose
    def pose(self):
        """(x, y, yaw_deg) dans le repère monde CARLA."""
        s, (x0, y0, p0) = self._state, self._ref
        dx, dy = s["x_m"] - x0, s["y_m"] - y0
        rel_x = math.cos(p0) * dx + math.sin(p0) * dy          # devant
        rel_y_left = -math.sin(p0) * dx + math.cos(p0) * dy    # gauche (ISO)
        rel_psi = s["psi_rad"] - p0
        sx, sy, syaw = self.spawn
        yl = -rel_y_left                                       # -> droite (CARLA)
        c, sn = math.cos(math.radians(syaw)), math.sin(math.radians(syaw))
        return (sx + rel_x * c - yl * sn,
                sy + rel_x * sn + yl * c,
                syaw - math.degrees(rel_psi))

    @property
    def speed_kmh(self):
        return 3.6 * self._state["v_x_mps"]

    @property
    def state(self):
        return self._state

    # ---------------------------------------------------------------- pilote
    def _lookahead_target(self, path_xy, x, y):
        # point de la trajectoire le plus proche (recherche vers l'avant seulement)
        i = self._path_idx
        while i + 1 < len(path_xy) and math.dist(path_xy[i + 1], (x, y)) <= math.dist(path_xy[i], (x, y)):
            i += 1
        self._path_idx = i
        ld = max(4.0, 0.7 * self._state["v_x_mps"])            # distance de visée (m)
        j, acc = i, 0.0
        while j + 1 < len(path_xy) and acc < ld:
            acc += math.dist(path_xy[j], path_xy[j + 1])
            j += 1
        return path_xy[j]

    def drive(self, path_xy, target_kmh):
        """Un pas (1/fps s) : pilote le long de path_xy (liste de (x, y) CARLA)
        à target_kmh, puis intègre OCD."""
        x, y, yaw_deg = self.pose()
        tx, ty = self._lookahead_target(path_xy, x, y)
        yaw = math.radians(yaw_deg)
        dx, dy = tx - x, ty - y
        lx = dx * math.cos(yaw) + dy * math.sin(yaw)           # devant
        ly = -dx * math.sin(yaw) + dy * math.cos(yaw)          # droite (CARLA)
        ld2 = max(lx * lx + ly * ly, 4.0)
        delta_right = math.atan(self.wheelbase * 2.0 * ly / ld2)   # braquage pure pursuit (rad)
        max_steer = 0.3                                        # angle_max_rad du PT1 OCD
        steer_iso = _clamp(-delta_right / max_steer, -1.0, 1.0)    # OCD : + = gauche
        th, br = self._speed_cmd(target_kmh)
        self._state = self.model.step(th, steer_iso, br, self.dt)
        return self._state
