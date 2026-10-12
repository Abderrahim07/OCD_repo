"""
Wrapper autour d'Open Car Dynamics (TUMFTM, Apache 2.0 —
https://github.com/TUMFTM/Open-Car-Dynamics) = modèle dynamique du
véhicule de la plateforme vECU, SANS simulateur.

PROTOTYPE NON TESTÉ À L'EXÉCUTION : OCD ne peut pas être buildé dans
l'environnement où ce fichier a été écrit (pas de ROS 2 / C++20). Tout ce
qui touche l'API OCD a été lu dans le code source, pas exécuté.

Architecture de ce fichier
--------------------------
  throttle [0,1] / brake [0,1] / steer [-1,1]      (entrées "type CARLA")
        │
        ▼
  EngineGearbox  (couche ajoutée ICI, OCD n'a ni moteur ni boîte)
     courbe de couple, rapports, pont, rendement, rpm, autobox,
     coupure de couple pendant le changement de rapport, frein moteur
        │  force longitudinale signée aux roues (N)
        ▼
  OCD  drivetrain=FX  (+ PT1 + SINGLE_TRACK + LINEAR + aéro DEFAULT)
        │  FX : Fx > 0 = traction, Fx < 0 = freinage (réparti par
        │  drivetrain.brake_bias_front, converti en couple avec
        │  drivetrain.wheel_radius_m)
        ▼
  état (x, y, psi, vx, vy) + rpm / rapport engagé

VÉRIFIÉ dans le code source
- Signature de VehicleFactory.create_vehicle() et enums (submodel_choices.py)
- FX : entrée = un seul double (N). Positif -> couple de traction
  (F * wheel_radius), négatif -> freinage réparti AV/AR
  (drivetrain_fx_model.hpp). Params : drivetrain.brake_bias_front (0.6),
  drivetrain.wheel_radius_m (0.3), drivetrain.intertia_kgm2 (0.0)
- Direction PT1 : steering_angle_rad ; params angle_max_rad, T_PT1...
- Sorties : vehicle_dynamics_output.{position_m, velocity_mps,
  orientation_rad} = Vector3D (.x .y .z) — confirmé via tam__common
- Pas d'intégration : 0.0008 s

NON VÉRIFIÉ (à tester après build)
(vérifié chez l'utilisateur : construct_drivetrain_input() renvoie un float,
 les clés drivetrain.* / steering_actuator.* existent)
- Stabilité : voir DEFAULT_TIRE_C_SX / DEFAULT_TIRE_C_ALPHA ci-dessous
- orientation_rad.z = lacet en single track

LIMITES (volontaires, prototype)
- Marche avant uniquement (pas de marche arrière).
- Autobox simple (seuils rpm), pas d'embrayage physique : à basse vitesse
  le rpm est plafonné au ralenti (glissement d'embrayage implicite).
- Le moteur est modélisé dans le wrapper, pas dans OCD : la dynamique
  d'inertie moteur reste à 0 (drivetrain.intertia_kgm2).
- Pas d'ABS/ESC : c'est le rôle du logiciel ECU testé.
- Paramètres par défaut OCD = monoplace générique : à identifier pour
  votre véhicule cible.

Build local
  colcon build --packages-up-to ocd_vehicle_models_py \\
      --cmake-args -DCMAKE_BUILD_TYPE=Release
  (ou l'image Docker du dépôt)
"""

import bisect
import math

try:
    from ocd_vehicle_models_py import (
        AerodynamicsType,
        DrivetrainType,
        SteeringActuatorType,
        TireType,
        VehicleDynamicsType,
        VehicleFactory,
    )
    OCD_AVAILABLE = True
except ImportError:
    OCD_AVAILABLE = False


# Mapping de notre schéma "vehicle_dynamics" (celui de
# configure_vehicle_dynamics côté CARLA) vers les clés OCD.
_PARAM_KEY_MAP = {
    "mass_kg": "vehicle_dynamics_single_track.mass_vehicle_kg",
    "wheelbase_m": "vehicle_dynamics_single_track.wheelbase_m",
    "cog_distance_from_front_axle_m": "vehicle_dynamics_single_track.cog.distance_from_front_axle_m",
    "moment_of_inertia_z_kgpm2": "vehicle_dynamics_single_track.moment_of_inertia_kgpm2.z",
    "drag_coefficient": "vehicle_dynamics_single_track.aerodynamics.c_d",
    "frontal_area_m2": "vehicle_dynamics_single_track.aerodynamics.A_m2",
    "tire_rolling_resistance": "vehicle_dynamics_single_track.tire.rolling_resistance_coefficient",
    # Drivetrain FX
    "wheel_radius_m": "drivetrain.wheel_radius_m",
    "brake_bias_front": "drivetrain.brake_bias_front",
    "engine_inertia_kgm2": "drivetrain.intertia_kgm2",  # (sic) faute d'orthographe dans OCD
}

_TIRE_WHEELS = ["front_left", "front_right", "rear_left", "rear_right"]

DEFAULT_MAX_BRAKE_FORCE_N = 10000.0  # force de freinage totale à brake=1 (à calibrer)
DEFAULT_WHEEL_RADIUS_M = 0.3         # défaut OCD (drivetrain.wheel_radius_m)

# ⚠️ Les défauts du pneu LINEAR d'OCD (C_sx=3000, C_alpha=-0.1) sont des
# placeholders NON PHYSIQUES : avec le pas fixe de 0,8 ms le modèle diverge
# (NaN / vitesses de -100 m/s dès qu'on applique une force). Vérifié en
# compilant la lib C++ et en simulant. Valeurs réalistes (voiture de
# tourisme, pneu sec), à identifier pour votre véhicule :
#   Fx = C_sx * kappa * Fz      (C_sx ~ 10-20, sans unité)
#   Fy = C_alpha * alpha * Fz   (convention OCD ; avec alpha = -atan(vy/vx),
#                                un C_alpha POSITIF ~ 8-12 /rad est stable,
#                                le défaut négatif ne l'est pas)
DEFAULT_TIRE_C_SX = 15.0

# ⚠️ Limite de stabilité numérique (mesurée en simulant la lib C++) : avec le
# pas fixe de 0,8 ms, le produit C_sx x masse doit rester sous ~15 000.
# Au-delà, le véhicule "cale" vers 6 m/s (la roue devient trop raide pour
# l'intégrateur) : ex. 1500 kg avec C_sx=15 ne dépasse jamais 5,9 m/s.
# La raideur longitudinale influence très peu le LDW (dynamique latérale),
# donc par défaut on la réduit quand la masse augmente.
MAX_CSX_TIMES_MASS = 12000.0
DEFAULT_TIRE_C_ALPHA = 10.0


def _coerce(old_value, new_value):
    """OCD refuse un int là où un double est déclaré (mass_kg=1500 plante) :
    on convertit en float quand le paramètre existant est un float."""
    if isinstance(old_value, float) and isinstance(new_value, (int, float)) \
            and not isinstance(new_value, bool):
        return float(new_value)
    return new_value


class EngineGearbox:
    """Couche moteur + boîte de vitesses (absente d'OCD).

    Convertit (throttle, vitesse, brake) en force longitudinale signée aux
    roues (N), entrée attendue par le drivetrain FX.

        F_drive = T_moteur(rpm, throttle) * rapport * pont * rendement / r_roue
    """

    def __init__(
        self,
        torque_curve=None,              # [(rpm, Nm à pleine charge), ...] croissant en rpm
        gear_ratios=(3.6, 2.1, 1.4, 1.0, 0.8, 0.65),
        final_drive=3.7,
        efficiency=0.92,
        wheel_radius_m=DEFAULT_WHEEL_RADIUS_M,
        idle_rpm=900.0,
        max_rpm=6500.0,
        upshift_rpm=5800.0,
        downshift_rpm=2200.0,
        shift_time_s=0.25,              # durée de coupure de couple
        engine_brake_nm_at_max_rpm=60.0,  # couple résistant à throttle=0 à max_rpm
        use_autobox=True,
    ):
        self.torque_curve = torque_curve or [
            (900.0, 120.0), (1500.0, 190.0), (2500.0, 250.0), (3500.0, 280.0),
            (4500.0, 275.0), (5500.0, 250.0), (6500.0, 210.0),
        ]
        self._rpms = [p[0] for p in self.torque_curve]
        self._torques = [p[1] for p in self.torque_curve]

        self.gear_ratios = tuple(gear_ratios)
        self.final_drive = final_drive
        self.efficiency = efficiency
        self.wheel_radius_m = wheel_radius_m
        self.idle_rpm = idle_rpm
        self.max_rpm = max_rpm
        self.upshift_rpm = upshift_rpm
        self.downshift_rpm = downshift_rpm
        self.shift_time_s = shift_time_s
        self.engine_brake_nm_at_max_rpm = engine_brake_nm_at_max_rpm
        self.use_autobox = use_autobox

        self.gear = 1                   # 1..len(gear_ratios)
        self.rpm = idle_rpm
        self._shift_timer_s = 0.0

    # -- courbe de couple (interpolation linéaire, saturée aux bornes) ----
    def _full_load_torque(self, rpm):
        if not math.isfinite(rpm):
            rpm = self.idle_rpm
        if rpm <= self._rpms[0]:
            return self._torques[0]
        if rpm >= self._rpms[-1]:
            # au-delà du dernier point : chute linéaire jusqu'à 0 à max_rpm
            span = max(self.max_rpm - self._rpms[-1], 1.0)
            return max(0.0, self._torques[-1] * (self.max_rpm - rpm) / span)
        i = bisect.bisect_right(self._rpms, rpm)
        r0, r1 = self._rpms[i - 1], self._rpms[i]
        t0, t1 = self._torques[i - 1], self._torques[i]
        return t0 + (t1 - t0) * (rpm - r0) / (r1 - r0)

    def _overall_ratio(self):
        return self.gear_ratios[self.gear - 1] * self.final_drive

    def _rpm_from_speed(self, v_x_mps):
        if not math.isfinite(v_x_mps):
            v_x_mps = 0.0
        wheel_rps = max(v_x_mps, 0.0) / self.wheel_radius_m / (2.0 * math.pi)
        return wheel_rps * self._overall_ratio() * 60.0

    # -- autobox -----------------------------------------------------------
    def _update_gear(self, v_x_mps, dt_s):
        if self._shift_timer_s > 0.0:
            self._shift_timer_s = max(0.0, self._shift_timer_s - dt_s)
        if not self.use_autobox or self._shift_timer_s > 0.0:
            return
        rpm = self._rpm_from_speed(v_x_mps)
        if rpm > self.upshift_rpm and self.gear < len(self.gear_ratios):
            self.gear += 1
            self._shift_timer_s = self.shift_time_s
        elif rpm < self.downshift_rpm and self.gear > 1:
            self.gear -= 1
            self._shift_timer_s = self.shift_time_s

    # -- API principale ----------------------------------------------------
    def compute_force_n(self, throttle, brake, v_x_mps, dt_s, max_brake_force_n):
        """Force longitudinale signée aux roues (N) : >0 traction, <0 freinage."""
        throttle = min(max(throttle, 0.0), 1.0)
        brake = min(max(brake, 0.0), 1.0)

        self._update_gear(v_x_mps, dt_s)

        # rpm moteur : lié aux roues, plancher = ralenti (embrayage qui patine)
        self.rpm = min(max(self._rpm_from_speed(v_x_mps), self.idle_rpm), self.max_rpm)

        if self._shift_timer_s > 0.0:
            engine_torque = 0.0                      # coupure pendant le passage
        else:
            full = self._full_load_torque(self.rpm)
            drag = self.engine_brake_nm_at_max_rpm * (self.rpm / self.max_rpm)
            # throttle=0 -> frein moteur (-drag) ; throttle=1 -> couple plein
            engine_torque = throttle * full - (1.0 - throttle) * drag

        wheel_force = engine_torque * self._overall_ratio() * self.efficiency / self.wheel_radius_m
        # À l'arrêt, pas de frein moteur (évite une force négative qui ferait reculer)
        if v_x_mps < 0.5 and wheel_force < 0.0:
            wheel_force = 0.0

        return wheel_force - brake * max_brake_force_n


class OCDVehicleModel:
    """Modèle dynamique bicyclette (single track, pneu linéaire, direction
    PT1, drivetrain FX) basé sur Open Car Dynamics, avec une couche
    moteur/boîte (EngineGearbox) ajoutée côté wrapper. Calcule sa propre
    trajectoire (x, y, psi) sans simulateur ; pour la visualisation, un
    renderer éventuel consomme simplement ces états."""

    def __init__(self, dynamics_params=None, engine=None,
                 max_brake_force_n=DEFAULT_MAX_BRAKE_FORCE_N):

        if not OCD_AVAILABLE:
            raise RuntimeError(
                "ocd_vehicle_models_py n'est pas installé dans cet environnement. "
                "Buildez-le via ROS 2 (colcon build --packages-up-to "
                "ocd_vehicle_models_py) ou l'image Docker du dépôt "
                "TUMFTM/Open-Car-Dynamics avant d'utiliser cette classe."
            )

        dynamics_params = dynamics_params or {}
        self.max_brake_force_n = max_brake_force_n

        self.vehicle = VehicleFactory.create_vehicle(
            drivetrain=DrivetrainType.FX,
            steering=SteeringActuatorType.PT1,
            vdm=VehicleDynamicsType.SINGLE_TRACK,
            tire=TireType.LINEAR,
            aero=AerodynamicsType.DEFAULT,
        )

        self._apply_dynamics_params(dynamics_params)

        # Rayon de roue : doit être identique côté OCD (conversion N -> Nm)
        # et côté moteur (conversion Nm -> N), sinon la force est faussée.
        wheel_radius = self.vehicle.get_param_manager().get_parameters_as_dict().get(
            "drivetrain.wheel_radius_m", DEFAULT_WHEEL_RADIUS_M
        )
        self.engine = engine or EngineGearbox(wheel_radius_m=wheel_radius)
        self.engine.wheel_radius_m = wheel_radius

        self.external_influences = self.vehicle.construct_external_influences()
        self.steering_input = self.vehicle.construct_steering_input()

        # FX : l'entrée drivetrain est un simple float (N) — inféré du code
        # C++ (DriverInput = double), à confirmer à l'exécution.
        self._fx_input_n = 0.0

        self._ocd_step_s = 0.0008   # 1250 Hz ; un tick à 20 FPS = ~63 sous-pas
        self._v_x_mps = 0.0

    def _apply_dynamics_params(self, dynamics_params):
        param_manager = self.vehicle.get_param_manager()
        param_dict = param_manager.get_parameters_as_dict()

        for our_key, ocd_key in _PARAM_KEY_MAP.items():
            if our_key in dynamics_params and ocd_key in param_dict:
                param_dict[ocd_key] = _coerce(param_dict[ocd_key], dynamics_params[our_key])

        c_alpha = dynamics_params.get("tire_cornering_stiffness", DEFAULT_TIRE_C_ALPHA)
        mass = float(dynamics_params.get("mass_kg", param_dict.get(
            "vehicle_dynamics_single_track.mass_vehicle_kg", 800.0)))
        if "tire_longitudinal_stiffness" in dynamics_params:
            c_sx = dynamics_params["tire_longitudinal_stiffness"]
            if c_sx * mass > 1.25 * MAX_CSX_TIMES_MASS:
                import warnings
                warnings.warn(
                    f"C_sx x masse = {c_sx * mass:.0f} > 15000 : le modèle risque "
                    "de caler vers 6 m/s (instabilité numérique). Réduisez C_sx."
                )
        else:
            c_sx = min(DEFAULT_TIRE_C_SX, MAX_CSX_TIMES_MASS / mass)
        for wheel in _TIRE_WHEELS:
            for name, value in (("C_alpha", c_alpha), ("C_sx", c_sx)):
                key = f"vehicle_dynamics_single_track.z.tire_model.{wheel}.{name}"
                if key in param_dict:
                    param_dict[key] = _coerce(param_dict[key], value)

        param_manager.set_parameters_from_dict(param_dict)

    def step(self, throttle, steer, brake, dt_s, full=False):
        """Avance de dt_s (typiquement 1/20 s). throttle/brake dans [0,1],
        steer dans [-1,1] (convention carla.VehicleControl).
        full=True : renvoie TOUTES les sorties (voir full_output())."""

        n_substeps = max(1, round(dt_s / self._ocd_step_s))

        # 1) couche moteur/boîte -> force signée aux roues (N)
        self._fx_input_n = self.engine.compute_force_n(
            throttle, brake, self._v_x_mps, dt_s, self.max_brake_force_n
        )

        # 2) direction
        max_steer_rad = self.vehicle.get_param_manager().get_parameters_as_dict().get(
            "steering_actuator.angle_max_rad", 0.3
        )
        self.steering_input.steering_angle_rad = steer * max_steer_rad

        # 3) intégration OCD
        for _ in range(n_substeps):
            self.vehicle.set_drivetrain_input(self._fx_input_n)
            self.vehicle.set_external_influences(self.external_influences)
            self.vehicle.set_steering_input(self.steering_input)
            self.vehicle.step()

        dyn = self.vehicle.get_vehicle_model_output().vehicle_dynamics_output
        self._v_x_mps = dyn.velocity_mps.x
        if not all(math.isfinite(v) for v in (dyn.position_m.x, dyn.velocity_mps.x, dyn.velocity_mps.y)):
            raise RuntimeError(
                "OCD a divergé (NaN). Vérifiez les paramètres de pneu "
                "(C_sx / C_alpha) et la force demandée à basse vitesse."
            )

        if full:
            return self.full_output()
        return {
            "x_m": dyn.position_m.x,
            "y_m": dyn.position_m.y,
            "psi_rad": dyn.orientation_rad.z,   # lacet (modèle planaire)
            "v_x_mps": dyn.velocity_mps.x,
            "v_y_mps": dyn.velocity_mps.y,
            "gear": self.engine.gear,
            "engine_rpm": self.engine.rpm,
            "wheel_force_n": self._fx_input_n,
        }


_W = (("fl", "front_left"), ("fr", "front_right"), ("rl", "rear_left"), ("rr", "rear_right"))
_G = 9.81


def _flat_vec(out, prefix, vec, names=("x", "y", "z")):
    for n in names:
        out[f"{prefix}_{n}"] = getattr(vec, n)


def _flat_wheel(out, prefix, dpw):
    for short, attr in _W:
        out[f"{prefix}_{short}"] = getattr(dpw, attr)


def _full_output(self):
    """Toutes les sorties d'OCD (noms vérifiés dans ocd_types_py/_cpp_binding.cpp)
    + celles de la couche moteur. Unités dans le nom (_m, _mps, _rad, _N...)."""
    mo = self.vehicle.get_vehicle_model_output()
    d = mo.vehicle_dynamics_output
    o = {}
    _flat_vec(o, "position_m", d.position_m)
    _flat_vec(o, "velocity_mps", d.velocity_mps)
    _flat_vec(o, "acceleration_mps2", d.acceleration_mps2)
    _flat_vec(o, "orientation_rad", d.orientation_rad)           # z = lacet (psi)
    _flat_vec(o, "angular_velocity_radps", d.angular_velocity_radps)
    _flat_vec(o, "angular_acceleration_radps2", d.angular_acceleration_radps2)
    _flat_wheel(o, "tire_slip_ratio", d.tire_longitudinal_slip_per_wheel)
    _flat_wheel(o, "tire_slip_angle_rad", d.tire_slip_angle_per_wheel_rad)
    _flat_wheel(o, "tire_force_longitudinal_N", d.longitudinal_tire_force_tire_frame_per_wheel_N)
    _flat_wheel(o, "tire_force_lateral_N", d.lateral_tire_force_tire_frame_per_wheel_N)
    _flat_wheel(o, "tire_force_vertical_N", d.vertical_tire_force_per_wheel_N)
    _flat_wheel(o, "wheel_speed_radps", mo.wheel_speeds_radps)
    _flat_wheel(o, "drivetrain_load_torque_Nm", mo.drivetrain_load_torque_per_wheel_Nm)
    _flat_wheel(o, "steering_angle_rad", mo.steering_angle_per_wheel_rad)
    _flat_wheel(o, "steering_load_torque_Nm", mo.steering_load_torque_per_wheel_Nm)
    # Couche moteur/boîte (wrapper)
    o["gear"] = self.engine.gear
    o["engine_rpm"] = self.engine.rpm
    o["wheel_force_command_N"] = self._fx_input_n
    # Dérivés pratiques
    o["speed_kmh"] = d.velocity_mps.x * 3.6
    o["lateral_accel_g"] = d.acceleration_mps2.y / _G
    o["longitudinal_accel_g"] = d.acceleration_mps2.x / _G
    return o


OCDVehicleModel.full_output = _full_output


def ocd_state_to_carla_transform(state, origin_transform):
    """Convertit un état OCD (repère local, origine = spawn) en
    carla.Transform absolu. Utile seulement si un rendu CARLA est branché ;
    la plateforme sans simulateur n'en a pas besoin."""

    import carla  # import local : ce module reste utilisable sans CARLA

    origin_yaw_rad = math.radians(origin_transform.rotation.yaw)

    # OCD : y à gauche, lacet + = gauche.  CARLA : y à droite, yaw + = droite -> on inverse.
    y_right = -state["y_m"]
    world_dx = state["x_m"] * math.cos(origin_yaw_rad) - y_right * math.sin(origin_yaw_rad)
    world_dy = state["x_m"] * math.sin(origin_yaw_rad) + y_right * math.cos(origin_yaw_rad)

    location = carla.Location(
        x=origin_transform.location.x + world_dx,
        y=origin_transform.location.y + world_dy,
        z=origin_transform.location.z,
    )
    rotation = carla.Rotation(
        pitch=origin_transform.rotation.pitch,
        yaw=origin_transform.rotation.yaw - math.degrees(state["psi_rad"]),
        roll=origin_transform.rotation.roll,
    )
    return carla.Transform(location, rotation)
