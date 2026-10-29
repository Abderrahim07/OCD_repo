"""
vehicle_preview.py — APERÇU du comportement d'un véhicule (OCD) pour un test case.

N'intervient JAMAIS dans l'exécution du test : on lit le test case, on déduit la
manœuvre (vitesse, côté de la dérive), puis OCD calcule la trajectoire du véhicule
choisi. Le résultat est une PRÉVISION du modèle, indépendante de la vidéo.

À exécuter là où OCD est installé (WSL / conteneur Linux).

    from vehicle_preview import preview
    out = preview("berline", test_case_dict, overrides={"drift_start_s": 3})
"""
import math
from ocd_ego import OCDEgo, PRESETS

DEFAULTS = dict(drift_start_s=3.0, lateral_rate_mps=0.4, lane_width_m=3.5, vehicle_width_m=1.8, steering_deg=0.15)
CAP_BEYOND_LINE_M = 0.6      # la dérive continue 0,6 m au-delà de la ligne
MIN_SPEED_KMH = 5.0


def _truthy(v):
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on", "1.0")
    return bool(v) and v != 0


def _split(tc):
    """Retourne (inputs, expected, description) pour les deux formats :
    JSON structuré {inputs, expected_outputs, description} ou vecteur plat
    {scenario, <signal>, expected_<signal>}."""
    desc = str(tc.get("description") or tc.get("_description") or tc.get("rationale") or "")
    if isinstance(tc.get("inputs"), dict):
        return tc["inputs"], tc.get("expected_outputs") or {}, desc
    meta = {"scenario", "id", "description", "rationale", "method", "type", "requirement", "st_id"}
    inputs = {k: v for k, v in tc.items() if not k.startswith("expected_") and not k.startswith("_") and k not in meta}
    expected = {k[9:]: v for k, v in tc.items() if k.startswith("expected_")}
    return inputs, expected, desc


def _find(d, *needles):
    for k, v in d.items():
        kl = k.lower()
        if all(n in kl for n in needles):
            return k, v
    return None, None


def parse_test_case(tc, overrides=None):
    """Déduit la manœuvre. Chaque valeur est accompagnée de sa source."""
    o = overrides or {}
    inputs, expected, desc = _split(tc or {})
    m, src, warn = {}, {}, []

    # vitesse
    if o.get("speed_kmh") is not None:
        m["speed_kmh"], src["speed_kmh"] = float(o["speed_kmh"]), "manual setting"
    else:
        k, v = _find(inputs, "speed")
        if k is not None:
            try:
                m["speed_kmh"], src["speed_kmh"] = float(v), f"test case: inputs.{k}"
            except (TypeError, ValueError):
                k = None
        if k is None:
            m["speed_kmh"], src["speed_kmh"] = 50.0, "default (no speed in the test case)"
            warn.append("No speed found in the test case: 50 km/h used by default.")

    # direction de la dérive
    if o.get("direction") in ("left", "right", "none"):
        m["direction"], src["direction"] = o["direction"], "manual setting"
    else:
        kr, vr = _find(expected, "warning", "right")
        kl, vl = _find(expected, "warning", "left")
        r, l = (_truthy(vr) if kr else False), (_truthy(vl) if kl else False)
        if r and not l:
            m["direction"], src["direction"] = "right", f"test case: expected_outputs.{kr} is true"
        elif l and not r:
            m["direction"], src["direction"] = "left", f"test case: expected_outputs.{kl} is true"
        elif r and l:
            m["direction"], src["direction"] = "none", "ambiguous"
            warn.append("Warning_Left and Warning_Right both expected: ambiguous direction, straight road simulated.")
        else:
            d = desc.lower()
            drift = any(w in d for w in ("drift", "depart", "deviat", "cross", "dévi", "sort"))
            if drift and "right" in d and "left" not in d:
                m["direction"], src["direction"] = "right", "test case description"
            elif drift and "left" in d and "right" not in d:
                m["direction"], src["direction"] = "left", "test case description"
            else:
                m["direction"], src["direction"] = "none", "no side warning expected: straight road"

    # signaux informatifs
    for side in ("left", "right"):
        k, v = _find(inputs, "turn", side)
        m["turn_signal_" + side] = _truthy(v) if k else False

    # paramètres de manœuvre (défauts modifiables)
    for key in ("drift_start_s", "lateral_rate_mps", "lane_width_m", "vehicle_width_m"):
        if o.get(key) is not None:
            m[key], src[key] = float(o[key]), "manual setting"
        else:
            m[key], src[key] = DEFAULTS[key], "default"
    # mode : "path" = dérive imposée à vitesse latérale constante ;
    #        "steering" (défaut) = petit angle de volant constant (la dérive dépend alors du véhicule)
    m["mode"] = o.get("mode") if o.get("mode") in ("path", "steering") else "steering"
    src["mode"] = "manual setting" if o.get("mode") else "default"
    if o.get("steering_deg") is not None:
        m["steering_deg"], src["steering_deg"] = float(o["steering_deg"]), "manual setting"
    else:
        m["steering_deg"], src["steering_deg"] = DEFAULTS["steering_deg"], "default"
    m["sources"] = src
    m["_warnings"] = warn
    return m


def _lane_edge(m):
    """Distance latérale (centre véhicule -> axe) à partir de laquelle une roue touche la ligne."""
    return m["lane_width_m"] / 2.0 - m["vehicle_width_m"] / 2.0


def simulate(vehicle, man, fps=20):
    cfg = PRESETS[vehicle.lower()] if isinstance(vehicle, str) else vehicle
    speed = max(float(man["speed_kmh"]), MIN_SPEED_KMH)
    ego = OCDEgo((0.0, 0.0, 0.0), cfg, fps)
    dt = 1.0 / fps
    warnings = list(man.get("_warnings", []))

    # 1) mise en vitesse (mêmes lois que le preroll de OCDEgo, mais enregistrées)
    acc_t, acc_v, t = [0.0], [0.0], 0.0
    for _ in range(int(90.0 * fps)):
        if abs(3.6 * ego._state["v_x_mps"] - speed) < 0.2:
            break
        th, br = ego._speed_cmd(speed)
        ego._state = ego.model.step(th, 0.0, br, dt)
        t += dt
        acc_t.append(round(t, 3)); acc_v.append(round(3.6 * ego._state["v_x_mps"], 2))
    s0 = ego._state
    ego._ref = (s0["x_m"], s0["y_m"], s0["psi_rad"])
    v_start = 3.6 * s0["v_x_mps"]
    if abs(v_start - speed) > 3.0:
        warnings.append(f"The vehicle only reaches {v_start:.0f} km/h (target {speed:.0f} km/h): "
                        "insufficient power for this speed.")

    # 2) trajectoire cible : axe de voie puis dérive latérale à vitesse constante
    sign = {"right": 1.0, "left": -1.0, "none": 0.0}[man["direction"]]       # +y = droite (repère CARLA)
    edge = _lane_edge(man)
    cap = man["lane_width_m"] / 2.0 + CAP_BEYOND_LINE_M
    rate = max(man["lateral_rate_mps"], 1e-3)
    start = man["drift_start_s"]
    dur = 8.0 if sign == 0 else min(40.0, start + cap / rate + 2.0)
    if man.get("mode") == "steering" and sign != 0: dur = min(40.0, start + 14.0)
    n = int(dur * fps) + 1
    ds = speed / 3.6 / fps
    path = []
    for i in range(n + int(8 * fps)):                      # + marge pour la distance de visée
        ti = i / fps
        off = 0.0 if ti < start else min(cap, rate * (ti - start))
        path.append((i * ds, sign * off))

    # 3) simulation OCD
    keys = ("t", "lat_m", "v_kmh", "yaw_deg", "steer_deg", "lat_acc_g", "gear", "rpm", "dist_to_line_m")
    S = {k: [] for k in keys}
    cross_t, cross_v = None, None
    steer_cmd = 0.0
    if man.get("mode") == "steering" and sign != 0:
        max_steer_deg = math.degrees(0.3)                  # angle_max_rad du PT1 OCD
        steer_cmd = -sign * max(-1.0, min(1.0, man["steering_deg"] / max_steer_deg))   # OCD : + = gauche
    for i in range(n):
        if steer_cmd and i / fps >= start:
            th, br = ego._speed_cmd(speed)
            ego._state = ego.model.step(th, steer_cmd, br, dt)
        else:
            ego.drive(path, speed)
        x, y, yaw = ego.pose()
        full = ego.model.full_output()
        dist = edge - abs(y) if sign == 0 else edge - sign * y        # <0 : une roue a franchi la ligne
        ti = i / fps
        S["t"].append(round(ti, 3)); S["lat_m"].append(round(y, 4)); S["v_kmh"].append(round(ego.speed_kmh, 2))
        S["yaw_deg"].append(round(yaw, 3))
        S["steer_deg"].append(round(math.degrees(full.get("steering_angle_rad_fl", 0.0)), 3))
        S["lat_acc_g"].append(round(full.get("lateral_accel_g", 0.0), 4))
        S["gear"].append(int(ego.state.get("gear", 0))); S["rpm"].append(round(ego.state.get("engine_rpm", 0.0)))
        S["dist_to_line_m"].append(round(dist, 4))
        if cross_t is None and dist < 0:
            cross_t, cross_v = ti, ego.speed_kmh
    if sign != 0 and cross_t is None:
        warnings.append("The vehicle never reaches the line during the simulation (drift too small or unstable vehicle).")
    return dict(
        series=S,
        accel_phase=dict(t=acc_t, v_kmh=acc_v),
        events=dict(
            crossing_time_s=None if cross_t is None else round(cross_t, 2),
            crossing_side=man["direction"] if cross_t is not None else None,
            crossing_speed_kmh=None if cross_v is None else round(cross_v, 1),
            time_from_drift_start_s=None if cross_t is None else round(cross_t - start, 2),
            max_lateral_g=round(max((abs(v) for v in S["lat_acc_g"]), default=0.0), 3),
            max_abs_lateral_m=round(max((abs(v) for v in S["lat_m"]), default=0.0), 3),
        ),
        geometry=dict(lane_width_m=man["lane_width_m"], vehicle_width_m=man["vehicle_width_m"], crossing_offset_m=round(edge, 3)),
        warnings=warnings,
    )


def preview(vehicle, test_case, overrides=None, fps=20):
    man = parse_test_case(test_case, overrides)
    res = simulate(vehicle, man, fps)
    man_out = {k: v for k, v in man.items() if not k.startswith("_")}
    res["maneuver"] = man_out
    res["disclaimer"] = ("Forecast of the OCD dynamics model for the chosen vehicle. "
                         "Independent of the video: this is not the test result.")
    return res


def profile(vehicle, test_case, overrides=None, fps=20):
    """
    Profil temporel du véhicule pour l'EXÉCUTION : ce que le firmware reçoit frame par frame.
    Départ déjà à la vitesse du test case ; OCD simule ensuite la manœuvre.
    Retourne des séries compactes (t, speed_kmh, ...) + le résumé de la manœuvre.
    """
    man = parse_test_case(test_case, overrides)
    res = simulate(vehicle, man, fps)
    S = res["series"]
    return dict(
        fps=fps,
        t=S["t"], speed_kmh=S["v_kmh"], lat_m=S["lat_m"], yaw_deg=S["yaw_deg"],
        steer_deg=S["steer_deg"], lat_acc_g=S["lat_acc_g"],
        events=res["events"], warnings=res["warnings"],
        maneuver={k: v for k, v in man.items() if not k.startswith("_")},
    )


# ──────────────────────────────────────────────────────────────────────────
# RÉPONSE DU VÉHICULE À LA DEMANDE DE BRAQUAGE DU FIRMWARE (boucle ouverte)
# ──────────────────────────────────────────────────────────────────────────
MAX_STEER_RAD = 0.3          # angle_max_rad du PT1 de direction OCD : entrée normalisée 1.0 = 0,3 rad
MAX_RESPONSE_POINTS = 2000


def response(vehicle, t, steer_rad, speed_kmh, fps=20, steer_ratio=1.0):
    """
    Que ferait le véhicule si la demande de braquage du firmware (SteeringRequest_rad, frame par frame)
    lui était appliquée ?   Boucle OUVERTE : la vidéo, l'écart mesuré par le firmware et son verdict
    ne changent pas ; on regarde seulement la réponse du modèle (paramètres).

    t, steer_rad : séries frame par frame issues de l'exécution (t en s, demande en rad, + = gauche)
    speed_kmh    : vitesse de l'essai (le véhicule est déjà à cette vitesse à t = 0, régulée ensuite)
    steer_ratio  : rad d'angle de roue pour 1 rad demandé (1.0 = la demande est un angle de roue ;
                   1/16 ≈ la demande est un angle au volant)
    Convention de sortie : latéral et lacet POSITIFS VERS LA GAUCHE (repère ISO d'OCD).
    """
    import math
    n = len(steer_rad)
    if n < 2 or n != len(t):
        raise ValueError("t and steer_rad must have the same length (>= 2)")
    if n > MAX_RESPONSE_POINTS:
        raise ValueError(f"too many points (max {MAX_RESPONSE_POINTS})")
    if not (5.0 <= float(speed_kmh) <= 400.0):
        raise ValueError("speed_kmh: 5 to 400")
    if not (0.001 <= abs(float(steer_ratio)) <= 10.0):
        raise ValueError("steer_ratio out of range")
    speed = float(speed_kmh)
    cfg = PRESETS[vehicle.lower()] if isinstance(vehicle, str) else vehicle
    ego = OCDEgo((0.0, 0.0, 0.0), cfg, fps)
    dt = 1.0 / fps
    warnings = []

    # mise en vitesse (identique à simulate) — non affichée, t = 0 quand le véhicule est à la vitesse de l'essai
    for _ in range(int(90.0 * fps)):
        if abs(3.6 * ego._state["v_x_mps"] - speed) < 0.2:
            break
        th, br = ego._speed_cmd(speed)
        ego._state = ego.model.step(th, 0.0, br, dt)
    s0 = ego._state
    ego._ref = (s0["x_m"], s0["y_m"], s0["psi_rad"])
    v_start = 3.6 * s0["v_x_mps"]
    if abs(v_start - speed) > 3.0:
        warnings.append(f"The vehicle only reaches {v_start:.0f} km/h (target {speed:.0f} km/h): insufficient power for this speed.")

    keys = ("t", "steer_req_deg", "steer_deg", "lat_m", "yaw_deg", "lat_acc_g", "v_kmh")
    S = {k: [] for k in keys}
    saturated = 0
    for i in range(n):
        wheel_rad = float(steer_rad[i]) * float(steer_ratio)
        norm = wheel_rad / MAX_STEER_RAD
        if abs(norm) > 1.0:
            saturated += 1
            norm = max(-1.0, min(1.0, norm))
        th, br = ego._speed_cmd(speed)
        ego._state = ego.model.step(th, norm, br, dt)          # OCD : + = gauche
        x, y, yaw = ego.pose()                                   # repère CARLA (y, lacet positifs vers la droite)
        full = ego.model.full_output()
        S["t"].append(round(float(t[i]), 3))
        S["steer_req_deg"].append(round(math.degrees(wheel_rad), 3))
        S["steer_deg"].append(round(math.degrees(full.get("steering_angle_rad_fl", 0.0)), 3))
        S["lat_m"].append(round(-y, 4) + 0.0)
        S["yaw_deg"].append(round(-yaw, 3) + 0.0)
        S["lat_acc_g"].append(round(full.get("lateral_accel_g", 0.0), 4))
        S["v_kmh"].append(round(ego.speed_kmh, 2))

    active = sum(1 for v in steer_rad if abs(float(v)) > 1e-9)
    if active == 0:
        warnings.append("The firmware never requested steering in this scenario (SteeringRequest_rad = 0 on every frame): the vehicle drives straight.")
    if saturated:
        warnings.append(f"{saturated} frame(s) exceed the steering actuator limit ({MAX_STEER_RAD} rad): the request was clipped.")
    ev = dict(
        steering_active_s=round(active * dt, 2),
        peak_request_deg=round(max((abs(v) for v in S["steer_req_deg"]), default=0.0), 3),
        max_lateral_m=round(max((abs(v) for v in S["lat_m"]), default=0.0), 3),
        final_lateral_m=S["lat_m"][-1],
        max_yaw_deg=round(max((abs(v) for v in S["yaw_deg"]), default=0.0), 3),
        max_lateral_g=round(max((abs(v) for v in S["lat_acc_g"]), default=0.0), 3),
    )
    return dict(series=S, events=ev, warnings=warnings,
                inputs=dict(speed_kmh=speed, steer_ratio=float(steer_ratio), fps=fps, frames=n),
                disclaimer="Open loop: the camera image and the firmware's measured offset do not change. "
                           "This is what the vehicle model would do if the firmware's steering request were applied. Lateral and yaw: + = left.")
