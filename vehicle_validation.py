"""
vehicle_validation.py — validation d'une configuration de véhicule (format OCDEgo).
Copie fidèle du validateur de la Lambda vecu-vehicle-files et de celui du front ;
utilisée par le service d'aperçu OCD. Pas de dépendance AWS.
"""


class HttpError(Exception):
    def __init__(self, code, msg):
        self.code, self.msg = code, msg


# ───────────────────────── vehicle validation (mirror of the front end) ─────────────────────────
FIELDS = [  # path, min, max, optional
    ("params.mass_kg", 400, 5000, False), ("params.wheelbase_m", 1.5, 4.5, False),
    ("params.cog_distance_from_front_axle_m", 0.3, 4.2, False),
    ("params.moment_of_inertia_z_kgpm2", 200, 15000, False), ("params.drag_coefficient", 0.1, 1.2, False),
    ("params.frontal_area_m2", 0.8, 6, False), ("params.tire_rolling_resistance", 0.004, 0.05, False),
    ("params.wheel_radius_m", 0.2, 0.5, False),
    ("params.tire_cornering_stiffness", 2, 20, True), ("params.tire_longitudinal_stiffness", 1, 30, True),
    ("engine.final_drive", 2, 6, False), ("engine.efficiency", 0.5, 1, False),
    ("engine.idle_rpm", 500, 1500, False), ("engine.max_rpm", 3000, 10000, False),
    ("engine.upshift_rpm", 2000, 10000, False), ("engine.downshift_rpm", 1000, 6000, False),
    ("engine.engine_brake_nm_at_max_rpm", 0, 200, False), ("brake_n", 2000, 30000, False),
]
ALLOWED_TOP = {"params", "engine", "brake_n"}
ALLOWED_PARAMS = {p.split(".")[1] for p, *_ in FIELDS if p.startswith("params.")}
ALLOWED_ENGINE = {p.split(".")[1] for p, *_ in FIELDS if p.startswith("engine.")} | {"gear_ratios", "torque_curve"}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def validate_vehicle_config(c):
    """Returns a list of error strings (empty = valid)."""
    if not isinstance(c, dict):
        return ["configuration must be an object"]
    errs = []
    if set(c) - ALLOWED_TOP: errs.append(f"unknown keys: {sorted(set(c) - ALLOWED_TOP)}")
    if not isinstance(c.get("params"), dict) or not isinstance(c.get("engine"), dict):
        return errs + ["params and engine are required"]
    if set(c["params"]) - ALLOWED_PARAMS: errs.append(f"unknown params: {sorted(set(c['params']) - ALLOWED_PARAMS)}")
    if set(c["engine"]) - ALLOWED_ENGINE: errs.append(f"unknown engine keys: {sorted(set(c['engine']) - ALLOWED_ENGINE)}")
    for p, lo, hi, opt in FIELDS:
        v = _get(c, p)
        if v is None:
            if not opt: errs.append(f"{p}: required")
            continue
        if not _num(v) or not (lo <= v <= hi): errs.append(f"{p}: must be a number in [{lo}, {hi}]")
    if errs: return errs
    wb, cog = c["params"]["wheelbase_m"], c["params"]["cog_distance_from_front_axle_m"]
    if not (0.2 <= cog <= wb - 0.2): errs.append("cog_distance_from_front_axle_m must lie inside the wheelbase")
    e = c["engine"]
    if not e["idle_rpm"] < e["downshift_rpm"] < e["upshift_rpm"] < e["max_rpm"]:
        errs.append("require idle_rpm < downshift_rpm < upshift_rpm < max_rpm")
    g = e.get("gear_ratios")
    if not (isinstance(g, list) and 1 <= len(g) <= 8 and all(_num(x) and 0.3 <= x <= 6 for x in g)
            and all(g[i] < g[i - 1] for i in range(1, len(g)))):
        errs.append("gear_ratios: 1-8 numbers in [0.3, 6], strictly decreasing")
    t = e.get("torque_curve")
    if not (isinstance(t, list) and 3 <= len(t) <= 20 and all(isinstance(p, list) and len(p) == 2 and _num(p[0]) and _num(p[1])
            and p[0] >= 0 and 0 <= p[1] <= 1500 for p in t) and all(t[i][0] > t[i - 1][0] for i in range(1, len(t)))):
        errs.append("torque_curve: 3-20 [rpm, Nm] points, rpm strictly increasing, torque in [0, 1500]")
    return errs


def validate_vehicle_payload(v):
    if not isinstance(v, dict): raise HttpError(400, "vehicle must be an object")
    if v.get("enabled") is False: return {"enabled": False}
    if v.get("mode") not in ("preset", "custom"): raise HttpError(400, "mode must be 'preset' or 'custom'")
    errs = validate_vehicle_config(v.get("ocd_vehicle"))
    if errs: raise HttpError(400, "invalid vehicle: " + "; ".join(errs))
    name = str(v.get("name") or "Vehicle").strip()[:40]
    preset = v.get("preset") if v["mode"] == "preset" and v.get("preset") in ("citadine", "berline", "suv") else None
    return {"enabled": True, "mode": v["mode"], "name": name, "preset": preset, "ocd_vehicle": v["ocd_vehicle"]}


