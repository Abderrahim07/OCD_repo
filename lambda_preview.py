"""
lambda_preview.py — Lambda (image conteneur) de la route POST /preview.
Handler : lambda_preview.handler
L'image contient OCD (ocd_vehicle_models_py) + vehicle_preview.py + vehicle_validation.py + ocd_ego.py + ocd_vehicle_dynamics.py.
Aucune permission AWS particulière (ne lit ni S3 ni DynamoDB).
"""
import base64, json, traceback
from vehicle_validation import validate_vehicle_config
from vehicle_preview import preview, profile, response, PRESETS

MAX_BODY = 400_000
CORS = {"Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "POST,OPTIONS"}


def _resp(code, obj):
    return {"statusCode": code, "headers": {"Content-Type": "application/json", **CORS}, "body": json.dumps(obj)}


def resolve_vehicle(v):
    if isinstance(v, str):
        if v.lower() not in PRESETS: raise ValueError(f"unknown vehicle: {v}")
        return v.lower()
    if isinstance(v, dict):
        if v.get("enabled") is False: raise ValueError("dynamics model disabled")
        cfg = v.get("ocd_vehicle", v)
        if isinstance(cfg, str): return resolve_vehicle(cfg)
        errs = validate_vehicle_config(cfg)
        if errs: raise ValueError("invalid vehicle: " + "; ".join(errs))
        return cfg
    raise ValueError("missing vehicle")


def _direct(event):
    """Appel direct (Lambda /run -> boto3 invoke), sans API Gateway : { "action": "profile", vehicle, test_case, overrides }."""
    veh = resolve_vehicle(event.get("vehicle"))
    tc = event.get("test_case") or {}
    ov = event.get("overrides") or {}
    return profile(veh, tc, ov)


def _response(body):
    """Réponse du véhicule à la demande de braquage du firmware : { vehicle, t:[], steer_rad:[], speed_kmh, steer_ratio? }."""
    veh = resolve_vehicle(body.get("vehicle"))
    t, st = body.get("t"), body.get("steer_rad")
    if not isinstance(t, list) or not isinstance(st, list): raise ValueError("t and steer_rad must be arrays")
    t = [float(x) for x in t]; st = [float(x) for x in st]
    return response(veh, t, st, float(body.get("speed_kmh")), steer_ratio=float(body.get("steer_ratio", 1.0)))


def handler(event, context):
    if event.get("action") == "profile":          # appel interne (pas de HTTP)
        try:
            return {"ok": True, "profile": _direct(event)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if event.get("action") == "response":         # appel interne : réponse du véhicule à SteeringRequest_rad
        try:
            return {"ok": True, "response": _response(event)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if (event.get("httpMethod") or "").upper() == "OPTIONS":
        return _resp(204, {})
    try:
        raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"): raw = base64.b64decode(raw).decode()
        if len(raw) > MAX_BODY: return _resp(413, {"error": "body too large"})
        body = json.loads(raw)
        if body.get("mode") == "response":
            return _resp(200, _response(body))
        veh = resolve_vehicle(body.get("vehicle"))
        tc = body.get("test_case") or {}
        ov = body.get("overrides") or {}
        for k in ("drift_start_s", "lateral_rate_mps", "speed_kmh"):
            if ov.get(k) is not None and not (0 <= float(ov[k]) <= 400): raise ValueError(f"{k} out of range")
        if ov.get("lateral_rate_mps") is not None and not (0.05 <= float(ov["lateral_rate_mps"]) <= 3):
            raise ValueError("lateral_rate_mps: 0.05 to 3 m/s")
        return _resp(200, preview(veh, tc, ov))
    except (ValueError, KeyError, TypeError) as e:
        return _resp(400, {"error": str(e)})
    except Exception as e:
        traceback.print_exc()
        return _resp(500, {"error": f"{type(e).__name__}: {e}"})
