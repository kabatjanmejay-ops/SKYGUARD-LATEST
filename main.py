"""
SkyGuard AI — main.py: FastAPI app, all route handlers.

Lives at repo root alongside config.py/data_fetch.py (see the actual
directory structure -- model/, data/, model_artifacts/ are subpackages;
this file and config.py are the two root-level pieces that tie them
together). Wires simulator.py's SimulatorState into the exact 9
endpoints the frontend is already built against
(DEVELOPMENT_PROGRESS.md's "Approved Contract Endpoints" list) -- no
endpoint here was invented; every route matches FRONTEND_ARCHITECTURE.md
exactly, including reusing POST /api/inject-anomaly as the
replay-mode trigger (see simulator.py's start_replay() docstring for
why, and the earlier confirmation flag on that design choice).
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio
import pandas as pd
import json
from datetime import datetime

load_dotenv()

sys.path.append(str(Path(__file__).parent))
from model.simulator import create_simulator_state, run_simulation_loop

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # basic string conversion for payload
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
            
        payload = json.dumps(message, default=json_serial)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception:
                self.disconnect(connection)

ws_manager = ConnectionManager()

async def ws_on_tick(latest_data):
    await ws_manager.broadcast({"type": "TICK", "data": "update"})


def _json_nullable(value):
    """Convert CSV/pandas NaN values to valid JSON nulls for API payloads."""
    if value is None:
        return None
    try:
        return None if bool(pd.isna(value)) else value
    except (TypeError, ValueError):
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sim = create_simulator_state()
    app.state.sim.on_tick_callbacks.append(ws_on_tick)
    # Fetch/ingest the first live snapshot before HTTP routes become
    # available. This prevents normal dashboard startup from racing the
    # first Open-Meteo request and receiving misleading 404 responses.
    await app.state.sim.tick()
    app.state.sim_task = asyncio.create_task(run_simulation_loop(app.state.sim))
    try:
        yield
    finally:
        app.state.sim_task.cancel()
        try:
            await app.state.sim_task
        except asyncio.CancelledError:
            pass
        await app.state.sim.close()


app = FastAPI(title="SkyGuard AI", lifespan=lifespan)

# allow_origins=["*"] -- fine for hackathon per BACKEND_BLUEPRINT.md
# section 6; tighten to the deployed frontend URL once known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.get("/api/system-status")
def get_system_status():
    """Small control-plane endpoint: frontend cadence follows backend mode."""
    sim = app.state.sim
    return {
        "mode": sim.mode,
        "replay_step_seconds": 2 if sim.mode == "replay" else None,
        "live_poll_interval_seconds": 30 * 60,
    }


@app.post("/api/system-mode")
def set_system_mode(body: dict):
    """Switch safely back to live when the dashboard replay toggle is off."""
    if body.get("mode") != "live":
        raise HTTPException(status_code=400, detail="Only mode='live' is supported by this endpoint.")
    app.state.sim.stop_replay()
    return {"mode": "live", "message": "Replay stopped; live buffers and view were reset."}


@app.get("/api/network-status")
def get_network_status():
    """Aggregate station health for the header badge — not a static demo label."""
    from datetime import datetime, timezone

    sim = app.state.sim
    statuses = []
    health_pcts = []
    for sid in sim.manager.buffers:
        station_health = sim.manager.get_station_status(sid)
        mapped = {"HEALTHY": "NORMAL", "WARNING": "WARNING", "OFFLINE": "OFFLINE"}.get(
            station_health["status"], "NORMAL"
        )
        statuses.append(mapped)
        health_pcts.append(_health_pct(sim.manager.buffers[sid].health.param_status))

    if "CRITICAL" in statuses:
        overall = "CRITICAL"
    elif "WARNING" in statuses or "OFFLINE" in statuses:
        overall = "WARNING"
    else:
        overall = "NORMAL"

    return {
        "overall_status": overall,
        "active_stations_count": sum(1 for status in statuses if status != "OFFLINE"),
        "total_stations_count": len(statuses),
        "active_anomalies_count": len(sim.recent_anomalies),
        "avg_sensor_health_pct": round(sum(health_pcts) / len(health_pcts)) if health_pcts else 100,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "mode": sim.mode,
    }


@app.post("/api/refresh-live")
async def refresh_live_snapshot():
    """Explicit, user-triggered Open-Meteo refresh; does not alter cadence."""
    sim = app.state.sim
    await sim.refresh_live_now()
    return {"mode": sim.mode, "message": "Live provider snapshot refreshed."}


# ---------------- GET /api/stations ----------------

@app.get("/api/stations")
def get_stations():
    sim = app.state.sim
    result = []
    for _, row in sim.metadata.iterrows():
        sid = row["station_id"]
        status = sim.manager.get_station_status(sid)["status"]
        # Contract wants NORMAL/WARNING/CRITICAL/OFFLINE, not health's
        # own HEALTHY/WARNING/OFFLINE vocabulary -- translate.
        mapped_status = {"HEALTHY": "NORMAL", "WARNING": "WARNING", "OFFLINE": "OFFLINE"}.get(status, "NORMAL")
        result.append({
            "station_id": sid,
            "name": row["name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "status": mapped_status,
        })
    return result


# ---------------- GET /api/current-reading ----------------

@app.get("/api/current-reading")
def get_current_reading(station_id: str):
    sim = app.state.sim
    entry = sim.latest.get(station_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No reading yet for {station_id}")

    raw = entry["raw_reading"]
    verdict = entry["verdict"]

    # normal_min/max: static fallback ranges since config.py's exact
    # constant names aren't in view here -- if config.py already
    # defines per-parameter normal ranges, swap these literals for
    # that import instead of duplicating the values.
    station_health = sim.manager.get_station_status(station_id)
    parameter_status = sim.manager.buffers[station_id].health.param_status
    return {
        "station_id": station_id,
        "timestamp": entry["timestamp"].isoformat(),
        "temperature_c": {"value": raw.get("temperature_c"), "normal_min": 10.0, "normal_max": 45.0},
        "pressure_hpa": {"value": raw.get("pressure_hpa"), "normal_min": 950.0, "normal_max": 1050.0},
        "humidity_pct": {"value": raw.get("humidity_pct"), "normal_min": 10.0, "normal_max": 100.0},
        "anomaly_score_pct": verdict["anomaly_score_pct"],
        "is_anomaly": bool(verdict.get("is_anomaly", False)),
        "fault_type": verdict.get("fault_type"),
        "severity": verdict.get("severity"),
        "model_confidence_pct": verdict.get("model_confidence_pct"),
        "rule_confidence_pct": verdict.get("rule_confidence_pct"),
        "risk_level": verdict["severity"],
        "sensor_health_pct": _health_pct(parameter_status),
        "sensor_health_status": station_health["status"],
        "sensor_parameters": parameter_status,
        "suggested_values": verdict.get("suggested_values", {}),
        "source": sim.manager.mode,
    }


# ---------------- GET /api/trends ----------------

@app.get("/api/trends")
def get_trends(station_id: str, hours: int = 6):
    sim = app.state.sim
    if station_id not in sim.manager.buffers:
        raise HTTPException(status_code=404, detail=f"Unknown station {station_id}")
    if not 1 <= hours <= 24 * 30:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 720")

    # HistoryStore is the durable frontend source. It preserves raw
    # values, fault labels, suggested values, and status at the time of
    # every reading; the simulator deque is only a short UI cache.
    points = sim.manager.get_station_history(station_id, hours=hours)
    if not points:
        points = list(sim.trend_history[station_id])

    trend_points = [
        {
            "timestamp": p["timestamp"].isoformat(),
            "temperature_c": _json_nullable(p["temperature_c"]),
            "pressure_hpa": _json_nullable(p["pressure_hpa"]),
            "humidity_pct": _json_nullable(p["humidity_pct"]),
            "is_anomaly": bool(p.get("is_anomaly", False)),
            "fault_type": _json_nullable(p.get("fault_type")),
            "severity": _json_nullable(p.get("severity")),
            "anomaly_score_pct": _json_nullable(p.get("anomaly_score_pct")),
            "suggested_temperature_c": _json_nullable(p.get("suggested_temperature_c")),
            "suggested_pressure_hpa": _json_nullable(p.get("suggested_pressure_hpa")),
            "suggested_humidity_pct": _json_nullable(p.get("suggested_humidity_pct")),
            "health_status": _json_nullable(p.get("health_status")),
            "source": p.get("source", sim.manager.mode),
        }
        for p in points
    ]

    anomaly_windows = []
    in_window = False
    for p in points:
        if p["is_anomaly"] and not in_window:
            window_start = p["timestamp"]
            in_window = True
        elif not p["is_anomaly"] and in_window:
            anomaly_windows.append({
                "start": window_start.isoformat(),
                "end": p["timestamp"].isoformat(),
                "label": "Anomaly Detected",
            })
            in_window = False
    if in_window:
        anomaly_windows.append({
            "start": window_start.isoformat(),
            "end": points[-1]["timestamp"].isoformat(),
            "label": "Anomaly Detected",
        })

    return {"station_id": station_id, "hours": hours, "points": trend_points, "anomaly_windows": anomaly_windows}


@app.get("/api/history.csv")
def download_station_history(station_id: str):
    """Operator export of the complete retained (up to 30-day) station CSV."""
    sim = app.state.sim
    if station_id not in sim.manager.buffers:
        raise HTTPException(status_code=404, detail=f"Unknown station {station_id}")
    df = sim.manager.history.get_all(station_id)
    csv_text = df.sort_values("timestamp").to_csv(index=False) if not df.empty else \
        "timestamp,station_id,temperature_c,pressure_hpa,humidity_pct,is_anomaly,fault_type,severity,anomaly_score_pct,suggested_temperature_c,suggested_pressure_hpa,suggested_humidity_pct,health_status,source\n"
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{station_id}_history.csv"'},
    )


# ---------------- GET /api/anomalies/latest ----------------

@app.get("/api/anomalies/latest")
def get_latest_anomaly(station_id: str):
    sim = app.state.sim
    for a in sim.recent_anomalies:
        if a["station_id"] == station_id:
            return {
                "anomaly_id": a["anomaly_id"],
                "timestamp": a["timestamp"].isoformat(),
                "station_id": a["station_id"],
                "anomaly_score_pct": a["anomaly_score_pct"],
                "severity": a["severity"],
                "type": a["type"],
                "root_cause": a["root_cause"],
                "description": f"{a['root_cause']} detected at {a['station_id']}.",
                "suggested_values": a.get("suggested_values"),
                "observed_values": a.get("observed_values"),
                "affected_parameters": a.get("affected_parameters", []),
            }
    # No recent anomaly is a healthy, expected state—not a missing resource.
    # Returning JSON null keeps the dashboard nominal and avoids a noisy 404
    # in the browser console for stations without an incident.
    return None


# ---------------- GET /api/anomalies/recent ----------------

@app.get("/api/anomalies/recent")
def get_recent_anomalies(station_id: str, limit: int = 5):
    sim = app.state.sim
    matches = [a for a in sim.recent_anomalies if a["station_id"] == station_id][:limit]
    return [
        {
            "anomaly_id": a["anomaly_id"],
            "type": a["type"],
            "label": a["root_cause"],
            "station_id": a["station_id"],
            "timestamp": a["timestamp"].isoformat(),
            "score_pct": a["anomaly_score_pct"],
            "severity": a["severity"],
            "suggested_values": a.get("suggested_values"),
            "observed_values": a.get("observed_values"),
            "affected_parameters": a.get("affected_parameters", []),
        }
        for a in matches
    ]


# ---------------- GET /api/explain/{anomaly_id} ----------------

@app.get("/api/explain/{anomaly_id}")
def get_explanation(anomaly_id: str):
    sim = app.state.sim

    match = next(
        (a for a in sim.recent_anomalies if a["anomaly_id"] == anomaly_id),
        None
    )

    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown anomaly_id {anomaly_id}"
        )

    return {
        "anomaly_id": anomaly_id,
        "features": match.get("shap_features", []),
        "likely_faulty_sensors": match.get("likely_faulty_sensors", []),
        "affected_parameters": match.get("affected_parameters", []),
        "observed_values": match.get("observed_values", {}),
        "suggested_values": match.get("suggested_values", {}),
        "model_confidence_pct": match.get("model_confidence_pct"),
        "rule_confidence_pct": match.get("rule_confidence_pct"),
        "anomaly_score_pct": match.get("anomaly_score_pct"),
        "fault_type": match.get("type"),
    }



# ---------------- GET /api/sensor-health ----------------

@app.get("/api/sensor-health")
def get_sensor_health(station_id: str):
    sim = app.state.sim
    if station_id not in sim.manager.buffers:
        raise HTTPException(status_code=404, detail=f"Unknown station {station_id}")
    status = sim.manager.get_station_status(station_id)
    mapped = {"HEALTHY": "HEALTHY", "WARNING": "WARNING", "OFFLINE": "OFFLINE"}.get(status["status"], "HEALTHY")
    parameter_status = sim.manager.buffers[station_id].health.param_status
    return {
        "station_id": station_id,
        "health_pct": _health_pct(parameter_status),
        "status": mapped,
        "parameters": parameter_status,
        "offline_reason": status["offline_reason"],
        "recovery_active": status["recovery_active"],
    }

# ---------------- POST /api/repair-sensor ----------------

@app.post("/api/repair-sensor")
def repair_sensor(body: dict):
    sim = app.state.sim

    station_id = body.get("station_id")
    if not station_id:
        raise HTTPException(
            status_code=400,
            detail="station_id is required"
        )

    if station_id not in sim.manager.buffers:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown station {station_id}"
        )

    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc)

    if body.get("force_recovery", False):
        sim.manager.force_recover_station(station_id)
        return {
            "success": True,
            "station_id": station_id,
            "status": "HEALTHY",
            "recovery_active": False,
            "message": "Sensor force-recovered and health counters reset.",
        }

    sim.manager.mark_station_repaired(station_id, timestamp)

    return {
        "success": True,
        "station_id": station_id,
        "status": "WARNING",
        "recovery_active": True,
        "message": "Sensor marked for repair recovery. Clean readings will be evaluated before returning it to HEALTHY."
    }


# ---------------- POST /api/inject-anomaly ----------------

@app.post("/api/inject-anomaly")
def inject_anomaly(body: dict):
    """
    Repurposed as the REPLAY-MODE trigger -- see simulator.py's
    start_replay() docstring. body's station_id/type are accepted for
    contract-shape compatibility but not used to target one station;
    replay always drives all 20 simultaneously from their own
    pre-injected faults, then auto-reverts to live mode when exhausted.
    """
    sim = app.state.sim
    if sim.mode == "replay":
        raise HTTPException(status_code=409, detail="Simulator replay already running.")
    anomaly_id = sim.start_replay()
    return {
        "success": True,
        "anomaly_id": anomaly_id,
        "message": "Simulator started: replaying labeled historical data with injected faults across all stations.",
    }


# ---------------- POST /api/maintenance-ticket ----------------

_ticket_counter = 0

@app.post("/api/maintenance-ticket")
def create_maintenance_ticket(body: dict):
    global _ticket_counter
    sim = app.state.sim

    anomaly_id = body.get("anomaly_id")
    match = next((a for a in sim.recent_anomalies if a["anomaly_id"] == anomaly_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Unknown anomaly_id {anomaly_id}")

    _ticket_counter += 1
    from datetime import datetime, timezone
    return {
        "ticket_id": f"TCK-{_ticket_counter:04d}",
        "station_id": match["station_id"],
        "issue": match["root_cause"],
        "priority": "high" if match["severity"] in ("high", "critical") else "medium",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _health_pct(parameter_status: dict[str, str]) -> int:
    """Stable health summary from actual per-parameter circuit breakers."""
    if not parameter_status:
        return 100
    score_by_status = {"HEALTHY": 100, "WARNING": 50, "OFFLINE": 0}
    return round(sum(score_by_status.get(value, 0) for value in parameter_status.values()) / len(parameter_status))
