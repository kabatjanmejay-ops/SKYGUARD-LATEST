"""
SkyGuard AI — simulator.py: drives the live/demo data stream.

TWO MODES, switched by the whole system at once (not per-station):

  LIVE (default): fetches CURRENT real weather from Open-Meteo for
  each station's real coordinates. No faults -- genuine, unsupervised
  detection against real present-day conditions. This is the actual
  production behavior.

  REPLAY (demo/simulator mode): steps through all 20 *_labeled.csv
  files in lockstep at live cadence. These already contain real
  injected faults with known ground truth (anomaly_injector.py) --
  detect.py scores each reading BLIND, exactly as it would any other
  reading, and whatever it flags is a genuine model detection, not a
  scripted/faked frontend event. This replaces an earlier design that
  reimplemented fault shapes live; that's been removed -- there's no
  reason to reinvent faults when validated labeled data already exists.

MODE SWITCH VIA EXISTING CONTRACT: the frontend's POST
/api/inject-anomaly button ({station_id, type} -> {anomaly_id,
message}) is the only trigger the frontend already has wired, and the
frontend doc explicitly prohibits inventing new endpoints. So that
route (wired in main.py, not here) calls SimulatorState.start_replay(),
which switches the WHOLE system into replay mode -- station_id/type
are accepted for contract-shape compatibility but not used to target
a single station, since replay always drives all 20 simultaneously
from their own pre-injected faults. Replay auto-reverts to LIVE once
every labeled file is exhausted. If per-station/per-type targeting is
actually wanted instead, this needs revisiting -- flagging the
assumption rather than guessing further.

LIVE-FETCH CAVEAT: the exact Open-Meteo "current conditions" call
below is written to the pattern data_fetch.py's archive-history call
likely follows (same station coordinates from stations_metadata.csv),
but I don't have data_fetch.py in context to confirm the exact
endpoint/params it uses. Verify _fetch_live_reading against your real
data_fetch.py before relying on it -- the shape may need adjusting.

============================================================================
PATCH (this pass): REPLAY -> LIVE was silently skipping state.py's own
purge fix. THIS IS THE ACTUAL BUG YOU FLAGGED.
============================================================================
The old start_replay()/_stop_replay() each did:

    self.manager = StateManager(self.metadata, self.manager.artifact)

-- i.e. they THREW AWAY the old StateManager and built a brand new one
from scratch, instead of calling the mode-switch methods state.py's own
rewrite was specifically built to provide (StateManager.start_replay() /
switch_to_live()). This mattered for one concrete reason:

  StateManager.__init__ does `self.history = history_store or
  HistoryStore()`. A freshly constructed HistoryStore() still points at
  the SAME on-disk DATA_DIR/data/history/ -- it's not a fresh sandbox,
  it's the exact same per-station CSV files the old manager was writing
  to. So building a new StateManager did NOT clear anything on disk.
  It just meant switch_to_live()'s actual fix -- purging every
  source="replay" row via HistoryStore.clear_all(source="replay") --
  never ran at all, on either transition. Replay's synthetic injected
  anomalies were left sitting permanently in the same file live rows
  get appended to, EXACTLY the bleed-through bug state.py's own rewrite
  was supposed to close. The only thing hiding it was
  get_station_history()'s defensive `source == self.mode` READ-TIME
  filter, which state.py's own docstring explicitly says is a second
  layer, "not a replacement for" the real purge.

  Rebuilding StateManager from scratch also meant every mode transition
  was silently O(this-run-only) instead of using the reset methods that
  actually integrate with HistoryStore correctly.

FIXED: both transitions now call self.manager.start_replay() /
self.manager.switch_to_live() on the SAME long-lived manager instance.
The manager (and its one HistoryStore) is now created ONCE, in
SimulatorState.__init__, and never rebuilt for the life of the process.

ALSO FIXED:
  - Import path: state.py lives at the backend root (imports bare
    `from config import ...` / `from history_store import HistoryStore`
    -- both root-level modules, not `model/`-prefixed), not inside
    model/. `from model.state import StateManager` would ImportError.
    Corrected to `from state import StateManager`; the sys.path append
    up to the parent of model/ is no longer needed for this import
    specifically but is left in place since detect.py/train.py's own
    `from model.X import Y` pattern still needs it if this file ever
    imports those directly.
  - Removed the duplicate `self.mode` string SimulatorState was
    tracking independently of StateManager.mode. Two independent mode
    trackers on two different objects is exactly the kind of "two
    derivations of the same fact that can silently drift apart" bug
    this project has hit before (see features.py's RULE_ONLY_PREFIXES
    comment, state.py/detect.py's threshold-drift postmortem). `.mode`
    is now a read-only property proxying self.manager.mode, so there is
    exactly one source of truth and any external caller (e.g. a status
    endpoint in main.py) reading sim_state.mode keeps working unchanged.
"""

import asyncio
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import pandas as pd

from model.state import StateManager
from config import RULE_BASE_CONFIDENCE, score_to_severity

DATA_DIR = Path(__file__).parent.parent / "data"
ARTIFACTS_PATH = Path(__file__).parent.parent / "model_artifacts" / "isolation_forest.pkl"

REPLAY_STEP_SECONDS = 2  # one historical hour is streamed every two wall-clock seconds
LIVE_FETCH_INTERVAL_SECONDS = 30 * 60  # Open-Meteo hourly data: poll no more than twice per hour
TREND_HISTORY_MAXLEN = 2000
RECENT_ANOMALIES_MAXLEN = 200

OPEN_METEO_CURRENT_URL = "https://api.open-meteo.com/v1/forecast"

ROOT_CAUSE_BY_FAULT_TYPE = {
    "physical_bounds": "Reading outside physically possible range",
    "dropout": "Sensor communication failure",
    "frozen_value": "Sensor stuck / communication fault",
    "drift": "Calibration drift suspected",
    "spike": "Sudden reading spike -- possible sensor malfunction",
    "statistical_anomaly": "Unusual reading pattern flagged by model",
}


async def _fetch_live_reading(client: httpx.AsyncClient, lat: float, lon: float) -> dict | None:
    """
    Pulls CURRENT conditions for one station's coordinates. Returns
    None on any failure so a transient network hiccup degrades to
    "reuse last cached value" (handled by the caller) rather than
    crashing the tick loop.

    VERIFY against your real data_fetch.py -- written to the likely
    Open-Meteo current-weather shape, not confirmed against your
    actual archive-fetch implementation.
    """
    try:
        resp = await client.get(
            OPEN_METEO_CURRENT_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,surface_pressure,relative_humidity_2m",
                "timezone": "UTC",
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()["current"]
        return {
            "temperature_c": float(data["temperature_2m"]),
            "pressure_hpa": float(data["surface_pressure"]),
            "humidity_pct": float(data["relative_humidity_2m"]),
            # Open-Meteo publishes hourly observations. Preserve that source
            # timestamp so server restarts/polls do not invent new readings.
            "_observed_at": data.get("time"),
        }
    except Exception as e:
        print(f"[simulator] live fetch failed for ({lat},{lon}): {e!r}")
        return None


class SimulatorState:
    """
    The object main.py's routes read from. One instance, created at
    FastAPI startup.

    self.manager is now created ONCE here and lives for the whole
    process -- see module docstring PATCH note. Mode transitions call
    self.manager.start_replay()/switch_to_live() on this SAME instance;
    they never rebuild it.
    """

    def __init__(self, metadata: pd.DataFrame, artifact: dict):
        self.metadata = metadata
        self.manager = StateManager(metadata, artifact)

        self.on_tick_callbacks = []
        self._replay_cursor_idx: int = 0
        self._replay_frames: dict[str, pd.DataFrame] = {}  # populated lazily on start_replay()
        self._replay_len: int = 0

        self._live_cache: dict[str, dict] = {}  # station_id -> last fetched reading
        self._live_observed_at: dict[str, datetime] = {}
        self._live_last_fetch: datetime | None = None
        # Last genuinely live readings remain available while an operator runs
        # a demo replay. They let the cards return immediately on exit rather
        # than showing a false "not found" state for up to 30 minutes.
        self._last_live_latest: dict[str, dict] = {}
        self._live_refresh_requested = False
        self._last_ingested: dict[str, dict] = {}
        self._last_ingested_timestamp: dict[str, datetime] = {}
        self.latest: dict[str, dict] = {}
        self.trend_history: dict[str, deque] = {
            sid: deque(maxlen=TREND_HISTORY_MAXLEN) for sid in metadata["station_id"]
        }
        self.recent_anomalies: deque = deque(maxlen=RECENT_ANOMALIES_MAXLEN)
        self._anomaly_counter = 0

        self._http_client = httpx.AsyncClient()

    @property
    def mode(self) -> str:
        """
        Read-only proxy to StateManager.mode -- see module docstring
        PATCH note for why this is no longer tracked as a separate
        field. Anything that used to read sim_state.mode (e.g. a status
        endpoint in main.py) keeps working unchanged.
        """
        return self.manager.mode

    async def close(self):
        """Release the shared live-data client during FastAPI shutdown."""
        await self._http_client.aclose()

    # ---------------- mode control ----------------

    def start_replay(self) -> str:
        """
        Called from main.py's POST /api/inject-anomaly handler. Loads
        every *_labeled.csv fresh (so repeated demo runs always replay
        from the start) and switches mode.

        FIXED (this patch): calls self.manager.start_replay() on the
        EXISTING manager instead of constructing a new StateManager.
        See module docstring PATCH note -- rebuilding silently skipped
        state.py's own history-purge logic since a fresh HistoryStore()
        still points at the same on-disk files.

        Returns a queued anomaly_id for the contract response -- the
        real detections happen asynchronously as tick() runs, this id
        is just a placeholder acknowledging the request per the
        existing response shape ("Anomaly injected and detected in next
        reading cycle.").
        """
        # A new replay must be a new *visual and diagnostic* session.  The
        # previous implementation reset cursor/buffers but retained older
        # source="replay" CSV rows.  get_recent() then served a window whose
        # newest timestamp came from a former run, while the new run restarted
        # at 2025-01-01 — exactly the piled-up chart seen in the dashboard.
        # Live rows are deliberately untouched.
        self.manager.history.clear_all(source="replay")
        if self.mode == "live":
            self._last_live_latest = dict(self.latest)

        self._replay_frames = {}
        for sid in self.metadata["station_id"]:
            path = DATA_DIR / f"{sid}_labeled.csv"
            if not path.exists():
                raise FileNotFoundError(f"No labeled data for {sid} at {path} -- run anomaly_injector.py first.")
            df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            self._replay_frames[sid] = df

        self._replay_len = min(len(df) for df in self._replay_frames.values())
        self._replay_cursor_idx = 0

        # Real fix: switches mode + resets every station's short-window
        # detection buffer (§A) on the SAME manager/HistoryStore, so a
        # prior live session's state can't leak into this replay run.
        self.manager.start_replay()

        # Simulator-local caches (trend graph buffer, last-ingested dedup,
        # anomaly feed) are this file's own concern, not state.py's --
        # still reset directly here.
        self._last_ingested = {}
        self._last_ingested_timestamp = {}
        self.latest = {}
        self.trend_history = {
            sid: deque(maxlen=TREND_HISTORY_MAXLEN)
            for sid in self.metadata["station_id"]
        }
        self.recent_anomalies.clear()

        self._anomaly_counter += 1
        return f"anom_{self._anomaly_counter:05d}"

    def _stop_replay(self):
        """
        FIXED (this patch): calls self.manager.switch_to_live() on the
        EXISTING manager instead of constructing a new StateManager.
        This is the transition that actually matters for your concern
        -- switch_to_live() resets the detection/health state on the
        existing manager. Durable replay records are retained for the
        history audit export, while StateManager's source filter keeps
        them out of every live graph. Rebuilding a fresh StateManager
        would bypass this explicit lifecycle entirely.
        """
        self.manager.switch_to_live()

        self._replay_frames = {}
        self._last_ingested = {}
        self._last_ingested_timestamp = {}
        # Restore the last Open-Meteo snapshot immediately. The next live
        # tick replaces it with a newly ingested sample; replay values never
        # leak back into the cards.
        self.latest = dict(self._last_live_latest)
        self.trend_history = {
            sid: deque(maxlen=TREND_HISTORY_MAXLEN)
            for sid in self.metadata["station_id"]
        }
        self.recent_anomalies.clear()

        # Keep the already-fetched Open-Meteo cache: it is the truthful latest
        # live reading and avoids a blank dashboard on the mode transition.
        self._live_last_fetch = None
        self._live_refresh_requested = True

    def stop_replay(self):
        """Operator-initiated replay -> live transition."""
        if self.mode == "replay":
            self._stop_replay()

    async def refresh_live_now(self) -> None:
        """Fetch the current provider observation outside the 30-min cadence.

        Used only on an explicit UI context change (station selection or
        replay->live), never by the regular polling loop.
        """
        if self.mode != "live":
            return
        self._live_last_fetch = None
        await self.tick()

    # ---------------- per-tick data sourcing ----------------

    async def _maybe_refresh_live_cache(self):
        now = datetime.now(timezone.utc)
        if (
            self._live_last_fetch is not None
            and (now - self._live_last_fetch).total_seconds() < LIVE_FETCH_INTERVAL_SECONDS
        ):
            return
        self._live_last_fetch = now

        station_requests = [
            (row["station_id"], _fetch_live_reading(self._http_client, row["lat"], row["lon"]))
            for _, row in self.metadata.iterrows()
        ]
        readings = await asyncio.gather(
            *(request for _, request in station_requests),
            return_exceptions=True,
        )
        for (sid, _), reading in zip(station_requests, readings):
            if isinstance(reading, Exception):
                print(f"[simulator] live fetch failed for {sid}: {reading!r}")
                continue
            if reading is not None:
                observed_at = reading.pop("_observed_at", None)
                if observed_at:
                    observed = pd.Timestamp(observed_at)
                    if observed.tzinfo is None:
                        observed = observed.tz_localize("UTC")
                    self._live_observed_at[sid] = observed.to_pydatetime()
                self._live_cache[sid] = reading
            # else: keep whatever was cached before -- degrade gracefully

    def _next_live_row(self, station_id: str) -> dict | None:
        return self._live_cache.get(station_id)  # None until first successful fetch

    def _next_replay_row(self, station_id: str) -> dict:
        df = self._replay_frames[station_id]
        row = df.iloc[self._replay_cursor_idx]
        return {
            "temperature_c": float(row["temperature_c"]) if pd.notna(row["temperature_c"]) else None,
            "pressure_hpa": float(row["pressure_hpa"]) if pd.notna(row["pressure_hpa"]) else None,
            "humidity_pct": float(row["humidity_pct"]) if pd.notna(row["humidity_pct"]) else None,
        }

    # ---------------- the tick ----------------

    async def tick(self):
        """
        One simulation step across all stations. Replay retains each
        CSV reading's original hourly timestamp even though the stream
        advances at two seconds per hour; graph placement and detector
        rolling windows must follow measurement time, never wall-clock
        animation speed. Live uses the current UTC measurement time.

        Reads self.mode (the manager.mode proxy) throughout -- see the
        property above; this is unchanged behavior, just now backed by
        the single real source of truth instead of a second tracked field.
        """
        now = datetime.now(timezone.utc)
        current_mode = self.mode

        if current_mode == "live":
            await self._maybe_refresh_live_cache()

        for station_id in self.metadata["station_id"]:
            if current_mode == "replay":
                raw_reading = self._next_replay_row(station_id)
                replay_row = self._replay_frames[station_id].iloc[self._replay_cursor_idx]
                reading_timestamp = pd.Timestamp(replay_row["timestamp"]).to_pydatetime()
            else:
                raw_reading = self._next_live_row(station_id)
                if raw_reading is None:
                    continue
                reading_timestamp = self._live_observed_at.get(station_id, now)

            # Replay: every CSV row is a genuine new reading.
            #
            # Live: only ingest when Open-Meteo gives us a genuinely
            # different reading. The UI still updates every 2 seconds,
            # but the ML/state history only gets new observations.
            should_ingest = (
                current_mode == "replay"
                or self._last_ingested_timestamp.get(station_id) != reading_timestamp
            )

            if should_ingest:
                verdict = self.manager.ingest_reading(
                    station_id,
                    raw_reading,
                    reading_timestamp
                )
                self._last_ingested[station_id] = dict(raw_reading)
                self._last_ingested_timestamp[station_id] = reading_timestamp
            else:
                cached = self.latest.get(station_id)
                if cached is None:
                    continue
                verdict = cached["verdict"]

            self.latest[station_id] = {
                "raw_reading": raw_reading,
                "verdict": verdict,
                "timestamp": reading_timestamp,
            }

            # UI receives replay points at stream cadence, but plots
            # these original timestamps on its x-axis.
            self.trend_history[station_id].append({
                "timestamp": reading_timestamp,
                "temperature_c": raw_reading.get("temperature_c"),
                "pressure_hpa": raw_reading.get("pressure_hpa"),
                "humidity_pct": raw_reading.get("humidity_pct"),
                "is_anomaly": verdict["is_anomaly"],
            })

            # A spike is confirmed by the following reading. Publish an
            # event at the original timestamp even though the current
            # confirming reading remains normal.
            for spike in verdict.get("confirmed_spikes", []):
                self._anomaly_counter += 1
                self.recent_anomalies.appendleft({
                    "anomaly_id": f"anom_{self._anomaly_counter:05d}",
                    "timestamp": spike["timestamp"],
                    "station_id": station_id,
                    "anomaly_score_pct": RULE_BASE_CONFIDENCE["spike"],
                    "model_confidence_pct": None,
                    "rule_confidence_pct": RULE_BASE_CONFIDENCE["spike"],
                    "suggested_values": {spike["parameter"]: spike["suggested_value"]},
                    "observed_values": {spike["parameter"]: spike["observed_value"]},
                    "affected_parameters": [spike["parameter"]],
                    "shap_features": [],
                    "likely_faulty_sensors": [spike["parameter"]],
                    "severity": "medium",
                    "type": "spike",
                    "root_cause": ROOT_CAUSE_BY_FAULT_TYPE["spike"],
                })

            # Only create a new anomaly event when a new reading
            # was actually processed.
            if should_ingest and verdict["is_anomaly"]:
                # Preserve the actual evidence with each incident.  Reports
                # must say which parameter was abnormal and what it read,
                # not merely name a fault category.
                affected_parameters = list(dict.fromkeys(
                    verdict.get("likely_faulty_sensors", [])
                    or [param for _rule, param, _confidence in verdict.get("rules_fired", [])]
                    or list((verdict.get("suggested_values") or {}).keys())
                ))
                observed_values = {
                    param: raw_reading.get(param)
                    for param in affected_parameters
                    if param in raw_reading
                }
                self._anomaly_counter += 1
                self.recent_anomalies.appendleft({
                    "anomaly_id": f"anom_{self._anomaly_counter:05d}",
                    "timestamp": reading_timestamp,
                    "station_id": station_id,
                    "anomaly_score_pct": verdict["anomaly_score_pct"],
                    "model_confidence_pct": verdict.get("model_confidence_pct"),
                    "rule_confidence_pct": verdict.get("rule_confidence_pct"),
                    "suggested_values": verdict.get("suggested_values"),
                    "observed_values": observed_values,
                    "affected_parameters": affected_parameters,
                    "shap_features": verdict.get("shap_features", []),
                    "likely_faulty_sensors": verdict.get("likely_faulty_sensors", []),
                    "severity": verdict["severity"],
                    "type": verdict["fault_type"] or "statistical_anomaly",
                    "root_cause": ROOT_CAUSE_BY_FAULT_TYPE.get(
                        verdict["fault_type"],
                        "Unusual reading pattern flagged by model"
                    ),
                })

        if current_mode == "replay":
            self._replay_cursor_idx += 1
            if self._replay_cursor_idx >= self._replay_len:
                self._stop_replay()

        for cb in self.on_tick_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(self.latest)
            else:
                cb(self.latest)


async def run_simulation_loop(sim_state: SimulatorState):
    """Replay steps every 2s; live fetch/ingest runs every 30 minutes."""
    next_live_tick = 0.0
    while True:
        try:
            now = asyncio.get_running_loop().time()
            if sim_state.mode == "replay" or now >= next_live_tick:
                await sim_state.tick()
                if sim_state.mode == "live":
                    # A replay can end inside tick().  Honour the requested
                    # immediate live refresh on the very next loop rather
                    # than scheduling the first live sample 30 minutes away.
                    if sim_state._live_refresh_requested:
                        next_live_tick = 0.0
                        sim_state._live_refresh_requested = False
                    else:
                        next_live_tick = asyncio.get_running_loop().time() + LIVE_FETCH_INTERVAL_SECONDS
        except Exception as e:
            # Preserve the actual file/line in server logs. A one-line error
            # hides whether an upstream record or detector rule failed.
            print(f"[simulator] tick failed: {e!r}\n{traceback.format_exc()}")
        # The short live wait notices a mode switch promptly. It does
        # not fetch or ingest live weather until next_live_tick.
        await asyncio.sleep(REPLAY_STEP_SECONDS if sim_state.mode == "replay" else 1)


def create_simulator_state() -> SimulatorState:
    """Called once from main.py's startup event."""
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(f"No trained model at {ARTIFACTS_PATH} -- run model/train.py first.")
    artifact = joblib.load(ARTIFACTS_PATH)
    if "rule_thresholds" not in artifact:
        raise KeyError("Artifact missing 'rule_thresholds' -- retrain with the current train.py.")

    metadata_path = DATA_DIR / "stations_metadata.csv"
    metadata = pd.read_csv(metadata_path)

    return SimulatorState(metadata, artifact)
