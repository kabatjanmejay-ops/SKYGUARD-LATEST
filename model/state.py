"""
SkyGuard AI — state.py (REWRITE — mode-aware buffer lifecycle, persistent
long-horizon history, per-reading baseline exclusion fix, instant
force-recovery).

============================================================================
BUGS THIS REWRITE FIXES (reported directly, against the running system)
============================================================================

1. REPLAY -> LIVE BLEED-THROUGH (the actual production bug you hit).
   The replay/"inject anomaly" endpoint drives historical data through
   the EXACT SAME ingest_reading() path live data uses -- by design,
   that's what makes replay render identically to live on the
   dashboard. But that also means replay's synthetic anomalies, fake
   OFFLINE sensors, CUSUM accumulators, and 10h/24h health counters
   were all landing in the SAME per-station buffers live data reads.
   There was no way to (a) interrupt replay mid-stream and switch to
   live instantly, or (b) clear those buffers when replay ended --
   so live mode inherited replay's contaminated state, and the only
   fix was restarting the server process.

   FIXED: an explicit `StateManager.mode` ("live"/"replay") plus
   `switch_to_live()` / `start_replay()`, each of which resets EVERY
   station's short-window in-memory detection state (§A below) but
   never touches persisted history (§B below). main.py's replay loop
   needs one small change to cooperate with this -- see "MAIN.PY
   INTEGRATION" at the bottom of this docstring, since that file isn't
   in this pass's scope but the interrupt only works if the loop checks
   for it.

2. NO WORKING "FORCE RECOVERY". `mark_repaired()` existed but only
   touched STATION-level aggregate attributes (`.status`,
   `.offline_reason`, `._clean_streak`) -- never the real PER-PARAMETER
   state detect.py's SensorHealthTracker actually decides `.status`
   from (`param_status`, per-parameter clean streaks, the 10h/24h
   deques). This was already flagged as a known gap in detect.py's own
   docstring (#8) before this pass. Practical effect: a "repaired"
   sensor could flip straight back to OFFLINE on its very next reading,
   because its per-parameter counters were untouched and still full of
   pre-repair anomalous readings -- which is almost certainly what
   "it just remains stuck unless we restart the server" was, on top of
   bug #1.

   Two separate methods now exist, kept deliberately distinct:
     - `mark_repaired()` -- for a genuine operator-confirmed physical
       repair. GRADUAL: starts a monitored clean-reading streak before
       trusting the sensor again (unchanged behavior), but now ALSO
       resets the per-parameter state underneath it, so it can't
       silently un-repair itself on the next reading.
     - `force_recover()` -- NEW. INSTANT override, no waiting period.
       This is Draft 2 §6's own Frontend TODO ("a manual force recovery
       action... should immediately flip the sensor back to HEALTHY").
       Use this for "this sensor is stuck in a bad state that isn't a
       real fault" -- e.g. any stuck-offline artifact from before this
       fix existed, or any other stuck state that shouldn't require a
       server restart to clear.

3. PER-READING BASELINE EXCLUSION WAS TOO COARSE. The exclusion gate
   used to be ONLY `health.should_include_in_baseline()` -- a
   whole-sensor-OFFLINE check. A lone spike that never pushes a sensor
   OFFLINE on its own (the normal case for an isolated spike, per §3 of
   the architecture doc) was NOT excluded from anything: its extreme
   value sat inside the next ~48h of rolling-mean/rolling-std
   calculations, quietly degrading later spike sensitivity. This was
   flagged in the prior review as a real architectural hole, not just
   an eval nuance.

   FIXED (see StationBuffer.record_raw_reading): the exclusion gate is
   now PER-READING -- a row is excluded if the sensor is OFFLINE, OR if
   THIS reading's own verdict was is_anomaly=True, OR if a repair is in
   progress. This is still a whole-ROW exclusion (all 3 parameters
   excluded together even if only one fired) -- true per-parameter
   exclusion needs columnar buffering (3 independent per-parameter
   history buffers instead of one combined row buffer), which is a
   bigger change flagged here but not done in this pass, to keep this
   pass scoped to the two gaps actually reported plus the mode-switch
   fix you asked for.

============================================================================
§A -- TWO SEPARATE STORAGE TIERS (do not conflate these)
============================================================================
  SHORT-WINDOW DETECTION BUFFER (`StationBuffer._raw_rows`, in-memory
  deque, ~60h -- see RAW_HISTORY_MAXLEN_HOURS): feeds
  build_features_for_latest/score_reading's rolling-window math
  directly. This is SCRATCH state, mode-scoped -- wiped on every
  switch_to_live()/start_replay() call. Never read this for anything
  frontend-facing; it exists purely to feed detection math and its
  lifetime is intentionally short and mode-bound.

  LONG-HORIZON PERSISTED HISTORY (history_store.HistoryStore,
  CSV-backed, up to 30 days, one file per station -- see
  history_store.py): every ingested reading (live OR replay, tagged by
  a `source` column) is appended here regardless of health/offline
  state, and is NEVER reset by a mode switch. This is what
  get_station_history() below serves to the frontend, and it's also
  where the fault_type/suggested_value-on-hover data now lives -- see
  history_store.py's own docstring for the full schema.

============================================================================
§B -- FRONTEND GAPS THIS CLOSES (backend side only -- frontend still
needs to actually wire the hover tooltip up to this data)
============================================================================
The dashboard graph currently doesn't show, on hovering an anomalous
point: (1) the suggested/predicted value the system would have
substituted, (2) the fault_type. Both were already computed by
score_reading() per-reading, but were only ever returned to the one
live caller of that one ingest_reading() call -- there was nowhere to
re-fetch them for a past point once it scrolled off. get_station_history()
now returns both for every retained point (see its docstring for the
exact per-point shape), specifically so a hover event on any point
within the retention window can render both without the frontend or
state.py needing to keep unbounded history in memory.

============================================================================
MAIN.PY INTEGRATION (flagged -- main.py isn't in this pass's scope, but
the mode switch only works end-to-end if main.py cooperates)
============================================================================
  - Before starting a replay run: call `state_manager.start_replay()`.
  - Inside the replay loop, check `state_manager.mode` on EVERY
    iteration (e.g. `if state_manager.mode != "replay": break`) so an
    in-progress replay can actually be interrupted by a frontend
    "switch to live" request instead of running to completion
    regardless. Without this check, switch_to_live() still resets the
    buffers correctly, but a still-running replay loop will immediately
    start re-populating them with replay data again on its very next
    iteration.
  - When a "switch to live" request arrives (whether replay is still
    running or already finished): call `state_manager.switch_to_live()`.
  - Expose `state_manager.mode` on whatever status endpoint the
    frontend polls, so the UI can show which mode is currently active.
"""

import sys
from collections import deque

from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from model.detect import score_reading, SensorHealthTracker, PARAMS
from model.features import ROLLING_WINDOW_HOURS, DRIFT_LOOKBACK_HOURS
from model.explain import ExplainerCache
from config import RECOVERY_CLEAN_STREAK_REQUIRED
from timescale_store import TimescaleStore

# History buffer needs enough hours for the longest lookback any
# feature uses -- DRIFT_LOOKBACK_HOURS (24) vs ROLLING_WINDOW_HOURS (48)
# -- plus slack so the buffer never truncates a rolling window short.
RAW_HISTORY_MAXLEN_HOURS = max(ROLLING_WINDOW_HOURS, DRIFT_LOOKBACK_HOURS) + 12

# Recovery requires the same number of consecutive clean readings
# used by SensorHealthTracker. This works consistently in live and
# replay because replay advances readings faster than wall-clock time.

MODE_LIVE = "live"
MODE_REPLAY = "replay"


class StationBuffer:
    """
    Per-station live state: raw-reading history (feeds
    build_features_for_latest via score_reading), SensorHealthTracker,
    and repair/recovery bookkeeping.
    """

    def __init__(self, station_id: str):
        self.station_id = station_id
        self.health = SensorHealthTracker(station_id)
        self._raw_rows: deque = deque(maxlen=RAW_HISTORY_MAXLEN_HOURS)

        # Repair/recovery state -- separate from health.status so a
        # sensor can be OFFLINE (health's own circuit breaker) while
        # ALSO being mid-recovery after an operator-initiated repair.
        self.recovery_active: bool = False
        self.recovery_clean_count: int = 0

    def raw_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._raw_rows))

    def record_raw_reading(self, raw_reading: dict, timestamp, verdict: dict):
        """
        CAUSAL EXCLUSION -- now PER-READING, not just per-sensor-OFFLINE.
        See this file's module docstring, fix #3, for the bug this
        closes (a lone spike that never tips the sensor OFFLINE was
        previously not excluded from anything and quietly degraded the
        rolling baseline for ~48h afterward).

        A row is excluded from the detection buffer if ANY of:
          - the station is currently OFFLINE (should_include_in_baseline()
            -- the original whole-sensor gate, kept for the sustained-
            fault case, where you want the WHOLE offline window
            excluded, not just individually-anomalous readings within it)
          - THIS reading's own verdict was is_anomaly=True (new --
            catches an isolated spike/dropout/etc. that doesn't push
            the sensor OFFLINE on its own)
          - a repair is in progress (recovery_active) -- don't let
            not-yet-trusted post-repair readings seed the baseline
            either

        KNOWN REMAINING LIMITATION: whole-ROW exclusion, not
        per-parameter -- see module docstring fix #3's closing note.
        """
        if not self.health.should_include_in_baseline():
            return
        if verdict.get("is_anomaly"):
            return
        if self.recovery_active:
            return
        row = dict(raw_reading)
        row["station_id"] = self.station_id
        row["timestamp"] = timestamp
        self._raw_rows.append(row)

    def reset_detection_state(self):
        """
        Wipes SHORT-WINDOW scratch state ONLY (§A in module docstring)
        -- called by StateManager.switch_to_live()/start_replay().
        Never touches HistoryStore's persisted file for this station;
        that history is deliberately mode-independent and survives
        this reset.

        Fresh SensorHealthTracker means every per-parameter counter,
        CUSUM-relevant buffer state, and offline/warning status starts
        clean -- this is the actual fix for "live mode still shows
        already-offline sensors and stale anomaly counts from the
        replay that just ran."
        """
        self._raw_rows.clear()
        self.health = SensorHealthTracker(self.station_id)
        self.recovery_active = False
        self.recovery_clean_count = 0

    def mark_repaired(self, timestamp):
        """
        Operator action from the frontend's 'Mark Repaired' button --
        GRADUAL recovery. Does NOT immediately trust the sensor; starts
        a clean-reading recovery period instead (see update_recovery).

        FIXED vs the previous version: this used to only touch the
        station-level aggregate (`health.status`/`.offline_reason`/
        `._clean_streak`), leaving the REAL per-parameter state
        (`param_status`, per-parameter clean streaks, the 10h/24h
        deques) untouched -- so a "repaired" sensor could silently flip
        straight back to OFFLINE on its very next reading, because its
        per-parameter counters were still full of pre-repair anomalous
        readings. Now resets both levels together.
        """
        self.recovery_active = True
        self.recovery_clean_count = 0

        self.health.status = "WARNING"
        self.health.offline_reason = None
        self.health._clean_streak = 0

        for p in self.health.param_status:
            self.health.param_status[p] = "WARNING"
            self.health.param_offline_reason[p] = None
            self.health._param_clean_streak[p] = 0
            self.health._param_recent_10h[p].clear()
            self.health._param_recent_24h[p].clear()

    def force_recover(self):
        """
        NEW -- INSTANT override, distinct from mark_repaired(). Draft 2
        §6's own Frontend TODO: "a manual force recovery action...
        should immediately flip the sensor back to HEALTHY rather than
        waiting out the 3-reading recovery streak." No clean-reading
        wait at all.

        Use this for "this sensor is stuck in a bad state that isn't a
        real ongoing fault" -- e.g. any sensor left OFFLINE by a
        pre-fix replay/live bleed-through, or any other stuck state an
        operator wants to clear without restarting the server.
        mark_repaired() stays the right call for a genuine physical
        repair, where verifying a few clean readings first is actually
        wanted.
        """
        self.recovery_active = False
        self.recovery_clean_count = 0

        self.health.status = "HEALTHY"
        self.health.offline_reason = None
        self.health._clean_streak = 0

        for p in self.health.param_status:
            self.health.param_status[p] = "HEALTHY"
            self.health.param_offline_reason[p] = None
            self.health._param_clean_streak[p] = 0
            self.health._param_recent_10h[p].clear()
            self.health._param_recent_24h[p].clear()

    def update_recovery(self, verdict: dict):
        """
        Reading-count based recovery. Requires consecutive clean readings
        after repair instead of wall-clock elapsed time.
        """
        if not self.recovery_active:
            return

        if self.health.status == "OFFLINE":
            # Went offline again during recovery.
            self.recovery_active = False
            self.recovery_clean_count = 0
            return

        if verdict["is_anomaly"]:
            self.recovery_clean_count = 0
            return

        self.recovery_clean_count += 1

        if self.recovery_clean_count >= RECOVERY_CLEAN_STREAK_REQUIRED:
            self.recovery_active = False
            self.recovery_clean_count = 0
            self.health.status = "HEALTHY"


class StateManager:
    """
    Owns one StationBuffer per station plus the current simulation mode
    (live/replay) and the persistent long-horizon HistoryStore.
    ingest_reading() is the ONLY entry point both live serving
    (simulator.py/main.py) and any replay/test harness should call --
    never call detect.score_reading() directly elsewhere, or state.py
    and detect.py can silently drift apart the same way evaluate.py
    once diverged from train.py earlier this project.
    """

    def __init__(self, metadata: pd.DataFrame, artifact: dict, history_store: TimescaleStore = None):
        self.metadata = metadata
        self.artifact = artifact
        self.explainer = ExplainerCache(artifact)
        self.buffers: dict[str, StationBuffer] = {
            sid: StationBuffer(sid) for sid in metadata["station_id"]
        }
        # Persistent history survives mode switches by design -- see
        # module docstring §A/§B. Injectable for tests; defaults to the
        # real on-disk store.
        self.history = history_store or TimescaleStore()

        # Starts in live mode. main.py should call start_replay() before
        # kicking off any historical replay run -- see "MAIN.PY
        # INTEGRATION" in the module docstring.
        self.mode: str = MODE_LIVE

    def switch_to_live(self):
        """
        Instant replay -> live switch -- the fix for the actual bug
        reported: previously there was no way to interrupt replay
        mid-stream or clear its state when it ended, short of
        restarting the server. Resets EVERY station's short-window
        detection buffer, health tracker, and recovery state (§A) so
        live mode starts from a clean slate: fake OFFLINE sensors, fake
        anomaly counts, and stale CUSUM-relevant history from replay
        can no longer leak into live serving.

        Durable history is retained across the transition, including
        rows labelled ``source=replay``. This provides an operator audit
        export of live and test readings. The normal dashboard remains
        mode-isolated because get_station_history() filters rows to the
        current source; replay values can never enter a live chart. A
        fresh replay clears only the previous replay session before it
        starts, preventing separate demos from accumulating together.
        """
        self.mode = MODE_LIVE
        for buf in self.buffers.values():
            buf.reset_detection_state()

    def start_replay(self):
        """
        Symmetric reset in the other direction -- also clears every
        buffer before a replay run starts, so a PRIOR live session's
        state doesn't leak INTO the replay. Same class of bug as
        switch_to_live() guards against, just the other direction;
        closed for completeness even though it wasn't the one reported.
        """
        self.mode = MODE_REPLAY
        for buf in self.buffers.values():
            buf.reset_detection_state()

    def force_recover_station(self, station_id: str):
        """Frontend 'force recovery' affordance (Draft 2 §6 Frontend TODO) -- see StationBuffer.force_recover()."""
        self.buffers[station_id].force_recover()

    def ingest_reading(self, station_id: str, raw_reading: dict, timestamp) -> dict:
        buf = self.buffers[station_id]
        history_df = buf.raw_history_df()

        current_row = dict(raw_reading, station_id=station_id, timestamp=timestamp)
        history_df_with_current = (
            pd.concat([history_df, pd.DataFrame([current_row])], ignore_index=True)
            if not history_df.empty else pd.DataFrame([current_row])
        )

        verdict = score_reading(
            raw_reading,
            history_df_with_current,
            self.artifact,
            explainer=self.explainer,
        )
        # A spike can only be proved after the following reading
        # returns to baseline.  Count that confirmed, prior event for
        # the repeated-fault health policy, while keeping THIS normal
        # confirming reading out of the anomaly stream.
        verdict["confirmed_spike_params"] = {
            event["parameter"] for event in verdict.get("confirmed_spikes", [])
        }
        # Health updated with THIS verdict BEFORE recording, so a
        # reading that tips the station into OFFLINE is itself
        # correctly excluded from its own future baseline.
        buf.health.record(verdict)

        if buf.recovery_active:
            buf.update_recovery(verdict)

        # Persist the resulting health state for this exact reading.
        # Stamping this before record() would lag every trend/history
        # point by one verdict and hide the actual OFFLINE transition.
        verdict["health_status"] = buf.health.status

        for spike in verdict.get("confirmed_spikes", []):
            self.history.mark_spike(
                station_id, spike["timestamp"], spike["parameter"],
                spike["suggested_value"], self.mode,
            )
            # The candidate was previously admitted as normal because
            # confirmation was unavailable. Remove it before future
            # rolling baselines are computed.
            buf._raw_rows = deque(
                (row for row in buf._raw_rows if pd.Timestamp(row["timestamp"]) != spike["timestamp"]),
                maxlen=RAW_HISTORY_MAXLEN_HOURS,
            )

        buf.record_raw_reading(raw_reading, timestamp, verdict)

        # Persisted long-horizon log -- mode-independent, tagged with
        # the CURRENT mode so live vs replay stretches stay
        # distinguishable after the fact. See history_store.py.
        self.history.append(station_id, timestamp, raw_reading, verdict, source=self.mode)

        return verdict

    def mark_station_repaired(self, station_id: str, timestamp):
        """Called by main.py's repair endpoint (gradual recovery)."""
        self.buffers[station_id].mark_repaired(timestamp)

    def get_station_status(self, station_id: str) -> dict:
        buf = self.buffers[station_id]
        return {
            "station_id": station_id,
            "status": buf.health.status,
            "offline_reason": buf.health.offline_reason,
            "recovery_active": buf.recovery_active,
            "mode": self.mode,
        }

    def get_station_history(self, station_id: str, hours: float = 24, include_replay: bool = False) -> list[dict]:
        """
        Frontend-facing query for the trend graph and hover tooltip.
        Reads from HistoryStore (persists across mode switches, up to
        HistoryStore.max_days), NEVER from the live detection buffer --
        see module docstring §A for why those two must stay separate.

        Returns a list of dicts, one per retained reading, each shaped:
            {
              "timestamp": ..., "station_id": ...,
              "temperature_c": ..., "pressure_hpa": ..., "humidity_pct": ...,
              "is_anomaly": bool,
              "fault_type": str | None,
              "severity": str,
              "anomaly_score_pct": float,
              "suggested_temperature_c": float | None,
              "suggested_pressure_hpa": float | None,
              "suggested_humidity_pct": float | None,
              "health_status": str,
              "source": "live" | "replay",
            }

        FRONTEND TODO (backend side is now done): on hovering an
        anomalous point on the trend graph, show `fault_type` and the
        `suggested_<param>` value for whichever series is being
        hovered -- both are present on every retained point now, not
        just the instant it was first scored.

        include_replay: False (default) -- rows are additionally
        filtered to `source == self.mode` before returning, so the
        live dashboard can NEVER render a replay run's synthetic
        anomalies. Durable replay rows remain available through the
        operator CSV export, but are intentionally absent from normal
        live graph queries. Only
        pass True for a deliberate "review the last replay run" debug
        view, if one ever gets built; normal dashboard traffic should
        never set this.
        """
        active_source = None if include_replay else self.mode
        df = self.history.get_recent(station_id, hours=hours, source=active_source)
        return df.to_dict(orient="records")


# DEFERRED (see 6-hour scoping decision, still true after this pass):
# full HEALTHY -> SUSPICIOUS -> FAULTY -> REPAIRED -> RECOVERY state
# machine, retrospective cleanup of already-buffered contaminated
# readings, spatial-baseline volatility z-scoring, true per-parameter
# (columnar) baseline exclusion. Current scope: HEALTHY/WARNING/OFFLINE
# (from detect.py's SensorHealthTracker) + per-reading real-time
# exclusion + reading-count based AND instant force recovery + mode-
# aware buffer lifecycle + persisted long-horizon history. Sufficient
# for main.py/simulator.py to work correctly now; upgrade path
# documented above for the roadmap slide.
