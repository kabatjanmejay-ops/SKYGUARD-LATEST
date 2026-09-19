"""
SkyGuard AI — timescale_store.py
Implementation of TimescaleDB for persistent, long-horizon sensor history.

This replaces the CSV-based `history_store.py` to allow efficient
time-series querying, automatic data retention policies, and scalable
writes using PostgreSQL and TimescaleDB.
"""

import os
import psycopg2
from psycopg2.extras import DictCursor
import pandas as pd
from datetime import datetime

# Schema definition for TimescaleDB
SCHEMA_SETUP_SQL = """
CREATE TABLE IF NOT EXISTS sensor_history (
    timestamp TIMESTAMPTZ NOT NULL,
    station_id TEXT NOT NULL,
    temperature_c DOUBLE PRECISION,
    pressure_hpa DOUBLE PRECISION,
    humidity_pct DOUBLE PRECISION,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    fault_type TEXT,
    severity TEXT,
    anomaly_score_pct DOUBLE PRECISION,
    suggested_temperature_c DOUBLE PRECISION,
    suggested_pressure_hpa DOUBLE PRECISION,
    suggested_humidity_pct DOUBLE PRECISION,
    health_status TEXT,
    source TEXT NOT NULL,
    UNIQUE(station_id, timestamp, source)
);

-- Convert standard PostgreSQL table into a TimescaleDB hypertable
-- partitioned by timestamp
SELECT create_hypertable('sensor_history', 'timestamp', if_not_exists => TRUE);

-- Add an index for fast lookups by station_id and timestamp
CREATE INDEX IF NOT EXISTS ix_sensor_history_station_time 
    ON sensor_history (station_id, timestamp DESC);
    
CREATE INDEX IF NOT EXISTS ix_sensor_history_source 
    ON sensor_history (source, timestamp DESC);
"""

RAW_PARAMS = ["temperature_c", "pressure_hpa", "humidity_pct"]

from contextlib import contextmanager
from psycopg2 import pool

from concurrent.futures import ThreadPoolExecutor

class TimescaleStore:
    def __init__(self, connection_string: str = None, max_days: int = 90):
        self.conn_string = connection_string or os.environ.get(
            "TIMESCALEDB_URI", 
            "postgresql://postgres:postgres@localhost:5432/skyguard"
        )
        self.max_days = max_days
        self._executor = ThreadPoolExecutor(max_workers=5)
        
        # Create a thread-safe connection pool to avoid opening a new TCP/TLS connection on every query
        # Min connections: 1, Max connections: 50 (sufficient for concurrent requests + simulation loop)
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(1, 50, self.conn_string)
        except psycopg2.OperationalError as e:
            print(f"[TimescaleDB] Could not initialize connection pool: {e}")
            self.pool = None
            
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Yields a connection from the pool, returning it automatically."""
        if not self.pool:
            raise RuntimeError("Database connection pool is not initialized.")
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    def _init_db(self):
        """Initializes the TimescaleDB hypertable."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA_SETUP_SQL)
                conn.commit()
        except psycopg2.OperationalError as e:
            print(f"[TimescaleDB] Could not connect to database: {e}")
            # In a production scenario, you might want to crash or fallback

    def append(self, station_id: str, timestamp, raw_reading: dict, verdict: dict, source: str):
        """Writes ONE row per ingested reading (live or replay) to TimescaleDB."""
        # Execute DB insert in background thread to prevent blocking the FastAPI event loop
        self._executor.submit(self._sync_append, station_id, timestamp, raw_reading, verdict, source)

    def _sync_append(self, station_id: str, timestamp, raw_reading: dict, verdict: dict, source: str):
        suggested = verdict.get("suggested_values", {}) or {}
        
        # Ensure timestamp is proper
        ts = pd.Timestamp(timestamp).to_pydatetime()
        
        query = """
            INSERT INTO sensor_history (
                timestamp, station_id, temperature_c, pressure_hpa, humidity_pct,
                is_anomaly, fault_type, severity, anomaly_score_pct,
                suggested_temperature_c, suggested_pressure_hpa, suggested_humidity_pct,
                health_status, source
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (station_id, timestamp, source) DO NOTHING;
        """
        
        values = (
            ts,
            station_id,
            raw_reading.get("temperature_c"),
            raw_reading.get("pressure_hpa"),
            raw_reading.get("humidity_pct"),
            bool(verdict.get("is_anomaly", False)),
            verdict.get("fault_type"),
            verdict.get("severity"),
            verdict.get("anomaly_score_pct"),
            suggested.get("temperature_c"),
            suggested.get("pressure_hpa"),
            suggested.get("humidity_pct"),
            verdict.get("health_status"),
            source
        )

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, values)
                conn.commit()
        except Exception as e:
            print(f"[TimescaleDB] Failed to insert row: {e}")

        # Note: We don't manually trim per-append here.
        # TimescaleDB data retention policies should handle trimming:
        # SELECT add_retention_policy('sensor_history', INTERVAL '90 days');

    def mark_spike(self, station_id: str, timestamp, parameter: str, suggested_value: float, source: str):
        """Retroactively annotate the original one-reading spike in the database."""
        ts = pd.Timestamp(timestamp).to_pydatetime()
        
        col_name = f"suggested_{parameter}"
        if col_name not in [f"suggested_{p}" for p in RAW_PARAMS]:
            return
            
        query = f"""
            UPDATE sensor_history
            SET is_anomaly = TRUE,
                fault_type = 'spike',
                severity = 'medium',
                {col_name} = %s
            WHERE station_id = %s AND timestamp = %s AND source = %s;
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (suggested_value, station_id, ts, source))
            conn.commit()

    def get_recent(
        self,
        station_id: str,
        hours: float = 24,
        relative_to: str = "latest",
        source: str | None = None,
    ) -> pd.DataFrame:
        """Returns retained rows for one station as a DataFrame."""
        with self._get_connection() as conn:
            # Determine cutoff
            if relative_to == "latest":
                if source:
                    latest_q = "SELECT MAX(timestamp) FROM sensor_history WHERE station_id = %s AND source = %s"
                    latest_vals = (station_id, source)
                else:
                    latest_q = "SELECT MAX(timestamp) FROM sensor_history WHERE station_id = %s"
                    latest_vals = (station_id,)
                    
                with conn.cursor() as cur:
                    cur.execute(latest_q, latest_vals)
                    res = cur.fetchone()
                    if not res or not res[0]:
                        return pd.DataFrame()
                    anchor = res[0]
            else:
                anchor = datetime.now()

            cutoff = anchor - pd.Timedelta(hours=hours)

            query = "SELECT * FROM sensor_history WHERE station_id = %s AND timestamp >= %s"
            params = [station_id, cutoff]
            
            if source is not None:
                query += " AND source = %s"
                params.append(source)
                
            query += " ORDER BY timestamp ASC"

            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                # Ensure timestamp is pandas Timestamp aware
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                
            return df

    def get_all(self, station_id: str) -> pd.DataFrame:
        """Full retained window for one station."""
        with self._get_connection() as conn:
            query = "SELECT * FROM sensor_history WHERE station_id = %s ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn, params=[station_id])
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df

    def clear_source(self, station_id: str, source: str):
        """PURGES every row tagged `source` from one station's persisted table."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM sensor_history WHERE station_id = %s AND source = %s",
                    (station_id, source)
                )
            conn.commit()

    def clear_all(self, source: str = None):
        """Purges `source` rows (or, if None, ALL rows) across every station."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if source is None:
                    cur.execute("TRUNCATE sensor_history")
                else:
                    cur.execute("DELETE FROM sensor_history WHERE source = %s", (source,))
            conn.commit()
