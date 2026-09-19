import { useState, useEffect, useCallback, useRef } from 'react';
import {
  CurrentSensorReading,
  TrendsResponse,
  LatestAnomaly,
  RecentAnomalyItem,
  TelemetryHistoryRecord,
  SensorHealth,
  SystemStreamStatus,
} from '../types';
import { currentReadingService } from '../services/currentReadingService';
import { trendsService } from '../services/trendsService';
import { anomalyService } from '../services/anomalyService';
import { sensorHealthService } from '../services/sensorHealthService';
import { systemStatusService } from '../services/systemStatusService';
import { ApiError, formatUserErrorMessage } from '../services/apiError';
import { calculateFreshness, FreshnessState } from '../utils/freshness';
import { TELEMETRY_REFRESH_EVENT } from '../utils/refreshEvents';
import { isMockMode, API_CONFIG } from '../config/api.config';

export interface DashboardDataState {
  currentReading: CurrentSensorReading | null;
  trends: TrendsResponse | null;
  latestAnomaly: LatestAnomaly | null;
  recentAnomalies: RecentAnomalyItem[];
  telemetryHistory: TelemetryHistoryRecord[];
  sensorHealth: SensorHealth | null;
  
  // Section-isolated loading states
  isLoadingReading: boolean;
  isLoadingTrends: boolean;
  isLoadingAnomalies: boolean;
  isLoadingHealth: boolean;

  // Section-isolated error states
  readingError: string | null;
  trendsError: string | null;
  anomaliesError: string | null;
  healthError: string | null;

  // Real-time metadata & freshness
  lastUpdated: Date | null;
  isStale: boolean;
  isDelayed: boolean;
  staleStatusText: 'LIVE' | 'DATA DELAYED' | 'DATA STALE';
  freshness: FreshnessState;
  pollStatusText: string;
  streamMode: 'live' | 'replay';

  // Pause / Resume controls [FRONTEND ONLY]
  isPaused: boolean;
  setIsPaused: (paused: boolean) => void;
  togglePause: () => void;

  // Manual triggers
  refreshAll: () => Promise<void>;
  refreshReading: () => Promise<void>;
  refreshTrends: (hours?: number) => Promise<void>;
  refreshAnomalies: () => Promise<void>;
  refreshHealth: () => Promise<void>;
  syncStreamStatus: () => Promise<void>;
}

export interface UseDashboardDataOptions {
  pollingIntervalMs?: number;
  autoPoll?: boolean;
  trendHours?: number;
}

/**
 * useDashboardData
 * 
 * Centralized telemetry data & polling hook reused across Dashboard and Monitor:
 * - Listens to selected stationId from StationContext
 * - Manages single centralized polling loop against service boundaries
 * - Provides pause/resume frontend controls without breaking state
 * - Maintains bounded telemetry history (max 150 items) derived from live feeds
 * - Provides section-isolated loading & error boundaries
 * - Implements frontend data staleness detection and graceful failure recovery
 */
export function useDashboardData(
  stationId: string | null | undefined,
  options: UseDashboardDataOptions = {}
): DashboardDataState {
  const { pollingIntervalMs = 30 * 60 * 1000, autoPoll = true, trendHours = 10 } = options;
  const trendHoursRef = useRef(trendHours);
  trendHoursRef.current = trendHours;

  const [currentReading, setCurrentReading] = useState<CurrentSensorReading | null>(null);
  const [trends, setTrends] = useState<TrendsResponse | null>(null);
  const [latestAnomaly, setLatestAnomaly] = useState<LatestAnomaly | null>(null);
  const [recentAnomalies, setRecentAnomalies] = useState<RecentAnomalyItem[]>([]);
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryHistoryRecord[]>([]);
  const [sensorHealth, setSensorHealth] = useState<SensorHealth | null>(null);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [streamStatus, setStreamStatus] = useState<SystemStreamStatus>({
    mode: 'live', replay_step_seconds: null, live_poll_interval_seconds: 30 * 60,
  });

  const [isLoadingReading, setIsLoadingReading] = useState<boolean>(true);
  const [isLoadingTrends, setIsLoadingTrends] = useState<boolean>(true);
  const [isLoadingAnomalies, setIsLoadingAnomalies] = useState<boolean>(true);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);

  const [readingError, setReadingError] = useState<string | null>(null);
  const [trendsError, setTrendsError] = useState<string | null>(null);
  const [anomaliesError, setAnomaliesError] = useState<string | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // References to prevent overlapping polling requests and race conditions
  const activeStationIdRef = useRef(stationId);
  activeStationIdRef.current = stationId;
  const isFetchingReadingRef = useRef(false);
  const isFetchingHealthRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const modeTimerRef = useRef<number | null>(null);
  const priorStreamModeRef = useRef<'live' | 'replay'>(streamStatus.mode);
  const isPausedRef = useRef(isPaused);
  isPausedRef.current = isPaused;

  // 1. Fetch Current Reading
  const fetchReading = useCallback(async () => {
    if (!stationId) return;
    if (isFetchingReadingRef.current) return;
    isFetchingReadingRef.current = true;
    const targetStationId = stationId;

    try {
      const data = await currentReadingService.getCurrentReading(targetStationId);
      if (activeStationIdRef.current !== targetStationId) {
        // Discard stale response if active station changed while awaiting network
        return;
      }
      setCurrentReading(data);
      setReadingError(null);
      const updateTime = new Date();
      setLastUpdated(updateTime);

      // Immediate rendering path: add the freshly processed backend
      // reading to the timestamped graph without waiting for CSV-backed
      // history or re-fetching the whole trend range.
      setTrends((previous) => {
        if (!previous || previous.station_id !== data.station_id) return previous;
        const point = {
          timestamp: data.timestamp,
          temperature_c: data.temperature_c.value,
          pressure_hpa: data.pressure_hpa.value,
          humidity_pct: data.humidity_pct.value,
          anomaly_score_pct: data.anomaly_score_pct,
          is_anomaly: data.is_anomaly ?? false,
          fault_type: data.fault_type,
          suggested_temperature_c: data.suggested_values?.temperature_c,
          suggested_pressure_hpa: data.suggested_values?.pressure_hpa,
          suggested_humidity_pct: data.suggested_values?.humidity_pct,
          health_status: data.sensor_health_status,
          source: data.source,
        };
        // A mode switch must start a fresh chart. Never draw replay history
        // beside real-time data (or vice versa), even during a race between
        // the status poll and the next reading poll.
        const existingSource = previous.points.at(-1)?.source;
        if (existingSource && point.source && existingSource !== point.source) {
          return { ...previous, points: [point] };
        }
        const byTimestamp = new Map(previous.points.map((item) => [item.timestamp, item]));
        byTimestamp.set(point.timestamp, point);
        const visibleHours = trendHoursRef.current;
        const cutoff = new Date(new Date(point.timestamp).getTime() - visibleHours * 60 * 60 * 1000).getTime();
        return {
          ...previous,
          hours: visibleHours,
          points: Array.from(byTimestamp.values())
            .filter((item) => new Date(item.timestamp).getTime() >= cutoff)
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()),
        };
      });

      // Append to in-memory telemetry history (bounded to 150 entries)
      setTelemetryHistory((prev) => {
        let statusToken: 'NORMAL' | 'WARNING' | 'CRITICAL' | 'OFFLINE' = 'NORMAL';
        if (data.risk_level === 'critical' || data.sensor_health_status === 'CRITICAL') statusToken = 'CRITICAL';
        else if (data.risk_level === 'high' || data.risk_level === 'medium' || data.sensor_health_status === 'WARNING') statusToken = 'WARNING';
        else if (data.sensor_health_status === 'OFFLINE') statusToken = 'OFFLINE';

        const newRecord: TelemetryHistoryRecord = {
          id: `${data.station_id}-${data.timestamp}-${Math.random().toString(36).slice(2, 6)}`,
          timestamp: data.timestamp,
          temperature_c: data.temperature_c.value,
          pressure_hpa: data.pressure_hpa.value,
          humidity_pct: data.humidity_pct.value,
          status: statusToken,
        };

        // Avoid exact duplicate timestamp at index 0
        if (prev.length > 0 && prev[0].timestamp === data.timestamp && prev[0].temperature_c === data.temperature_c.value) {
          return prev;
        }

        return [newRecord, ...prev].slice(0, 150);
      });
    } catch (err) {
      if (activeStationIdRef.current !== targetStationId) return;
      // Retain last known valid reading while setting error indicator
      setReadingError(formatUserErrorMessage(err, 'Unable to load current sensor readings.'));
      // Uvicorn can accept the frontend request a moment before the first
      // concurrent Open-Meteo bootstrap finishes. A 404 here means "not
      // fetched yet", not a broken station; retry quickly once instead of
      // leaving the dashboard empty until the normal 30-minute live poll.
      if (err instanceof ApiError && err.status === 404) {
        window.setTimeout(() => {
          if (activeStationIdRef.current === targetStationId) fetchReading();
        }, 3000);
      }
    } finally {
      if (activeStationIdRef.current === targetStationId) {
        setIsLoadingReading(false);
      }
      isFetchingReadingRef.current = false;
    }
  }, [stationId]);

  // 2. Fetch Trends
  const fetchTrends = useCallback(async (hours: number = trendHours) => {
    if (!stationId) return;
    const targetStationId = stationId;
    setIsLoadingTrends(true);
    try {
      const data = await trendsService.getTrends(targetStationId, hours);
      if (activeStationIdRef.current !== targetStationId) return;
      setTrends(data);
      setTrendsError(null);

      // Pre-seed telemetry history from trend points on initial load if history is empty
      setTelemetryHistory((prev) => {
        if (prev.length > 0) return prev;
        const initialFromTrends: TelemetryHistoryRecord[] = (data.points || [])
          .slice(-25)
          .reverse()
          .map((pt, idx) => {
            const isAnomaly = (pt.anomaly_score_pct || 0) > 75;
            return {
              id: `hist-${idx}-${pt.timestamp}`,
              timestamp: pt.timestamp,
              temperature_c: pt.temperature_c,
              pressure_hpa: pt.pressure_hpa,
              humidity_pct: pt.humidity_pct,
              status: isAnomaly ? 'CRITICAL' : 'NORMAL',
            };
          });
        return initialFromTrends;
      });
    } catch (err) {
      if (activeStationIdRef.current !== targetStationId) return;
      setTrendsError(formatUserErrorMessage(err, 'Unable to load sensor trends.'));
    } finally {
      if (activeStationIdRef.current === targetStationId) {
        setIsLoadingTrends(false);
      }
    }
  }, [stationId, trendHours]);

  // 3. Fetch Anomalies
  const fetchAnomalies = useCallback(async () => {
    if (!stationId) return;
    const targetStationId = stationId;
    setIsLoadingAnomalies(true);
    try {
      const [latest, recent] = await Promise.all([
        anomalyService.getLatestAnomaly(targetStationId),
        anomalyService.getRecentAnomalies(targetStationId, 5),
      ]);
      if (activeStationIdRef.current !== targetStationId) return;
      setLatestAnomaly(latest);
      setRecentAnomalies(recent);
      setAnomaliesError(null);
    } catch (err) {
      if (activeStationIdRef.current !== targetStationId) return;
      setAnomaliesError(formatUserErrorMessage(err, 'Unable to load anomaly summary.'));
    } finally {
      if (activeStationIdRef.current === targetStationId) {
        setIsLoadingAnomalies(false);
      }
    }
  }, [stationId]);

  // 4. Fetch Sensor Health [API: GET /api/sensor-health?station_id=...]
  const fetchHealth = useCallback(async () => {
    if (!stationId) return;
    if (isFetchingHealthRef.current) return;
    isFetchingHealthRef.current = true;
    const targetStationId = stationId;

    try {
      const healthData = await sensorHealthService.getSensorHealth(targetStationId);
      if (activeStationIdRef.current !== targetStationId) return;
      setSensorHealth(healthData);
      setHealthError(null);
    } catch (err) {
      if (activeStationIdRef.current !== targetStationId) return;
      setHealthError(formatUserErrorMessage(err, 'Unable to load sensor health.'));
    } finally {
      if (activeStationIdRef.current === targetStationId) {
        setIsLoadingHealth(false);
      }
      isFetchingHealthRef.current = false;
    }
  }, [stationId]);

  const fetchStreamStatus = useCallback(async () => {
    try {
      setStreamStatus(await systemStatusService.get());
    } catch {
      // Keep the last known mode: a control-plane hiccup must not make
      // the graph discard timestamped telemetry already on screen.
    }
  }, []);

  // 5. Refresh all sections
  const refreshAll = useCallback(async () => {
    await Promise.all([fetchReading(), fetchTrends(trendHours), fetchAnomalies(), fetchHealth()]);
  }, [fetchReading, fetchTrends, fetchAnomalies, fetchHealth, trendHours]);

  // Toggle pause frontend polling
  const togglePause = useCallback(() => {
    setIsPaused((prev) => !prev);
  }, []);

  // Initialize and stationId change effect
  useEffect(() => {
    if (!stationId) {
      setCurrentReading(null);
      setTrends(null);
      setLatestAnomaly(null);
      setRecentAnomalies([]);
      setTelemetryHistory([]);
      setSensorHealth(null);
      setIsLoadingReading(false);
      setIsLoadingTrends(false);
      setIsLoadingAnomalies(false);
      setIsLoadingHealth(false);
      return;
    }

    // Immediately clear previous station's data and set loading
    setCurrentReading(null);
    setTrends(null);
    setLatestAnomaly(null);
    setRecentAnomalies([]);
    setSensorHealth(null);
    setTelemetryHistory([]);

    setIsLoadingReading(true);
    setIsLoadingTrends(true);
    setIsLoadingAnomalies(true);
    setIsLoadingHealth(true);
    setReadingError(null);
    setTrendsError(null);
    setAnomaliesError(null);
    setHealthError(null);

    const live = priorStreamModeRef.current === 'live';
    if (live) {
      systemStatusService.refreshLive().finally(fetchReading);
    } else {
      fetchReading();
    }
    fetchTrends(trendHoursRef.current);
    fetchAnomalies();
    fetchHealth();
  }, [stationId, fetchReading, fetchTrends, fetchAnomalies, fetchHealth]);

  useEffect(() => {
    if (!stationId) return;
    fetchTrends(trendHours);
  }, [stationId, trendHours, fetchTrends]);

  // Mode is deliberately light-weight and checked each second so a
  // user-triggered replay changes data cadence promptly. Sensor data
  // itself remains 30-min live / 1-sec replay.
  useEffect(() => {
    fetchStreamStatus();
    modeTimerRef.current = window.setInterval(fetchStreamStatus, 1000);
    return () => {
      if (modeTimerRef.current !== null) clearInterval(modeTimerRef.current);
    };
  }, [fetchStreamStatus]);

  // A replay and live stream use different timelines. Clear the visual
  // timeline and fetch the new source's retained window at the transition,
  // rather than appending it to the old source's points.
  useEffect(() => {
    const previousMode = priorStreamModeRef.current;
    priorStreamModeRef.current = streamStatus.mode;
    if (!stationId || previousMode === streamStatus.mode) return;

    setTrends(null);
    setCurrentReading(null);
    setSensorHealth(null);
    setLatestAnomaly(null);
    setRecentAnomalies([]);
    setTelemetryHistory([]);
    setLastUpdated(null);
    setIsLoadingReading(true);
    setIsLoadingTrends(true);
    setIsLoadingHealth(true);
    setIsLoadingAnomalies(true);
    fetchTrends(trendHoursRef.current);
    fetchAnomalies();
    if (streamStatus.mode === 'live') {
      systemStatusService.refreshLive().finally(fetchReading);
    } else {
      fetchReading();
    }
    fetchHealth();
  }, [streamStatus.mode, stationId, fetchTrends, fetchReading, fetchHealth, fetchAnomalies]);

  useEffect(() => {
    const onRefresh = () => {
      if (!activeStationIdRef.current) return;
      refreshAll();
    };
    window.addEventListener(TELEMETRY_REFRESH_EVENT, onRefresh);
    return () => window.removeEventListener(TELEMETRY_REFRESH_EVENT, onRefresh);
  }, [refreshAll]);

  // Single Centralized Polling loop for telemetry (3–5s)
  useEffect(() => {
    if (!autoPoll || !stationId || isPaused) {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    const effectivePollingMs = streamStatus.mode === 'replay' ? 1000 : pollingIntervalMs;
    
    let ws: WebSocket | null = null;
    if (!isMockMode()) {
      const wsUrl = API_CONFIG.baseUrl.replace(/^http/, 'ws') + '/api/ws';
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "TICK" && !isPausedRef.current) {
            fetchReading();
            fetchHealth();
          }
        } catch (e) {}
      };
    } else {
      timerRef.current = window.setInterval(() => {
        if (!isPausedRef.current) {
          fetchReading();
          fetchHealth();
        }
      }, effectivePollingMs);
    }

    return () => {
      if (ws) ws.close();
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [autoPoll, stationId, isPaused, pollingIntervalMs, streamStatus.mode, fetchReading, fetchHealth]);

  // Calculate freshness state
  // lastUpdated means the backend successfully answered a reading request;
  // do not label valid hourly Open-Meteo data stale after 60 seconds.
  const freshness = calculateFreshness(
    lastUpdated,
    isPaused,
    streamStatus.mode === 'replay' ? 2 : streamStatus.live_poll_interval_seconds
  );
  const isDelayed = freshness.status === 'DATA DELAYED';
  const isStale = freshness.status === 'DATA STALE';
  const staleStatusText: 'LIVE' | 'DATA DELAYED' | 'DATA STALE' = isStale
    ? 'DATA STALE'
    : isDelayed
    ? 'DATA DELAYED'
    : 'LIVE';
  const replaySeconds = streamStatus.replay_step_seconds ?? 2;
  const pollStatusText = streamStatus.mode === 'replay'
    ? `REPLAY · 1H / ${replaySeconds}S`
    : freshness.label;

  return {
    currentReading,
    trends,
    latestAnomaly,
    recentAnomalies,
    telemetryHistory,
    sensorHealth,
    isLoadingReading,
    isLoadingTrends,
    isLoadingAnomalies,
    isLoadingHealth,
    readingError,
    trendsError,
    anomaliesError,
    healthError,
    lastUpdated,
    isStale,
    isDelayed,
    staleStatusText,
    freshness,
    pollStatusText,
    streamMode: streamStatus.mode,
    isPaused,
    setIsPaused,
    togglePause,
    refreshAll,
    refreshReading: fetchReading,
    refreshTrends: fetchTrends,
    refreshAnomalies: fetchAnomalies,
    refreshHealth: fetchHealth,
    syncStreamStatus: fetchStreamStatus,
  };
}
