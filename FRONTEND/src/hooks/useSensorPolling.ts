import { useState, useEffect, useCallback, useRef } from 'react';
import { SystemStatusSummary } from '../types';
import { anomalyService } from '../services/anomalyService';
import { systemStatusService } from '../services/systemStatusService';
import { requestTelemetryRefresh } from '../utils/refreshEvents';
import { isMockMode, API_CONFIG } from '../config/api.config';

export interface UseSensorPollingResult {
  readings: never[];
  systemStatus: SystemStatusSummary | null;
  loading: boolean;
  error: string | null;
  isPolling: boolean;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

/**
 * Header network-health poll. Station telemetry lives in useDashboardData;
 * this hook only keeps the real aggregate status badge honest.
 */
export function useSensorPolling(autoPoll: boolean = false): UseSensorPollingResult {
  const [systemStatus, setSystemStatus] = useState<SystemStatusSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timerRef = useRef<number | null>(null);

  const fetchData = useCallback(async (alsoRefreshLive = false) => {
    try {
      setError(null);
      if (alsoRefreshLive && !isMockMode()) {
        try {
          await systemStatusService.refreshLive();
        } catch {
          // Network status should still update even if Open-Meteo refresh fails.
        }
      }
      const statusData = await anomalyService.getSystemStatus();
      setSystemStatus(statusData);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch network status.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData(false);
    
    let ws: WebSocket | null = null;
    
    if (autoPoll) {
      if (!isMockMode()) {
        const wsUrl = API_CONFIG.baseUrl.replace(/^http/, 'ws') + '/api/ws';
        ws = new WebSocket(wsUrl);
        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "TICK") {
              fetchData(false);
            }
          } catch (e) {}
        };
      } else {
        timerRef.current = window.setInterval(() => {
          fetchData(false);
        }, 5000);
      }
    }
    
    return () => {
      if (ws) ws.close();
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
    };
  }, [fetchData, autoPoll]);

  const refresh = useCallback(async () => {
    await fetchData(true);
    requestTelemetryRefresh();
  }, [fetchData]);

  return {
    readings: [],
    systemStatus,
    loading,
    error,
    isPolling: autoPoll,
    lastUpdated,
    refresh,
  };
}
