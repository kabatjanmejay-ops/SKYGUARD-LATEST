/**
 * SkyGuard AI — Runtime API Response Validators & Type Guards
 *
 * Protects frontend React components and charts from malformed or incomplete
 * backend responses when real FastAPI endpoints are integrated.
 *
 * Rules:
 * - No `any` is allowed anywhere in this file.
 * - All backend field normalization happens here (not in components or hooks).
 * - Invalid required fields must throw ApiError.validationError — never silently default.
 */

import {
  Station,
  CurrentSensorReading,
  TrendsResponse,
  TrendPoint,
  AnomalyWindow,
  LatestAnomaly,
  RecentAnomalyItem,
  AnomalyExplanation,
  ExplanationFeature,
  SensorHealth,
  InjectAnomalyResponse,
  MaintenanceTicketResponse,
  RepairSensorResponse,
  AnomalySeverity,
  AnomalyType,
  SystemStatusSummary,
  SystemOverallStatus,
} from '../types';
import { ApiError } from './apiError';

// ---------------------------------------------------------------------------
// Allowed constant sets
// ---------------------------------------------------------------------------

const VALID_SEVERITIES: readonly AnomalySeverity[] = ['low', 'medium', 'high', 'critical'];
const VALID_ANOMALY_TYPES: readonly AnomalyType[] = [
  'spike',
  'frozen_value',
  'drift',
  'dropout',
  'sensor_fail_low',
  'multivariate_inconsistency',
];

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

function isObject(val: unknown): val is Record<string, unknown> {
  return typeof val === 'object' && val !== null && !Array.isArray(val);
}

function optionalObservedValues(value: unknown): Record<string, number | null> | undefined {
  if (!isObject(value)) return undefined;
  const readings: Record<string, number | null> = {};
  for (const [key, reading] of Object.entries(value)) {
    if (typeof reading === 'number' || reading === null) readings[key] = reading;
  }
  return readings;
}

function optionalAffectedParameters(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.filter((parameter): parameter is string => typeof parameter === 'string');
}

// ---------------------------------------------------------------------------
// 1. Validate GET /api/stations response
//    Backend only returns: NORMAL | WARNING | OFFLINE (no CRITICAL from this endpoint)
//    CRITICAL is retained in UI types but must not be accepted from the API response.
// ---------------------------------------------------------------------------
export function validateStations(data: unknown): Station[] {
  if (!Array.isArray(data)) {
    throw ApiError.validationError('Stations response must be an array.');
  }

  const allowedStatuses = ['NORMAL', 'WARNING', 'CRITICAL', 'OFFLINE'] as const;
  type AllowedStatus = (typeof allowedStatuses)[number];

  return data.map((item, idx) => {
    if (!isObject(item)) {
      throw ApiError.validationError(`Station at index ${idx} is not an object.`);
    }

    if (
      typeof item.station_id !== 'string' ||
      typeof item.name !== 'string' ||
      typeof item.lat !== 'number' ||
      typeof item.lon !== 'number'
    ) {
      throw ApiError.validationError(
        `Station at index ${idx} is missing required fields (station_id, name, lat, lon).`
      );
    }

    if (
      typeof item.status !== 'string' ||
      !allowedStatuses.includes(item.status as AllowedStatus)
    ) {
      throw ApiError.validationError(
        `Station at index ${idx} has an invalid or unrecognised status: "${String(item.status)}".`
      );
    }

    const status = item.status as Station['status'];

    return {
      station_id: String(item.station_id),
      name: String(item.name),
      lat: item.lat,
      lon: item.lon,
      status,
      elevation_m: typeof item.elevation_m === 'number' ? item.elevation_m : undefined,
      region: typeof item.region === 'string' ? item.region : undefined,
    };
  });
}

// ---------------------------------------------------------------------------
// 2. Validate GET /api/current-reading response
// ---------------------------------------------------------------------------
export function validateCurrentReading(data: unknown): CurrentSensorReading {
  if (!isObject(data)) {
    throw ApiError.validationError('Current reading response must be an object.');
  }

  if (typeof data.station_id !== 'string' || data.station_id.trim() === '') {
    throw ApiError.validationError('Current reading missing valid station_id string.');
  }

  if (typeof data.timestamp !== 'string' || data.timestamp.trim() === '') {
    throw ApiError.validationError('Current reading missing valid timestamp string.');
  }

  const validateMetric = (metric: unknown, name: string) => {
    if (
      !isObject(metric) ||
      typeof metric.value !== 'number' ||
      typeof metric.normal_min !== 'number' ||
      typeof metric.normal_max !== 'number'
    ) {
      throw ApiError.validationError(
        `Current reading metric "${name}" is missing required numeric fields (value, normal_min, normal_max).`
      );
    }
    return {
      value: Number(metric.value),
      normal_min: Number(metric.normal_min),
      normal_max: Number(metric.normal_max),
    };
  };

  if (typeof data.anomaly_score_pct !== 'number') {
    throw ApiError.validationError('Current reading missing numeric anomaly_score_pct.');
  }

  const validRiskLevels = ['low', 'medium', 'high', 'critical'] as const;
  type RiskLevelValue = (typeof validRiskLevels)[number];

  if (
    typeof data.risk_level !== 'string' ||
    !validRiskLevels.includes(data.risk_level as RiskLevelValue)
  ) {
    throw ApiError.validationError(
      `Current reading has invalid risk_level: "${String(data.risk_level)}".`
    );
  }
  const riskLevel = data.risk_level as RiskLevelValue;

  if (typeof data.sensor_health_pct !== 'number') {
    throw ApiError.validationError('Current reading missing numeric sensor_health_pct.');
  }

  const validHealthStatuses = ['HEALTHY', 'WARNING', 'CRITICAL', 'OFFLINE'] as const;
  type HealthStatus = (typeof validHealthStatuses)[number];

  if (
    typeof data.sensor_health_status !== 'string' ||
    !validHealthStatuses.includes(data.sensor_health_status as HealthStatus)
  ) {
    throw ApiError.validationError(
      `Current reading has invalid sensor_health_status: "${String(data.sensor_health_status)}".`
    );
  }
  const healthStatus = data.sensor_health_status as HealthStatus;

  return {
    station_id: String(data.station_id),
    timestamp: String(data.timestamp),
    temperature_c: validateMetric(data.temperature_c, 'temperature_c'),
    pressure_hpa: validateMetric(data.pressure_hpa, 'pressure_hpa'),
    humidity_pct: validateMetric(data.humidity_pct, 'humidity_pct'),
    anomaly_score_pct: Number(data.anomaly_score_pct),
    risk_level: riskLevel,
    sensor_health_pct: Number(data.sensor_health_pct),
    sensor_health_status: healthStatus,
    ...(typeof data.is_anomaly === 'boolean' ? { is_anomaly: data.is_anomaly } : {}),
    ...(typeof data.fault_type === 'string' && (VALID_ANOMALY_TYPES as readonly string[]).includes(data.fault_type)
      ? { fault_type: data.fault_type as AnomalyType } : {}),
    ...(isObject(data.suggested_values) ? { suggested_values: Object.fromEntries(Object.entries(data.suggested_values).filter(([, value]) => typeof value === 'number')) as Record<string, number> } : {}),
    ...(data.source === 'live' || data.source === 'replay' ? { source: data.source } : {}),
  };
}

// ---------------------------------------------------------------------------
// 3. Validate GET /api/trends response
// ---------------------------------------------------------------------------
export function validateTrends(data: unknown, fallbackHours: number = 6): TrendsResponse {
  if (!isObject(data)) {
    throw ApiError.validationError('Trends response must be an object.');
  }

  if (typeof data.station_id !== 'string' || data.station_id.trim() === '') {
    throw ApiError.validationError('Trends response missing valid station_id string.');
  }

  if (!Array.isArray(data.points)) {
    throw ApiError.validationError('Trends response points must be an array.');
  }

  const parsedPoints: TrendPoint[] = data.points.map((pt: unknown, idx: number) => {
    if (!isObject(pt)) {
      throw ApiError.validationError(`Trend point at index ${idx} is not an object.`);
    }

    if (typeof pt.timestamp !== 'string' || pt.timestamp.trim() === '') {
      throw ApiError.validationError(`Trend point at index ${idx} is missing valid timestamp string.`);
    }
    if (Number.isNaN(new Date(pt.timestamp).getTime())) {
      throw ApiError.validationError(`Trend point at index ${idx} has an invalid timestamp.`);
    }

    if (
      typeof pt.temperature_c !== 'number' ||
      typeof pt.pressure_hpa !== 'number' ||
      typeof pt.humidity_pct !== 'number'
    ) {
      throw ApiError.validationError(
        `Trend point at index ${idx} is missing valid numeric metrics (temperature_c, pressure_hpa, humidity_pct).`
      );
    }

    return {
      timestamp: String(pt.timestamp),
      temperature_c: Number(pt.temperature_c),
      pressure_hpa: Number(pt.pressure_hpa),
      humidity_pct: Number(pt.humidity_pct),
      ...(typeof pt.anomaly_score_pct === 'number'
        ? { anomaly_score_pct: Number(pt.anomaly_score_pct) }
        : {}),
      ...(typeof pt.is_anomaly === 'boolean' ? { is_anomaly: pt.is_anomaly } : {}),
      ...(typeof pt.fault_type === 'string' && (VALID_ANOMALY_TYPES as readonly string[]).includes(pt.fault_type)
        ? { fault_type: pt.fault_type as AnomalyType } : {}),
      ...(typeof pt.severity === 'string' && VALID_SEVERITIES.includes(pt.severity as AnomalySeverity)
        ? { severity: pt.severity as AnomalySeverity } : {}),
      ...(typeof pt.suggested_temperature_c === 'number' ? { suggested_temperature_c: pt.suggested_temperature_c } : {}),
      ...(typeof pt.suggested_pressure_hpa === 'number' ? { suggested_pressure_hpa: pt.suggested_pressure_hpa } : {}),
      ...(typeof pt.suggested_humidity_pct === 'number' ? { suggested_humidity_pct: pt.suggested_humidity_pct } : {}),
      ...(typeof pt.health_status === 'string' ? { health_status: pt.health_status as TrendPoint['health_status'] } : {}),
      ...(pt.source === 'live' || pt.source === 'replay' ? { source: pt.source } : {}),
    };
  });

  // Legacy development servers wrote several "live" samples per
  // minute. Live Open-Meteo is intentionally a 30-minute cadence, so
  // retain only the latest sample in each 30-minute live bucket. Replay
  // points retain every original hourly timestamp unchanged.
  const pointMap = new Map<string, TrendPoint>();
  for (const point of parsedPoints) {
    const timestamp = new Date(point.timestamp).getTime();
    const key = point.source === 'live'
      ? `live-${Math.floor(timestamp / (30 * 60 * 1000))}`
      : `timestamp-${point.timestamp}`;
    pointMap.set(key, point);
  }
  const points = Array.from(pointMap.values()).sort(
    (left, right) => new Date(left.timestamp).getTime() - new Date(right.timestamp).getTime()
  );

  let anomaly_windows: AnomalyWindow[] | undefined;
  if (Array.isArray(data.anomaly_windows)) {
    anomaly_windows = data.anomaly_windows
      .filter((w: unknown): w is Record<string, unknown> => isObject(w))
      .map((w: Record<string, unknown>) => ({
        start: typeof w.start === 'string' ? w.start : '',
        end: typeof w.end === 'string' ? w.end : '',
        label: typeof w.label === 'string' ? w.label : undefined,
      }))
      .filter((w) => w.start !== '' && w.end !== '');
  }

  const hours = typeof data.hours === 'number' ? data.hours : fallbackHours;

  return {
    station_id: String(data.station_id),
    hours,
    points,
    ...(anomaly_windows && anomaly_windows.length > 0 ? { anomaly_windows } : {}),
  };
}

// ---------------------------------------------------------------------------
// 4. Validate GET /api/anomalies/latest response
// ---------------------------------------------------------------------------
export function validateLatestAnomaly(data: unknown, expectedStationId?: string): LatestAnomaly | null {
  if (data === null || data === undefined) {
    return null;
  }

  if (!isObject(data)) {
    throw ApiError.validationError('Latest anomaly response must be an object or null.');
  }

  if (typeof data.anomaly_id !== 'string' || data.anomaly_id.trim() === '') {
    throw ApiError.validationError('Latest anomaly missing valid anomaly_id string.');
  }

  if (typeof data.station_id !== 'string' || data.station_id.trim() === '') {
    throw ApiError.validationError('Latest anomaly missing valid station_id string.');
  }

  if (expectedStationId && data.station_id !== expectedStationId) {
    throw ApiError.validationError(
      `Latest anomaly station_id mismatch: expected "${expectedStationId}", got "${data.station_id}".`
    );
  }

  if (typeof data.timestamp !== 'string' || data.timestamp.trim() === '') {
    throw ApiError.validationError('Latest anomaly missing valid timestamp string.');
  }

  if (typeof data.anomaly_score_pct !== 'number') {
    throw ApiError.validationError('Latest anomaly missing numeric anomaly_score_pct.');
  }

  if (
    typeof data.severity !== 'string' ||
    !(VALID_SEVERITIES as readonly string[]).includes(data.severity)
  ) {
    throw ApiError.validationError(
      `Latest anomaly has invalid severity: "${String(data.severity)}".`
    );
  }
  const severity = data.severity as AnomalySeverity;

  if (
    typeof data.type !== 'string' ||
    !(VALID_ANOMALY_TYPES as readonly string[]).includes(data.type)
  ) {
    throw ApiError.validationError(
      `Latest anomaly has invalid type: "${String(data.type)}".`
    );
  }
  const type = data.type as AnomalyType;

  if (typeof data.root_cause !== 'string' || data.root_cause.trim() === '') {
    throw ApiError.validationError('Latest anomaly missing valid root_cause string.');
  }

  const description =
    typeof data.description === 'string' && data.description.trim() !== ''
      ? data.description
      : `${data.root_cause} detected at ${data.station_id}.`;

  // Normalise suggested_values: backend returns a dict of string→number or {}
  let suggested_values: Record<string, number> | undefined;
  if (isObject(data.suggested_values)) {
    suggested_values = {};
    for (const [k, v] of Object.entries(data.suggested_values)) {
      if (typeof v === 'number') {
        suggested_values[k] = v;
      }
    }
  }
  const observed_values = optionalObservedValues(data.observed_values);
  const affected_parameters = optionalAffectedParameters(data.affected_parameters);

  return {
    anomaly_id: String(data.anomaly_id),
    timestamp: String(data.timestamp),
    station_id: String(data.station_id),
    anomaly_score_pct: Number(data.anomaly_score_pct),
    severity,
    type,
    root_cause: String(data.root_cause),
    description,
    ...(suggested_values !== undefined ? { suggested_values } : {}),
    ...(observed_values !== undefined ? { observed_values } : {}),
    ...(affected_parameters !== undefined ? { affected_parameters } : {}),
  };
}

// ---------------------------------------------------------------------------
// 5. Validate GET /api/anomalies/recent response
//    Normalization: backend returns `score_pct` → frontend field `anomaly_score_pct`
//                   backend returns `label`     → frontend field `root_cause`
// ---------------------------------------------------------------------------
export function validateRecentAnomalies(data: unknown, expectedStationId?: string): RecentAnomalyItem[] {
  if (!Array.isArray(data)) {
    throw ApiError.validationError('Recent anomalies response must be an array.');
  }

  return data.map((item: unknown, idx) => {
    if (!isObject(item)) {
      throw ApiError.validationError(`Recent anomaly at index ${idx} is not an object.`);
    }

    if (typeof item.anomaly_id !== 'string' || item.anomaly_id.trim() === '') {
      throw ApiError.validationError(
        `Recent anomaly at index ${idx} missing valid anomaly_id string.`
      );
    }

    if (typeof item.station_id !== 'string' || item.station_id.trim() === '') {
      throw ApiError.validationError(
        `Recent anomaly at index ${idx} missing valid station_id string.`
      );
    }

    if (expectedStationId && item.station_id !== expectedStationId) {
      throw ApiError.validationError(
        `Recent anomaly at index ${idx} station_id mismatch: expected "${expectedStationId}", got "${item.station_id}".`
      );
    }

    if (typeof item.timestamp !== 'string' || item.timestamp.trim() === '') {
      throw ApiError.validationError(
        `Recent anomaly at index ${idx} missing valid timestamp string.`
      );
    }

    // Backend sends `score_pct`; normalise to `anomaly_score_pct`
    const rawScore = item.score_pct !== undefined ? item.score_pct : item.anomaly_score_pct;
    if (typeof rawScore !== 'number') {
      throw ApiError.validationError(
        `Recent anomaly at index ${idx} missing numeric score_pct / anomaly_score_pct.`
      );
    }
    const anomaly_score_pct = Number(rawScore);

    if (
      typeof item.severity !== 'string' ||
      !(VALID_SEVERITIES as readonly string[]).includes(item.severity)
    ) {
      throw ApiError.validationError(
        `Recent anomaly at index ${idx} has invalid severity: "${String(item.severity)}".`
      );
    }
    const severity = item.severity as AnomalySeverity;

    if (
      typeof item.type !== 'string' ||
      !(VALID_ANOMALY_TYPES as readonly string[]).includes(item.type)
    ) {
      throw ApiError.validationError(
        `Recent anomaly at index ${idx} has invalid type: "${String(item.type)}".`
      );
    }
    const type = item.type as AnomalyType;

    // Backend sends `label`; normalise to `root_cause`
    const rawLabel = item.label !== undefined ? item.label : item.root_cause;
    if (typeof rawLabel !== 'string' || rawLabel.trim() === '') {
      throw ApiError.validationError(
        `Recent anomaly at index ${idx} missing valid label / root_cause string.`
      );
    }
    const root_cause = String(rawLabel);

    const description =
      typeof item.description === 'string' && item.description.trim() !== ''
        ? item.description
        : `${root_cause} detected at ${item.station_id}.`;

    // Normalise suggested_values
    let suggested_values: Record<string, number> | undefined;
    if (isObject(item.suggested_values)) {
      suggested_values = {};
      for (const [k, v] of Object.entries(item.suggested_values)) {
        if (typeof v === 'number') {
          suggested_values[k] = v;
        }
      }
    }
    const observed_values = optionalObservedValues(item.observed_values);
    const affected_parameters = optionalAffectedParameters(item.affected_parameters);

    return {
      anomaly_id: String(item.anomaly_id),
      timestamp: String(item.timestamp),
      station_id: String(item.station_id),
      anomaly_score_pct,
      severity,
      type,
      root_cause,
      description,
      ...(suggested_values !== undefined ? { suggested_values } : {}),
      ...(observed_values !== undefined ? { observed_values } : {}),
      ...(affected_parameters !== undefined ? { affected_parameters } : {}),
    };
  });
}

// ---------------------------------------------------------------------------
// 6. Validate GET /api/explain/{anomaly_id} response
// ---------------------------------------------------------------------------
export function validateAnomalyExplanation(
  data: unknown,
  expectedAnomalyId?: string
): AnomalyExplanation {
  if (!isObject(data)) {
    throw ApiError.validationError('Anomaly explanation response must be an object.');
  }

  if (typeof data.anomaly_id !== 'string' || data.anomaly_id.trim() === '') {
    throw ApiError.validationError('Anomaly explanation missing valid anomaly_id string.');
  }

  if (expectedAnomalyId && data.anomaly_id !== expectedAnomalyId) {
    throw ApiError.validationError(
      `Anomaly explanation anomaly_id mismatch: expected "${expectedAnomalyId}", got "${data.anomaly_id}".`
    );
  }

  if (!Array.isArray(data.features)) {
    throw ApiError.validationError('Anomaly explanation features must be an array.');
  }

  const features: ExplanationFeature[] = data.features.map((f: unknown, idx: number) => {
    if (!isObject(f)) {
      throw ApiError.validationError(`Explanation feature at index ${idx} is not an object.`);
    }

    if (typeof f.name !== 'string' || f.name.trim() === '') {
      throw ApiError.validationError(`Explanation feature at index ${idx} missing valid name string.`);
    }

    if (typeof f.impact !== 'number' || isNaN(f.impact)) {
      throw ApiError.validationError(
        `Explanation feature "${String(f.name)}" at index ${idx} missing valid numeric impact.`
      );
    }

    return {
      name: String(f.name),
      impact: Number(f.impact),
    };
  });

  let likely_faulty_sensors: string[] | undefined;
  if (data.likely_faulty_sensors !== undefined) {
    if (!Array.isArray(data.likely_faulty_sensors)) {
      throw ApiError.validationError('Anomaly explanation likely_faulty_sensors must be an array.');
    }
    likely_faulty_sensors = data.likely_faulty_sensors.map((s: unknown, idx: number) => {
      if (typeof s !== 'string') {
        throw ApiError.validationError(
          `likely_faulty_sensors at index ${idx} must be a string.`
        );
      }
      return String(s);
    });
  }
  const affected_parameters = optionalAffectedParameters(data.affected_parameters);
  const observed_values = optionalObservedValues(data.observed_values);
  let suggested_values: Record<string, number> | undefined;
  if (isObject(data.suggested_values)) {
    suggested_values = {};
    for (const [key, value] of Object.entries(data.suggested_values)) {
      if (typeof value === 'number') suggested_values[key] = value;
    }
  }
  const numericField = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) ? value : undefined;
  const model_confidence_pct = data.model_confidence_pct === null ? null : numericField(data.model_confidence_pct);
  const rule_confidence_pct = numericField(data.rule_confidence_pct);
  const anomaly_score_pct = numericField(data.anomaly_score_pct);
  const fault_type = typeof data.fault_type === 'string' && (VALID_ANOMALY_TYPES as readonly string[]).includes(data.fault_type)
    ? data.fault_type as AnomalyType : undefined;

  return {
    anomaly_id: String(data.anomaly_id),
    features,
    ...(likely_faulty_sensors !== undefined ? { likely_faulty_sensors } : {}),
    ...(affected_parameters !== undefined ? { affected_parameters } : {}),
    ...(observed_values !== undefined ? { observed_values } : {}),
    ...(suggested_values !== undefined ? { suggested_values } : {}),
    ...(model_confidence_pct !== undefined ? { model_confidence_pct } : {}),
    ...(rule_confidence_pct !== undefined ? { rule_confidence_pct } : {}),
    ...(anomaly_score_pct !== undefined ? { anomaly_score_pct } : {}),
    ...(fault_type !== undefined ? { fault_type } : {}),
  };
}

// ---------------------------------------------------------------------------
// 7. Validate GET /api/sensor-health response
//    Normalization: backend returns `health_pct` → frontend field `sensor_health_pct`
//                   backend returns `status`     → frontend field `sensor_health_status`
// ---------------------------------------------------------------------------
export function validateSensorHealth(data: unknown, expectedStationId?: string): SensorHealth {
  if (!isObject(data)) {
    throw ApiError.validationError('Sensor health response must be an object.');
  }

  if (typeof data.station_id !== 'string' || !data.station_id.trim()) {
    throw ApiError.validationError('Sensor health response missing or invalid station_id.');
  }

  if (expectedStationId && data.station_id !== expectedStationId) {
    throw ApiError.validationError(
      `Sensor health station_id mismatch: expected ${expectedStationId}, got ${data.station_id}`
    );
  }

  // Backend sends `health_pct`; mock fixtures may provide `sensor_health_pct`
  const rawHealth =
    typeof data.health_pct === 'number'
      ? data.health_pct
      : typeof data.sensor_health_pct === 'number'
      ? data.sensor_health_pct
      : null;

  if (rawHealth === null || !Number.isFinite(rawHealth) || rawHealth < 0 || rawHealth > 100) {
    throw ApiError.validationError(
      `Invalid health_pct: must be a finite number between 0 and 100, received ${String(data.health_pct ?? data.sensor_health_pct)}.`
    );
  }

  const validHealthStatuses = ['HEALTHY', 'WARNING', 'CRITICAL', 'OFFLINE'] as const;
  type HealthStatusValue = (typeof validHealthStatuses)[number];

  // Backend sends `status`; normalise to `sensor_health_status`
  const rawStatus = data.status ?? data.sensor_health_status;
  if (
    typeof rawStatus !== 'string' ||
    !validHealthStatuses.includes(rawStatus as HealthStatusValue)
  ) {
    throw ApiError.validationError(
      `Invalid sensor health status: received ${String(rawStatus)}.`
    );
  }

  return {
    station_id: data.station_id,
    sensor_health_pct: rawHealth,
    sensor_health_status: rawStatus as HealthStatusValue,
  };
}

// ---------------------------------------------------------------------------
// 8. Validate POST /api/inject-anomaly response
// ---------------------------------------------------------------------------
export function validateInjectAnomalyResponse(data: unknown): InjectAnomalyResponse {
  if (!isObject(data)) {
    throw ApiError.validationError('Inject anomaly response must be an object.');
  }

  if (typeof data.success !== 'boolean') {
    throw ApiError.validationError('Inject anomaly response missing or invalid success flag.');
  }

  if (typeof data.anomaly_id !== 'string' || !data.anomaly_id.trim()) {
    throw ApiError.validationError('Inject anomaly response missing or invalid anomaly_id.');
  }

  if (typeof data.message !== 'string') {
    throw ApiError.validationError('Inject anomaly response missing or invalid message.');
  }

  return {
    success: data.success,
    anomaly_id: data.anomaly_id,
    message: data.message,
  };
}

// ---------------------------------------------------------------------------
// 9. Validate POST /api/maintenance-ticket response
// ---------------------------------------------------------------------------
export function validateMaintenanceTicketResponse(
  data: unknown
): MaintenanceTicketResponse {
  if (!isObject(data)) {
    throw ApiError.validationError('Maintenance ticket response must be an object.');
  }

  if (typeof data.ticket_id !== 'string' || !data.ticket_id.trim()) {
    throw ApiError.validationError('Maintenance ticket response missing or invalid ticket_id.');
  }

  if (typeof data.station_id !== 'string' || !data.station_id.trim()) {
    throw ApiError.validationError('Maintenance ticket response missing or invalid station_id.');
  }

  if (typeof data.issue !== 'string' || !data.issue.trim()) {
    throw ApiError.validationError('Maintenance ticket response missing or invalid issue.');
  }

  if (typeof data.priority !== 'string' || !data.priority.trim()) {
    throw ApiError.validationError('Maintenance ticket response missing or invalid priority.');
  }

  if (typeof data.created_at !== 'string' || !data.created_at.trim()) {
    throw ApiError.validationError('Maintenance ticket response missing or invalid created_at.');
  }

  return {
    ticket_id: data.ticket_id,
    station_id: data.station_id,
    issue: data.issue,
    priority: data.priority,
    created_at: data.created_at,
  };
}

// ---------------------------------------------------------------------------
// 10. Validate POST /api/repair-sensor response
// ---------------------------------------------------------------------------
export function validateRepairSensorResponse(
  data: unknown,
  expectedStationId?: string
): RepairSensorResponse {
  if (!isObject(data)) {
    throw ApiError.validationError('Repair sensor response must be an object.');
  }

  if (typeof data.success !== 'boolean') {
    throw ApiError.validationError('Repair sensor response missing or invalid success flag.');
  }

  if (typeof data.station_id !== 'string' || !data.station_id.trim()) {
    throw ApiError.validationError('Repair sensor response missing or invalid station_id.');
  }

  if (expectedStationId && data.station_id !== expectedStationId) {
    throw ApiError.validationError(
      `Repair sensor station_id mismatch: expected ${expectedStationId}, got ${data.station_id}`
    );
  }

  if (typeof data.status !== 'string' || !data.status.trim()) {
    throw ApiError.validationError('Repair sensor response missing or invalid status.');
  }

  if (typeof data.recovery_active !== 'boolean') {
    throw ApiError.validationError('Repair sensor response missing or invalid recovery_active flag.');
  }

  if (typeof data.message !== 'string') {
    throw ApiError.validationError('Repair sensor response missing or invalid message.');
  }

  return {
    success: data.success,
    station_id: data.station_id,
    status: data.status,
    recovery_active: data.recovery_active,
    message: data.message,
  };
}

const NETWORK_STATUSES: readonly SystemOverallStatus[] = ['NORMAL', 'WARNING', 'CRITICAL', 'OFFLINE'];

export function validateNetworkStatus(data: unknown): SystemStatusSummary {
  if (!isObject(data)) {
    throw ApiError.validationError('Network status response must be an object.');
  }
  if (typeof data.overall_status !== 'string' || !NETWORK_STATUSES.includes(data.overall_status as SystemOverallStatus)) {
    throw ApiError.validationError('Network status has an invalid overall_status.');
  }
  if (typeof data.active_stations_count !== 'number' || typeof data.total_stations_count !== 'number') {
    throw ApiError.validationError('Network status is missing station counts.');
  }
  if (typeof data.active_anomalies_count !== 'number') {
    throw ApiError.validationError('Network status is missing active_anomalies_count.');
  }
  const avg =
    typeof data.avg_sensor_health_pct === 'number' ? data.avg_sensor_health_pct : 100;
  return {
    overall_status: data.overall_status as SystemOverallStatus,
    active_stations_count: data.active_stations_count,
    total_stations_count: data.total_stations_count,
    active_anomalies_count: data.active_anomalies_count,
    avg_sensor_health_pct: avg,
    last_updated: typeof data.last_updated === 'string' ? data.last_updated : new Date().toISOString(),
  };
}
