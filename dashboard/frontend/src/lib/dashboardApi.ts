export type MissionStatus =
  | "idle"
  | "localizing"
  | "navigating"
  | "terrain_approach"
  | "terrain_traversal"
  | "resuming"
  | "completed"
  | "failed"
  | "stopped";

export type ControlOwner =
  | "none"
  | "manual"
  | "mission"
  | "navigation"
  | "stair_approach"
  | "stair_rl"
  | "recovery"
  | "emergency";

export type TerrainType =
  | "unknown"
  | "flat"
  | "stair_approach"
  | "stair_ascent"
  | "upper_platform";

export interface Pose2D {
  x: number;
  y: number;
  yaw: number;
}

export interface RobotPose extends Pose2D {
  frame: string;
  valid: boolean;
  covariance: number[];
}

export interface MissionStage {
  id: string;
  label: string;

  status:
    | "waiting"
    | "active"
    | "complete"
    | "failed";
}

export interface DashboardState {
  source: "preview" | "ros";

  robot: {
    connected: boolean;
    robot_name: string;
    mode: string | null;
    battery_percent: number | null;
    control_owner: ControlOwner;
  };

  navigation: {
    localization:
      | "offline"
      | "localizing"
      | "valid"
      | "degraded";

    pose: RobotPose;

    goal: Pose2D | null;

    global_path: Pose2D[];
    local_path: Pose2D[];

    distance_remaining_m: number | null;
    eta_seconds: number | null;
  };

  mission: {
    id: string | null;
    name: string | null;
    status: MissionStatus;
    stage_index: number;
    progress: number;
    stages: MissionStage[];
  };

  terrain: {
    current: TerrainType;
    detected: boolean;
    confidence: number | null;

    stair: {
      rise_m: number | null;
      tread_m: number | null;
      steps: number | null;
      alignment_error_deg: number | null;
      distance_m: number | null;
    } | null;
  };

  safety: {
    ready: boolean;
    stable: boolean | null;
    fall_detected: boolean | null;
    command_timeout: boolean | null;
    reason: string | null;
  };

  events: {
    sequence: number;
    kind: string;
    message: string;
    source: string;
  }[];
}

export interface StateMessage {
  type: "state";
  version: number;
  data: DashboardState;
}


function backendHost() {
  return window.location.hostname || "127.0.0.1";
}


export function backendHttpUrl(path: string) {
  return `http://${backendHost()}:8000${path}`;
}


export function backendWebSocketUrl() {
  return `ws://${backendHost()}:8000/ws/state`;
}


export async function fetchDashboardState() {
  const response = await fetch(
    backendHttpUrl("/api/state"),
  );

  if (!response.ok) {
    throw new Error(
      `Dashboard state request failed: ${response.status}`,
    );
  }

  return (
    await response.json()
  ) as DashboardState;
}


async function post(path: string) {
  const response = await fetch(
    backendHttpUrl(path),
    {
      method: "POST",
    },
  );

  if (!response.ok) {
    throw new Error(
      `Dashboard command failed: ${response.status}`,
    );
  }
}


export async function startPreviewMission() {
  await post(
    "/api/mission/preview/start",
  );
}


export async function stopPreviewMission() {
  await post(
    "/api/mission/preview/stop",
  );
}
