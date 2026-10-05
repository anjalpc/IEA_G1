import { useState } from "react";

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Circle,
  Crosshair,
  Flag,
  Footprints,
  LocateFixed,
  Map,
  MapPin,
  Navigation,
  Pause,
  Play,
  Route,
  ShieldCheck,
  Square,
  Target,
  Waypoints,
} from "lucide-react";

import { G1HeadMarker } from "../components/G1HeadMarker";
import { useDashboardState } from "../hooks/useDashboardState";
import {
  startPreviewMission,
  stopPreviewMission,
} from "../lib/dashboardApi";

import "./NavigationPage.css";


const missionStages = [
  {
    id: "localize",
    label: "Localize",
    detail: "Establish robot pose",
  },
  {
    id: "navigate",
    label: "Navigate",
    detail: "Regular locomotion",
  },
  {
    id: "terrain",
    label: "Terrain",
    detail: "Detect + align",
  },
  {
    id: "traverse",
    label: "Traverse",
    detail: "Stair RL policy",
  },
  {
    id: "resume",
    label: "Resume",
    detail: "Return to navigation",
  },
  {
    id: "goal",
    label: "Goal",
    detail: "Mission complete",
  },
] as const;


export function NavigationPage() {
  const [selectedTool, setSelectedTool] =
    useState<"inspect" | "goal">("inspect");

  const [lowerDeck, setLowerDeck] =
    useState<
      "status" | "timeline" | "events"
    >("status");

  const [mapLayers, setMapLayers] =
    useState({
      map: true,
      path: true,
      terrain: true,
      fov: true,
    });

  function toggleMapLayer(
    layer: keyof typeof mapLayers,
  ) {
    setMapLayers((current) => ({
      ...current,
      [layer]: !current[layer],
    }));
  }

  const {
    state: dashboardState,
    connection: backendConnection,
  } = useDashboardState();


  const mission =
    dashboardState?.mission;

  const navigation =
    dashboardState?.navigation;

  const terrain =
    dashboardState?.terrain;

  const robot =
    dashboardState?.robot;


  const previewStep =
    mission?.stage_index ?? 0;


  const previewRunning =
    mission !== undefined &&
    ![
      "idle",
      "completed",
      "failed",
      "stopped",
    ].includes(
      mission.status,
    );


  const previewMission =
    mission?.id ===
    "preview-upper-platform";


  const previewProgress =
    mission?.progress ?? 0;


  const controlOwner =
    (
      robot?.control_owner ??
      "none"
    )
      .replaceAll("_", " ")
      .toUpperCase();


  const terrainState =
    (
      terrain?.current ??
      "unknown"
    )
      .replaceAll("_", " ")
      .toUpperCase();


  const localizationStatus =
    (
      navigation?.localization ??
      "offline"
    ).toUpperCase();


  const poseValid =
    navigation?.pose.valid ??
    false;


  const poseX =
    poseValid
      ? navigation?.pose.x.toFixed(2)
      : "--";


  const poseY =
    poseValid
      ? navigation?.pose.y.toFixed(2)
      : "--";


  const poseYaw =
    poseValid
      ? `${(
          (navigation?.pose.yaw ?? 0) *
          (180 / Math.PI)
        ).toFixed(1)}°`
      : "--";


  const missionProgressLabel =
    previewMission
      ? `${Math.round(
          previewProgress,
        )}%`
      : "--";


  const goalDistance =
    navigation?.distance_remaining_m != null
      ? `${navigation.distance_remaining_m.toFixed(2)} m`
      : "--";


  const eta =
    navigation?.eta_seconds != null
      ? `${Math.round(
          navigation.eta_seconds,
        )} s`
      : "--";


  const globalPathLength =
    navigation?.global_path &&
    navigation.global_path.length > 1
      ? navigation.global_path
          .slice(1)
          .reduce(
            (
              total,
              point,
              index,
            ) => {
              const previous =
                navigation.global_path[
                  index
                ];

              return (
                total +
                Math.hypot(
                  point.x - previous.x,
                  point.y - previous.y,
                )
              );
            },
            0,
          )
          .toFixed(2)
      : null;


  const terrainConfidence =
    terrain?.confidence != null
      ? `${Math.round(
          terrain.confidence * 100,
        )}%`
      : "--";


  const stairRise =
    terrain?.stair?.rise_m != null
      ? `${(
          terrain.stair.rise_m *
          100
        ).toFixed(0)} cm`
      : "--";


  const stairTread =
    terrain?.stair?.tread_m != null
      ? `${(
          terrain.stair.tread_m *
          100
        ).toFixed(0)} cm`
      : "--";


  const stairSteps =
    terrain?.stair?.steps != null
      ? String(
          terrain.stair.steps,
        )
      : "--";


  const stairAlignment =
    terrain?.stair
      ?.alignment_error_deg != null
      ? `${terrain.stair.alignment_error_deg.toFixed(
          1,
        )}°`
      : "--";


  const terrainNext =
    !previewMission
      ? "--"
      : previewStep < 2
        ? "STAIR ASCENT"
        : previewStep === 2
          ? "RL TRAVERSE"
          : previewStep === 3
            ? "UPPER PLATFORM"
            : "REGULAR NAV";


  function getStageState(
    index: number,
  ) {
    return (
      mission?.stages[index]
        ?.status ??
      "waiting"
    );
  }


  async function togglePreviewMission() {
    try {
      if (previewRunning) {
        await stopPreviewMission();
      } else {
        await startPreviewMission();
      }
    } catch (error) {
      console.error(
        "Preview mission command failed:",
        error,
      );
    }
  }


  return (
    <main className="navigation-page">
      <section className="navigation-heading">
        <div>
          <span className="navigation-eyebrow">
            MISSION OPERATIONS
          </span>

          <h1>Navigation</h1>

          <p>
            Localization, route execution, terrain transitions
            and mission-level autonomy.
          </p>
        </div>

        <div className="navigation-autonomy-state">
          <span
            className={`navigation-state-dot ${
              backendConnection === "connected"
                ? "backend-linked"
                : backendConnection === "connecting"
                  ? "backend-connecting"
                  : ""
            }`}
          />

          <div>
            <strong>
              {backendConnection === "connected"
                ? "BACKEND LINKED"
                : backendConnection === "connecting"
                  ? "BACKEND CONNECTING"
                  : "BACKEND OFFLINE"}
            </strong>

            <small>
              {robot?.connected
                ? "Robot transport connected"
                : "Robot transport disconnected"}
            </small>
          </div>
        </div>
      </section>


      <section className="navigation-commandbar">
        <div className="navigation-command-group">
          <button
            type="button"
            className={
              selectedTool === "inspect"
                ? "active"
                : ""
            }
            onClick={() =>
              setSelectedTool("inspect")
            }
          >
            <Crosshair
              size={15}
              strokeWidth={1.8}
            />
            Inspect
          </button>

          <button
            type="button"
            className={
              selectedTool === "goal"
                ? "active"
                : ""
            }
            onClick={() =>
              setSelectedTool("goal")
            }
          >
            <Target
              size={15}
              strokeWidth={1.8}
            />
            Set Goal
          </button>
        </div>

        <div className="navigation-command-status">
          <div>
            <small>CONTROL OWNER</small>
            <strong>{controlOwner}</strong>
          </div>

          <div>
            <small>MISSION</small>
            <strong>
              {previewMission
                ? "PREVIEW"
                : "IDLE"}
            </strong>
          </div>

          <div>
            <small>LOCALIZATION</small>
            <strong>
              {(
                navigation?.localization ??
                "offline"
              ).toUpperCase()}
            </strong>
          </div>
        </div>

        <div className="navigation-command-actions">
          <button
            type="button"
            className="mission-preview-button"
            onClick={() => {
              void togglePreviewMission();
            }}
          >
            {previewRunning ? (
              <>
                <Pause size={15} />
                Stop Preview
              </>
            ) : (
              <>
                <Play size={15} />
                Preview Mission
              </>
            )}
          </button>

          <button
            type="button"
            className="mission-start-button"
            disabled
          >
            <Navigation size={15} />
            Start Mission
          </button>
        </div>
      </section>


      <section className="navigation-workspace">
        <div className="navigation-map-panel">
          <div className="navigation-map-toolbar">
            <div>
              <span className="map-toolbar-kicker">
                SPATIAL VIEW
              </span>

              <strong>
                Mission Map
              </strong>
            </div>

            <div className="map-toolbar-right">
              <div className="map-layer-controls">
                {(
                  [
                    ["map", "MAP"],
                    ["path", "PATH"],
                    ["terrain", "TERRAIN"],
                    ["fov", "FOV"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    className={
                      mapLayers[id]
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      toggleMapLayer(id)
                    }
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="map-toolbar-meta">
                <span>
                  <LocateFixed size={12} />
                  LOCALIZATION {localizationStatus}
                </span>

                <span>
                  <Route size={12} />

                  {previewMission
                    ? (
                        mission?.stages[
                          previewStep
                        ]?.label ??
                        missionStages[
                          previewStep
                        ]?.label ??
                        "MISSION"
                      ).toUpperCase()
                    : "NO ACTIVE PATH"}
                </span>
              </div>
            </div>
          </div>


          <div className="navigation-map-canvas">
            <div className="map-grid" />

            {mapLayers.map && previewMission && (
              <div className="preview-occupancy-layer">
                <div className="map-wall wall-a" />
                <div className="map-wall wall-b" />
                <div className="map-wall wall-c" />
                <div className="map-wall wall-d" />

                <div className="map-obstacle obstacle-a" />
                <div className="map-obstacle obstacle-b" />

                <div className="map-clearance-zone clearance-a" />
                <div className="map-clearance-zone clearance-b" />
              </div>
            )}

            <div className="map-axis-label axis-n">
              N
            </div>

            <div className="map-axis-label axis-e">
              E
            </div>


            <div
              className={`map-origin ${
                previewMission
                  ? `preview-step-${previewStep}`
                  : ""
              }`}
            >
              <span className="origin-ring" />

              {previewMission && (
                <>
                  <span className="robot-footprint" />
                  <span className="localization-covariance" />
                </>
              )}

              {mapLayers.fov && (
                <span className="robot-heading-cone" />
              )}

              <G1HeadMarker />

              <small>
                {previewMission
                  ? controlOwner
                  : "G1"}
              </small>
            </div>


            {previewMission && mapLayers.map && (
              <div
                className={`local-costmap preview-step-${previewStep}`}
              >
                <span className="costmap-ring costmap-ring-a" />
                <span className="costmap-ring costmap-ring-b" />
                <span className="costmap-cell cell-a" />
                <span className="costmap-cell cell-b" />
                <span className="costmap-cell cell-c" />
              </div>
            )}


            {previewMission && (
              <>
                {mapLayers.path && (
                  <>
                    <svg
                      className="mission-route"
                      viewBox="0 0 1000 620"
                      preserveAspectRatio="none"
                    >
                      <path
                        className="route-remaining"
                        d="
                          M 235 465
                          C 300 435, 345 395, 410 360
                          S 535 315, 590 300
                          C 645 286, 690 245, 725 205
                          S 790 155, 842 135
                        "
                      />

                      <path
                        className={`route-traversed step-${previewStep}`}
                        d="
                          M 235 465
                          C 300 435, 345 395, 410 360
                          S 535 315, 590 300
                          C 645 286, 690 245, 725 205
                          S 790 155, 842 135
                        "
                      />
                    </svg>

                    <svg
                      className="local-trajectory"
                      viewBox="0 0 1000 620"
                      preserveAspectRatio="none"
                    >
                      <path
                        d="
                          M 235 465
                          C 270 445, 305 425, 342 402
                        "
                      />
                    </svg>
                  </>
                )}

                <div className="mission-waypoint waypoint-a">
                  <span>1</span>
                  <small>APPROACH</small>
                </div>

                {mapLayers.terrain && (
                  <div
                    className={`terrain-region ${
                      previewStep === 2
                        ? "detected"
                        : previewStep === 3
                          ? "traversing"
                          : previewStep > 3
                            ? "cleared"
                            : ""
                    }`}
                  >
                  <div className="terrain-region-pattern">
                    <span />
                    <span />
                    <span />
                  </div>

                  <strong>
                    {previewStep < 2
                      ? "STAIR TRANSITION"
                      : previewStep === 2
                        ? "STAIR DETECTED"
                        : previewStep === 3
                          ? "STAIR TRAVERSAL"
                          : "STAIR CLEARED"}
                  </strong>

                  <small>
                    {previewStep === 3
                      ? "control: stair rl"
                      : previewStep > 3
                        ? "navigation resumed"
                        : "terrain event"}
                  </small>
                  </div>
                )}

                <div className="mission-waypoint waypoint-b">
                  <span>2</span>
                  <small>UPPER PLATFORM</small>
                </div>

                <div
                  className={`mission-goal ${
                    previewStep === 5
                      ? "active"
                      : ""
                  }`}
                >
                  <Flag
                    size={18}
                    strokeWidth={1.8}
                  />

                  <div>
                    <strong>MISSION GOAL</strong>
                    <span>Upper platform + 2 m</span>
                  </div>
                </div>
              </>
            )}


            {!previewMission && (
              <div className="navigation-map-empty">
                <Map
                  size={34}
                  strokeWidth={1.25}
                />

                <strong>
                  No map or localization data
                </strong>

                <span>
                  The robot pose, global path, terrain regions
                  and mission waypoints will appear here.
                </span>
              </div>
            )}


            <div className="map-scale">
              <span />
              <small>1 m</small>
            </div>


            {previewMission && (
              <div className="map-legend">
                <span>
                  <i className="legend-path" />
                  Global path
                </span>

                <span>
                  <i className="legend-local" />
                  Local trajectory
                </span>

                <span>
                  <i className="legend-terrain" />
                  Terrain transition
                </span>
              </div>
            )}

            <div className="map-status-strip">
              <div>
                <small>POSE</small>
                <strong>
                  {poseValid
                    ? `${poseX}, ${poseY}`
                    : "--"}
                </strong>
              </div>

              <div>
                <small>HEADING</small>
                <strong>
                  {poseYaw}
                </strong>
              </div>

              <div>
                <small>GOAL DISTANCE</small>
                <strong>
                  {goalDistance}
                </strong>
              </div>

              <div>
                <small>PATH LENGTH</small>
                <strong>
                  {globalPathLength
                    ? `${globalPathLength} m`
                    : "--"}
                </strong>
              </div>
            </div>
          </div>
        </div>


        <aside className="mission-panel">
          <div className="mission-panel-heading">
            <div>
              <span>
                MISSION MANAGER
              </span>

              <strong>
                Navigate to Upper Platform
              </strong>
            </div>

            <div className="mission-state-badge">
              {previewMission
                ? "PREVIEW"
                : "IDLE"}
            </div>
          </div>


          <div className="mission-destination">
            <div className="mission-destination-icon">
              <MapPin
                size={20}
                strokeWidth={1.7}
              />
            </div>

            <div>
              <small>DESTINATION</small>
              <strong>Upper Platform</strong>
              <span>
                Mission target not sent
              </span>
            </div>
          </div>


          <div className="mission-progress">
            <div>
              <span>MISSION PROGRESS</span>
              <strong>
                {missionProgressLabel}
              </strong>
            </div>

            <div className="mission-progress-track">
              <span
                style={{
                  width: previewMission
                    ? `${previewProgress}%`
                    : "0%",
                }}
              />
            </div>
          </div>


          <div className="mission-metrics">
            <div>
              <small>DISTANCE</small>
              <strong>
                {goalDistance}
              </strong>
            </div>

            <div>
              <small>ETA</small>
              <strong>
                {eta}
              </strong>
            </div>

            <div>
              <small>WAYPOINT</small>
              <strong>
                {previewMission
                  ? `${Math.min(
                      previewStep + 1,
                      missionStages.length,
                    )} / ${missionStages.length}`
                  : "-- / --"}
              </strong>
            </div>
          </div>


          <div className="mission-section-label">
            MISSION PIPELINE
          </div>

          <div className="mission-stage-list">
            {(mission?.stages.length
              ? mission.stages
              : missionStages
            ).map(
              (stage, index) => {
                const state =
                  getStageState(index);

                return (
                  <div
                    className={`mission-stage ${state}`}
                    key={stage.id}
                  >
                    <div className="mission-stage-index">
                      {index + 1}
                    </div>

                    <div className="mission-stage-copy">
                      <strong>
                        {stage.label}
                      </strong>

                      <span>
                        {state === "active"
                          ? "active"
                          : state}
                      </span>
                    </div>

                    {state === "complete" ? (
                      <CheckCircle2
                        size={13}
                        strokeWidth={1.8}
                      />
                    ) : (
                      <Circle
                        size={12}
                        strokeWidth={1.5}
                      />
                    )}
                  </div>
                );
              },
            )}
          </div>


          <div className="mission-stop-row">
            <button
              type="button"
              disabled
            >
              <Square
                size={13}
                fill="currentColor"
              />
              Stop Mission
            </button>
          </div>
        </aside>
      </section>


      <section className="navigation-lower-deck">
        <div className="navigation-deck-header">
          <div className="navigation-deck-tabs">
            <button
              type="button"
              className={
                lowerDeck === "status"
                  ? "active"
                  : ""
              }
              onClick={() =>
                setLowerDeck("status")
              }
            >
              <Activity size={13} />
              Mission Status
            </button>

            <button
              type="button"
              className={
                lowerDeck === "timeline"
                  ? "active"
                  : ""
              }
              onClick={() =>
                setLowerDeck("timeline")
              }
            >
              <Route size={13} />
              Timeline
            </button>

            <button
              type="button"
              className={
                lowerDeck === "events"
                  ? "active"
                  : ""
              }
              onClick={() =>
                setLowerDeck("events")
              }
            >
              <Waypoints size={13} />
              Events
            </button>
          </div>

          <div className="navigation-deck-summary">
            <span>MISSION</span>
            <strong>
              {previewMission
                ? (
                    mission?.stages[
                      previewStep
                    ]?.label ??
                    "MISSION"
                  ).toUpperCase()
                : "IDLE"}
            </strong>
          </div>
        </div>


        {lowerDeck === "status" && (
          <div className="navigation-state-grid">
            <article className="navigation-state-card">
              <div className="state-card-heading">
                <div className="state-card-icon">
                  <LocateFixed size={18} />
                </div>

                <div>
                  <span>LOCALIZATION</span>
                  <strong>Pose Estimate</strong>
                </div>
              </div>

              <div className="state-card-primary">
                <span
                  className={`state-indicator ${
                    poseValid
                      ? "active"
                      : "neutral"
                  }`}
                />

                <strong>
                  {poseValid
                    ? localizationStatus
                    : "No localization"}
                </strong>
              </div>

              <div className="state-card-values">
                <div>
                  <small>X</small>
                  <strong>
                    {poseX}
                  </strong>
                </div>

                <div>
                  <small>Y</small>
                  <strong>
                    {poseY}
                  </strong>
                </div>

                <div>
                  <small>YAW</small>
                  <strong>
                    {poseYaw}
                  </strong>
                </div>
              </div>
            </article>


            <article className="navigation-state-card terrain-card">
              <div className="state-card-heading">
                <div className="state-card-icon">
                  <Footprints size={18} />
                </div>

                <div>
                  <span>TERRAIN</span>
                  <strong>Terrain Manager</strong>
                </div>
              </div>

              <div className="state-card-primary">
                <span className="state-indicator neutral" />

                <strong>
                  {previewMission
                    ? terrainState
                    : "No terrain data"}
                </strong>
              </div>

              <div className="terrain-transition">
                <div>
                  <small>CURRENT</small>
                  <strong>
                    {previewMission
                      ? terrainState
                      : "--"}
                  </strong>
                </div>

                <span>→</span>

                <div>
                  <small>NEXT</small>
                  <strong>
                    {terrainNext}
                  </strong>
                </div>
              </div>

              {terrain?.stair && (
                <div className="terrain-metrics">
                  <div>
                    <small>CONF</small>
                    <strong>
                      {terrainConfidence}
                    </strong>
                  </div>

                  <div>
                    <small>RISE</small>
                    <strong>
                      {stairRise}
                    </strong>
                  </div>

                  <div>
                    <small>TREAD</small>
                    <strong>
                      {stairTread}
                    </strong>
                  </div>

                  <div>
                    <small>STEPS</small>
                    <strong>
                      {stairSteps}
                    </strong>
                  </div>

                  <div>
                    <small>ALIGN</small>
                    <strong>
                      {stairAlignment}
                    </strong>
                  </div>
                </div>
              )}
            </article>


            <article className="navigation-state-card authority-card">
              <div className="state-card-heading">
                <div className="state-card-icon">
                  <Waypoints size={18} />
                </div>

                <div>
                  <span>CONTROL</span>
                  <strong>Authority</strong>
                </div>
              </div>

              <div className="state-card-primary">
                <span
                  className={`state-indicator ${
                    previewMission
                      ? "active"
                      : "neutral"
                  }`}
                />

                <strong>
                  {controlOwner}
                </strong>
              </div>

              <div className="authority-chain">
                <span>NAV</span>
                <span>→</span>
                <span>STAIR APPROACH</span>
                <span>→</span>
                <span>RL POLICY</span>
              </div>
            </article>


            <article className="navigation-state-card">
              <div className="state-card-heading">
                <div className="state-card-icon">
                  <ShieldCheck size={18} />
                </div>

                <div>
                  <span>SAFETY</span>
                  <strong>Mission Guard</strong>
                </div>
              </div>

              <div className="state-card-primary">
                <span className="state-indicator neutral" />
                <strong>Awaiting robot</strong>
              </div>

              <div className="safety-checks">
                <span>Stability --</span>
                <span>Timeout --</span>
                <span>Fall --</span>
              </div>
            </article>
          </div>
        )}


        {lowerDeck === "timeline" && (
          <div className="mission-timeline compact">
            <div className="timeline-line" />

            <div className="timeline-node">
              <CheckCircle2 size={18} />
              <strong>Start</strong>
              <span>Mission accepted</span>
            </div>

            <div className="timeline-node">
              <Navigation size={18} />
              <strong>Navigate</strong>
              <span>Regular locomotion</span>
            </div>

            <div className="timeline-node terrain">
              <AlertTriangle size={18} />
              <strong>Terrain</strong>
              <span>Detect + align</span>
            </div>

            <div className="timeline-node terrain">
              <Footprints size={18} />
              <strong>Traverse</strong>
              <span>Stair RL policy</span>
            </div>

            <div className="timeline-node">
              <Route size={18} />
              <strong>Resume</strong>
              <span>Navigation restored</span>
            </div>

            <div className="timeline-node">
              <Flag size={18} />
              <strong>Goal</strong>
              <span>Mission complete</span>
            </div>
          </div>
        )}


        {lowerDeck === "events" && (
          <>
            {dashboardState?.events.length ? (
              <div className="navigation-event-list">
                {[...dashboardState.events]
                  .reverse()
                  .slice(0, 8)
                  .map((event) => (
                    <div
                      key={event.sequence}
                      className="navigation-event-row"
                    >
                      <span className="event-sequence">
                        {String(
                          event.sequence,
                        ).padStart(
                          2,
                          "0",
                        )}
                      </span>

                      <div>
                        <strong>
                          {event.kind
                            .replaceAll(
                              "_",
                              " ",
                            )
                            .toUpperCase()}
                        </strong>

                        <small>
                          {event.message}
                        </small>
                      </div>

                      <span className="event-source">
                        {event.source}
                      </span>
                    </div>
                  ))}
              </div>
            ) : (
              <div className="navigation-event-empty compact">
                <Activity
                  size={22}
                  strokeWidth={1.4}
                />

                <span>
                  No mission events received.
                </span>
              </div>
            )}
          </>
        )}
      </section>
    </main>
  );
}
