import { useState } from "react";

import {
  Camera,
  CircleDot,
  Eye,
  Maximize2,
  ScanLine,
  Waves,
} from "lucide-react";

import "./VisionPage.css";


type VisionView =
  | "rgb"
  | "depth"
  | "lidar";


const views: {
  id: VisionView;
  label: string;
  icon: typeof Camera;
}[] = [
  {
    id: "rgb",
    label: "RGB",
    icon: Camera,
  },
  {
    id: "depth",
    label: "Depth",
    icon: ScanLine,
  },
  {
    id: "lidar",
    label: "LiDAR",
    icon: Waves,
  },
];


const sensorInfo = {
  rgb: {
    title: "Front RGB Camera",
    device: "Unitree onboard vision",
    source: "videohub_pc4",
    frame: "g1_rgb_camera",
  },

  depth: {
    title: "Depth Camera",
    device: "Intel RealSense D435i",
    source: "Depth backend",
    frame: "g1_depth_camera",
  },

  lidar: {
    title: "3D LiDAR",
    device: "Point-cloud source",
    source: "Not integrated",
    frame: "--",
  },
} satisfies Record<
  VisionView,
  {
    title: string;
    device: string;
    source: string;
    frame: string;
  }
>;


export function VisionPage() {
  const [view, setView] =
    useState<VisionView>("rgb");

  const active =
    sensorInfo[view];

  return (
    <main className="vision-page">
      <section className="vision-heading">
        <div>
          <span className="vision-eyebrow">
            PERCEPTION CONSOLE
          </span>

          <h1>Vision</h1>

          <p>
            Camera, depth and spatial perception streams from
            the G1 sensor stack.
          </p>
        </div>

        <div className="vision-link-state">
          <span />

          <div>
            <strong>NO LIVE STREAM</strong>
            <small>Robot transport disconnected</small>
          </div>
        </div>
      </section>


      <section className="vision-workspace">
        <div className="vision-main">
          <div className="vision-toolbar">
            <div className="vision-view-tabs">
              {views.map(
                ({
                  id,
                  label,
                  icon: Icon,
                }) => (
                  <button
                    key={id}
                    type="button"
                    className={
                      view === id
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      setView(id)
                    }
                  >
                    <Icon
                      size={14}
                      strokeWidth={1.8}
                    />

                    {label}
                  </button>
                ),
              )}
            </div>

            <div className="vision-toolbar-actions">
              <span>
                {view === "lidar"
                  ? "3D VIEW"
                  : "IMAGE VIEW"}
              </span>

              <button
                type="button"
                aria-label="Expand vision view"
                disabled
              >
                <Maximize2
                  size={14}
                  strokeWidth={1.7}
                />
              </button>
            </div>
          </div>


          <div
            className={`vision-stream vision-stream-${view}`}
          >
            <div className="vision-grid-overlay" />

            {view === "rgb" && (
              <div className="vision-placeholder">
                <div className="vision-placeholder-icon">
                  <Camera
                    size={34}
                    strokeWidth={1.35}
                  />
                </div>

                <strong>RGB stream unavailable</strong>

                <span>
                  Waiting for Unitree camera transport
                </span>
              </div>
            )}


            {view === "depth" && (
              <div className="vision-placeholder">
                <div className="vision-depth-symbol">
                  <span />
                  <span />
                  <span />
                  <span />
                </div>

                <strong>Depth stream unavailable</strong>

                <span>
                  Waiting for RealSense D435i depth frames
                </span>
              </div>
            )}


            {view === "lidar" && (
              <div className="vision-placeholder">
                <div className="vision-lidar-symbol">
                  <CircleDot
                    size={46}
                    strokeWidth={1.1}
                  />

                  <span className="lidar-ray lidar-ray-a" />
                  <span className="lidar-ray lidar-ray-b" />
                </div>

                <strong>Point cloud unavailable</strong>

                <span>
                  LiDAR integration is not active yet
                </span>
              </div>
            )}


            <div className="vision-stream-top">
              <div>
                <span className="vision-stream-dot" />
                OFFLINE
              </div>

              <span>
                {active.frame}
              </span>
            </div>


            <div className="vision-stream-bottom">
              <div>
                <small>FPS</small>
                <strong>--</strong>
              </div>

              <div>
                <small>RESOLUTION</small>
                <strong>--</strong>
              </div>

              <div>
                <small>LATENCY</small>
                <strong>-- ms</strong>
              </div>

              <div>
                <small>DROPPED</small>
                <strong>--</strong>
              </div>
            </div>
          </div>
        </div>


        <aside className="vision-sidebar">
          <section className="vision-info-card">
            <span className="vision-card-kicker">
              ACTIVE SENSOR
            </span>

            <div className="vision-sensor-title">
              <div>
                {view === "rgb" && (
                  <Camera size={21} />
                )}

                {view === "depth" && (
                  <ScanLine size={21} />
                )}

                {view === "lidar" && (
                  <Waves size={21} />
                )}
              </div>

              <div>
                <strong>{active.title}</strong>
                <span>{active.device}</span>
              </div>
            </div>

            <dl className="vision-property-list">
              <div>
                <dt>Status</dt>
                <dd>Offline</dd>
              </div>

              <div>
                <dt>Source</dt>
                <dd>{active.source}</dd>
              </div>

              <div>
                <dt>Frame</dt>
                <dd>{active.frame}</dd>
              </div>

              <div>
                <dt>Transport</dt>
                <dd>--</dd>
              </div>

              <div>
                <dt>Timestamp</dt>
                <dd>--</dd>
              </div>
            </dl>
          </section>


          <section className="vision-info-card compact">
            <span className="vision-card-kicker">
              SENSOR STACK
            </span>

            <div className="vision-device-row">
              <div className="vision-device-icon">
                <Camera size={17} />
              </div>

              <div>
                <strong>RGB Camera</strong>
                <span>Unitree onboard</span>
              </div>

              <small>NO DATA</small>
            </div>

            <div className="vision-device-row">
              <div className="vision-device-icon">
                <ScanLine size={17} />
              </div>

              <div>
                <strong>Depth Camera</strong>
                <span>RealSense D435i</span>
              </div>

              <small>NO DATA</small>
            </div>

            <div className="vision-device-row">
              <div className="vision-device-icon">
                <Waves size={17} />
              </div>

              <div>
                <strong>3D LiDAR</strong>
                <span>Future stream</span>
              </div>

              <small>OFFLINE</small>
            </div>

            <div className="vision-device-row">
              <div className="vision-device-icon">
                <Eye size={17} />
              </div>

              <div>
                <strong>Perception</strong>
                <span>Detections / overlays</span>
              </div>

              <small>INACTIVE</small>
            </div>
          </section>
        </aside>
      </section>


      <section className="vision-lower-grid">
        <article className="vision-analysis-card">
          <div className="vision-analysis-heading">
            <div>
              <span>PERCEPTION</span>
              <strong>Detection Overlay</strong>
            </div>

            <small>INACTIVE</small>
          </div>

          <div className="vision-analysis-empty">
            <Eye
              size={23}
              strokeWidth={1.35}
            />

            <span>
              Object, hand and navigation overlays will appear
              here when perception nodes are active.
            </span>
          </div>
        </article>


        <article className="vision-analysis-card">
          <div className="vision-analysis-heading">
            <div>
              <span>DEPTH</span>
              <strong>Range Inspector</strong>
            </div>

            <small>NO DATA</small>
          </div>

          <div className="vision-depth-scale">
            <div className="vision-depth-bar" />

            <div>
              <span>NEAR</span>
              <span>FAR</span>
            </div>
          </div>

          <div className="vision-analysis-empty small">
            <span>
              Cursor depth and ROI statistics will appear when
              depth frames are available.
            </span>
          </div>
        </article>
      </section>
    </main>
  );
}
