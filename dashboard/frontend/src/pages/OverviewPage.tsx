import { useState } from "react";

import {
  Activity,
  Battery,
  Bot,
  Camera,
  Cpu,
  Gauge,
  Hand,
  Network,
  ScanLine,
} from "lucide-react";

import { G1Model3D } from "../components/G1Model3D";

import "./OverviewPage.css";


const telemetry = [
  {
    label: "Battery",
    value: "--",
    detail: "No telemetry",
    icon: Battery,
  },
  {
    label: "Uptime",
    value: "--",
    detail: "Robot not linked",
    icon: Activity,
  },
  {
    label: "Network RTT",
    value: "-- ms",
    detail: "No connection",
    icon: Network,
  },
  {
    label: "Compute",
    value: "--",
    detail: "CPU / memory",
    icon: Cpu,
  },
];


const sensors = [
  {
    name: "RGB Camera",
    detail: "Unitree vision",
    icon: Camera,
  },
  {
    name: "Depth Camera",
    detail: "RealSense D435i",
    icon: ScanLine,
  },
  {
    name: "3D LiDAR",
    detail: "No stream",
    icon: Gauge,
  },
  {
    name: "IMU",
    detail: "No telemetry",
    icon: Activity,
  },
];


const hardware = [
  {
    name: "Left Hand",
    detail: "BrainCo Revo2",
    icon: Hand,
  },
  {
    name: "Right Hand",
    detail: "BrainCo Revo2",
    icon: Hand,
  },
  {
    name: "Upper Body",
    detail: "Arm joints",
    icon: Bot,
  },
  {
    name: "Lower Body",
    detail: "Leg joints",
    icon: Bot,
  },
];


export function OverviewPage() {
  const [modelView, setModelView] =
    useState<"upper" | "full">("upper");

  return (
    <main className="overview-page">
      <section className="overview-heading">
        <div>
          <span className="overview-eyebrow">
            ROBOT DIGITAL TWIN
          </span>

          <h1>G1 Overview</h1>

          <p>
            Hardware, telemetry and system state for the
            Unitree G1 platform.
          </p>
        </div>

        <div className="overview-link-state">
          <span className="overview-status-dot" />

          <div>
            <strong>NOT LINKED</strong>
            <span>Waiting for robot</span>
          </div>
        </div>
      </section>


      <section className="overview-hero">
        <aside className="overview-identity-panel">
          <div className="overview-panel-kicker">
            ROBOT
          </div>

          <div className="overview-robot-name">
            <div className="overview-robot-icon">
              <Bot size={25} strokeWidth={1.7} />
            </div>

            <div>
              <strong>G1 EDU U7</strong>
              <span>Unitree Robotics</span>
            </div>
          </div>

          <dl className="overview-spec-list">
            <div>
              <dt>Locomotion</dt>
              <dd>29 DoF</dd>
            </div>

            <div>
              <dt>Robot state</dt>
              <dd>--</dd>
            </div>

            <div>
              <dt>Motion mode</dt>
              <dd>--</dd>
            </div>

            <div>
              <dt>Control owner</dt>
              <dd>None</dd>
            </div>

            <div>
              <dt>Software</dt>
              <dd>--</dd>
            </div>
          </dl>

          <div className="overview-offline-note">
            Live identity and state will populate when the
            dashboard backend links to the robot.
          </div>
        </aside>


        <div className="overview-model-stage">
          <div className="overview-model-topline">
            <span>DIGITAL TWIN</span>

            <div className="overview-model-tools">
              <div className="overview-model-toggle">
                <button
                  type="button"
                  className={
                    modelView === "upper"
                      ? "active"
                      : ""
                  }
                  onClick={() =>
                    setModelView("upper")
                  }
                >
                  UPPER BODY
                </button>

                <button
                  type="button"
                  className={
                    modelView === "full"
                      ? "active"
                      : ""
                  }
                  onClick={() =>
                    setModelView("full")
                  }
                >
                  FULL BODY
                </button>
              </div>

              <span>DRAG TO INSPECT</span>
            </div>
          </div>

          <G1Model3D
            view={
              modelView === "upper"
                ? "overview"
                : "overview-full"
            }
            showLabel={false}
          />

          <div className="overview-model-footer">
            <div>
              <span>MODEL</span>
              <strong>G1 29-DOF</strong>
            </div>

            <div>
              <span>VIEW</span>
              <strong>
                {modelView === "upper"
                  ? "UPPER BODY"
                  : "FULL BODY"}
              </strong>
            </div>

            <div>
              <span>JOINT STREAM</span>
              <strong>OFFLINE</strong>
            </div>
          </div>
        </div>


        <aside className="overview-telemetry-panel">
          <div className="overview-panel-kicker">
            TELEMETRY
          </div>

          <div className="overview-telemetry-grid">
            {telemetry.map((item) => {
              const Icon = item.icon;

              return (
                <article
                  className="overview-telemetry-card"
                  key={item.label}
                >
                  <div className="overview-mini-icon">
                    <Icon
                      size={18}
                      strokeWidth={1.8}
                    />
                  </div>

                  <div>
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                    <small>{item.detail}</small>
                  </div>
                </article>
              );
            })}
          </div>

          <div className="overview-connection-card">
            <div className="overview-connection-head">
              <Network
                size={18}
                strokeWidth={1.8}
              />

              <span>ROBOT LINK</span>
            </div>

            <strong>Disconnected</strong>

            <p>
              Network and ROS 2 transport information will
              appear here after connection.
            </p>
          </div>
        </aside>
      </section>


      <section className="overview-section">
        <div className="overview-section-heading">
          <div>
            <span>PERCEPTION</span>
            <h2>Sensors</h2>
          </div>

          <small>No live streams</small>
        </div>

        <div className="overview-device-grid">
          {sensors.map((sensor) => {
            const Icon = sensor.icon;

            return (
              <article
                className="overview-device-card"
                key={sensor.name}
              >
                <div className="overview-device-icon">
                  <Icon
                    size={22}
                    strokeWidth={1.7}
                  />
                </div>

                <div className="overview-device-copy">
                  <strong>{sensor.name}</strong>
                  <span>{sensor.detail}</span>
                </div>

                <div className="overview-device-state">
                  <span />
                  NO DATA
                </div>
              </article>
            );
          })}
        </div>
      </section>


      <section className="overview-section">
        <div className="overview-section-heading">
          <div>
            <span>ACTUATION</span>
            <h2>Hardware</h2>
          </div>

          <small>Awaiting diagnostics</small>
        </div>

        <div className="overview-device-grid">
          {hardware.map((device) => {
            const Icon = device.icon;

            return (
              <article
                className="overview-device-card"
                key={device.name}
              >
                <div className="overview-device-icon">
                  <Icon
                    size={22}
                    strokeWidth={1.7}
                  />
                </div>

                <div className="overview-device-copy">
                  <strong>{device.name}</strong>
                  <span>{device.detail}</span>
                </div>

                <div className="overview-device-state">
                  <span />
                  UNKNOWN
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </main>
  );
}
