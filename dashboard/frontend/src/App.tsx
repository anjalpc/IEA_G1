import { useMemo, useState } from "react";
import {
  Activity,
  BatteryMedium,
  Camera,
  Cpu,
  LayoutDashboard,
  Map,
  Radio,
  Shield,
  Wifi,
} from "lucide-react";

import { ActionDeck } from "./components/ActionDeck";
import { ActionDrawer } from "./components/ActionDrawer";
import { VirtualJoystick } from "./components/VirtualJoystick";
import type { StickValue } from "./components/VirtualJoystick";
import type {
  ActionGroup,
  G1Action,
} from "./data/g1Actions";

import { OverviewPage } from "./pages/OverviewPage";
import { VisionPage } from "./pages/VisionPage";
import { NavigationPage } from "./pages/NavigationPage";

import "./App.css";

type Page =
  | "overview"
  | "control"
  | "vision"
  | "navigation"
  | "analysis"
  | "system";

const pages: {
  id: Page;
  label: string;
  icon: typeof LayoutDashboard;
}[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "control", label: "Control", icon: Radio },
  { id: "vision", label: "Vision", icon: Camera },
  { id: "navigation", label: "Navigation", icon: Map },
  { id: "analysis", label: "Analysis", icon: Activity },
  { id: "system", label: "System", icon: Cpu },
];


function App() {
  const [page, setPage] = useState<Page>("control");

  const [leftStick, setLeftStick] = useState<StickValue>({
    x: 0,
    y: 0,
  });

  const [rightStick, setRightStick] = useState<StickValue>({
    x: 0,
    y: 0,
  });

  const [armed, setArmed] = useState(false);
  const [lastAction, setLastAction] = useState("None");

  const [actionDrawerOpen, setActionDrawerOpen] =
    useState(false);

  const [drawerGroup, setDrawerGroup] =
    useState<ActionGroup>("posture");

  const command = useMemo(
    () => ({
      vx: leftStick.y * 0.6,
      vy: leftStick.x * 0.4,
      yaw: rightStick.x,
    }),
    [leftStick, rightStick],
  );

  function resetSticks() {
    setLeftStick({ x: 0, y: 0 });
    setRightStick({ x: 0, y: 0 });
  }

  function toggleControl() {
    setArmed((current) => !current);
    resetSticks();
  }

  function executeAction(action: G1Action) {
    if (!armed) return;

    setLastAction(action.label);
  }

  function emergencyStop() {
    resetSticks();
    setArmed(false);
    setLastAction("Emergency Stop");
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="product">
          <div className="product-mark">
            <span>G1</span>
          </div>

          <div className="product-name">
            <strong>FIELD CONSOLE</strong>
            <span>ROBOTICS OPERATIONS / RESEARCH</span>
          </div>
        </div>

        <nav className="top-navigation">
          {pages.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={page === id ? "active" : ""}
              onClick={() => setPage(id)}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </nav>

        <div className="global-health">
          <span className="health-dot preview" />

          <div>
            <small>ROBOT</small>
            <strong>PREVIEW</strong>
          </div>
        </div>
      </header>

      {page === "overview" ? (
        <OverviewPage />
      ) : page === "vision" ? (
        <VisionPage />
      ) : page === "navigation" ? (
        <NavigationPage />
      ) : page === "control" ? (
        <main className="control-page">
          <section className="control-statusbar">
            <div className="status-cluster">
              <div className="status-property">
                <span className="status-light waiting" />
                <div>
                  <small>CONNECTION</small>
                  <strong>Not linked</strong>
                </div>
              </div>

              <div className="status-property">
                <Wifi size={16} />
                <div>
                  <small>LATENCY</small>
                  <strong>-- ms</strong>
                </div>
              </div>

              <div className="status-property">
                <BatteryMedium size={17} />
                <div>
                  <small>BATTERY</small>
                  <strong>--%</strong>
                </div>
              </div>
            </div>

            <div className="mode-label">
              <small>CONTROL MODE</small>
              <strong>MANUAL CONTROL</strong>
            </div>

            <button
              className={`control-lock ${armed ? "armed" : ""}`}
              onClick={toggleControl}
            >
              <Shield size={16} />
              {armed ? "CONTROL ARMED" : "ENABLE CONTROL"}
            </button>
          </section>

          <section className="cockpit">
            <div className="cockpit-grid" />

            <div className="stick-anchor stick-anchor-left">
              <VirtualJoystick
                label="LOCOMOTION"
                description="Forward / reverse / lateral"
                value={leftStick}
                onChange={setLeftStick}
                disabled={!armed}
              />

              <div className="stick-command">
                <div>
                  <small>VX</small>
                  <strong>
                    {command.vx >= 0 ? "+" : ""}
                    {command.vx.toFixed(2)}
                  </strong>
                  <span>m/s</span>
                </div>

                <div>
                  <small>VY</small>
                  <strong>
                    {command.vy >= 0 ? "+" : ""}
                    {command.vy.toFixed(2)}
                  </strong>
                  <span>m/s</span>
                </div>
              </div>
            </div>

            <div className="operator-hud">
              <div className="operator-hud-top">
                <div>
                  <span className="operator-hud-kicker">
                    COMMAND HUD
                  </span>

                  <strong>MANUAL CONTROL</strong>
                </div>

                <div
                  className={`operator-hud-state ${
                    armed ? "armed" : ""
                  }`}
                >
                  <span />

                  {armed
                    ? "CONTROL ARMED"
                    : "CONTROL LOCKED"}
                </div>
              </div>


              <div className="operator-hud-core">
                <div className="command-reticle">
                  <span className="reticle-ring reticle-ring-outer" />
                  <span className="reticle-ring reticle-ring-inner" />
                  <span className="reticle-axis reticle-axis-x" />
                  <span className="reticle-axis reticle-axis-y" />

                  <div className="command-speed">
                    <small>COMMAND SPEED</small>

                    <strong>
                      {Math.hypot(
                        command.vx,
                        command.vy,
                      ).toFixed(2)}
                    </strong>

                    <span>m/s</span>
                  </div>
                </div>

                <div className="operator-command-grid">
                  <div>
                    <small>FORWARD</small>

                    <strong>
                      {command.vx >= 0 ? "+" : ""}
                      {command.vx.toFixed(2)}
                    </strong>

                    <span>VX / m/s</span>
                  </div>

                  <div>
                    <small>LATERAL</small>

                    <strong>
                      {command.vy >= 0 ? "+" : ""}
                      {command.vy.toFixed(2)}
                    </strong>

                    <span>VY / m/s</span>
                  </div>

                  <div>
                    <small>YAW</small>

                    <strong>
                      {command.yaw >= 0 ? "+" : ""}
                      {command.yaw.toFixed(2)}
                    </strong>

                    <span>rad/s</span>
                  </div>
                </div>
              </div>


              <div className="operator-hud-footer">
                <div>
                  <small>ROBOT MODE</small>
                  <strong>--</strong>
                </div>

                <div>
                  <small>LAST ACTION</small>
                  <strong>{lastAction}</strong>
                </div>

                <div>
                  <small>COMMAND RATE</small>
                  <strong>0 Hz</strong>
                </div>
              </div>

              <div className="operator-hud-offline">
                <span />

                LIVE ROBOT TELEMETRY UNAVAILABLE
              </div>
            </div>

            <div className="stick-anchor stick-anchor-right">
              <VirtualJoystick
                label="ROTATION"
                description="Yaw / heading"
                value={rightStick}
                onChange={setRightStick}
                disabled={!armed}
              />

              <div className="stick-command single">
                <div>
                  <small>YAW</small>
                  <strong>
                    {command.yaw >= 0 ? "+" : ""}
                    {command.yaw.toFixed(2)}
                  </strong>
                  <span>rad/s</span>
                </div>
              </div>
            </div>
          </section>
          <ActionDeck
            armed={armed}
            onAction={executeAction}
            onOpenActions={(group) => {
              setDrawerGroup(group);
              setActionDrawerOpen(true);
            }}
            onEmergency={emergencyStop}
          />

          <footer className="control-footer">
            <span>UI PREVIEW / ROBOT TRANSPORT DISCONNECTED</span>

            <div>
              <span>ROS DOMAIN 42</span>
              <span>•</span>
              <span>G1 EDU</span>
              <span>•</span>
              <span>29 DOF</span>
            </div>
          </footer>

          {actionDrawerOpen && (
            <ActionDrawer
              open={actionDrawerOpen}
              armed={armed}
              initialGroup={drawerGroup}
              onClose={() =>
                setActionDrawerOpen(false)
              }
              onAction={executeAction}
            />
          )}
        </main>
      ) : (
        <main className="workspace-page">
          <div className="workspace-heading">
            <span>G1 FIELD CONSOLE</span>
            <h1>
              {pages.find((item) => item.id === page)?.label}
            </h1>
          </div>

          <div className="workspace-placeholder">
            <div className="workspace-number">0{pages.findIndex((x) => x.id === page) + 1}</div>

            <div>
              <span>WORKSPACE</span>
              <h2>
                {pages.find((item) => item.id === page)?.label}
              </h2>
              <p>
                This workspace will be built after the operator control
                surface is visually locked.
              </p>
            </div>
          </div>
        </main>
      )}
    </div>
  );
}

export default App;
