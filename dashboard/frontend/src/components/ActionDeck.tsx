import {
  ChevronRight,
  Gauge,
  Hand,
  Octagon,
  ShieldAlert,
  Sparkles,
  Power,
} from "lucide-react";

import { G1_ACTIONS } from "../data/g1Actions";

import type {
  ActionGroup,
  G1Action,
} from "../data/g1Actions";

interface ActionDeckProps {
  armed: boolean;
  onAction: (action: G1Action) => void;
  onOpenActions: (group: ActionGroup) => void;
  onEmergency: () => void;
}

interface PanelDefinition {
  title: string;
  subtitle: string;
  group: ActionGroup;
  icon: typeof Power;
  actionIds: string[];
}

const panels: PanelDefinition[] = [
  {
    title: "Robot State",
    subtitle: "Native G1 state transitions",
    group: "posture",
    icon: Power,
    actionIds: [
      "zero_torque",
      "damping",
      "locked_standing",
      "lie_to_stand",
      "sit",
      "squat",
    ],
  },
  {
    title: "Locomotion",
    subtitle: "Unitree motion modes",
    group: "locomotion",
    icon: Gauge,
    actionIds: [
      "regular_mode",
      "running_mode",
      "climb_mode",
      "low_speed",
      "high_speed",
      "stop_move",
    ],
  },
  {
    title: "Interaction",
    subtitle: "Native interaction actions",
    group: "interaction",
    icon: Hand,
    actionIds: [
      "shake_hand",
      "wave",
      "reject",
      "face_wave",
      "clap",
      "right_hand_up",
    ],
  },
  {
    title: "Special Motions",
    subtitle: "Unitree demos & motions",
    group: "performance",
    icon: Sparkles,
    actionIds: [
      "dance_mode",
      "dance_1",
      "dance_2",
      "dance_3",
      "kung_fu",
      "twist",
    ],
  },
];

function findAction(id: string) {
  return G1_ACTIONS.find((action) => action.id === id);
}

export function ActionDeck({
  armed,
  onAction,
  onOpenActions,
  onEmergency,
}: ActionDeckProps) {
  return (
    <section className="operator-zone operator-zone-v2">
      <div className="operator-panels operator-panels-v2">
        {panels.map((panel) => {
          const Icon = panel.icon;

          const actions = panel.actionIds
            .map(findAction)
            .filter(
              (action): action is G1Action =>
                action !== undefined,
            );

          return (
            <article
              className="operator-panel"
              key={panel.title}
            >
              <header className="operator-panel-header">
                <div className="operator-panel-title">
                  <span className="operator-panel-icon">
                    <Icon size={16} />
                  </span>

                  <div>
                    <strong>{panel.title}</strong>
                    <span>{panel.subtitle}</span>
                  </div>
                </div>

                <button
                  className="panel-more"
                  onClick={() =>
                    onOpenActions(panel.group)
                  }
                >
                  More
                  <ChevronRight size={14} />
                </button>
              </header>

              <div className="operator-action-grid operator-action-grid-v2">
                {actions.map((action) => {
                  const isStop =
                    action.id === "stop_move";

                  const isCaution =
                    action.id === "zero_torque" ||
                    action.id === "damping";

                  return (
                    <button
                      key={action.id}
                      disabled={!armed}
                      className={[
                        "operator-action",
                        isStop ? "stop-action" : "",
                        isCaution
                          ? "caution-action"
                          : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      onClick={() => onAction(action)}
                    >
                      {action.id === "stop_move" && (
                        <Octagon size={14} />
                      )}

                      {action.id === "shake_hand" && (
                        <Hand size={14} />
                      )}

                      <span>{action.label}</span>
                    </button>
                  );
                })}
              </div>
            </article>
          );
        })}
      </div>

      <aside className="operator-safety">
        <div>
          <ShieldAlert size={18} />

          <span>
            <small>SAFETY</small>
            <strong>Immediate stop</strong>
          </span>
        </div>

        <button
          className="emergency-button-v3"
          onClick={onEmergency}
        >
          <strong>STOP</strong>
          <span>EMERGENCY</span>
        </button>
      </aside>
    </section>
  );
}
