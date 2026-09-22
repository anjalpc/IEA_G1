import {
  useEffect,
  useMemo,
  useState,
} from "react";

import { X } from "lucide-react";

import { G1_ACTIONS } from "../data/g1Actions";

import type {
  ActionGroup,
  G1Action,
} from "../data/g1Actions";

interface ActionDrawerProps {
  open: boolean;
  armed: boolean;
  initialGroup: ActionGroup;
  onClose: () => void;
  onAction: (action: G1Action) => void;
}

const groups: {
  id: ActionGroup;
  label: string;
}[] = [
  {
    id: "posture",
    label: "Core Robot",
  },
  {
    id: "locomotion",
    label: "Locomotion",
  },
  {
    id: "interaction",
    label: "Interaction",
  },
  {
    id: "performance",
    label: "Special Motions",
  },
  {
    id: "custom",
    label: "Project",
  },
];

const sourceLabels: Record<
  G1Action["source"],
  string
> = {
  loco: "LOCO SDK",
  arm: "ARM SDK",
  remote: "REMOTE",
  mimic: "MIMIC",
  custom: "PROJECT",
};

export function ActionDrawer({
  open,
  armed,
  initialGroup,
  onClose,
  onAction,
}: ActionDrawerProps) {
  const [group, setGroup] =
    useState<ActionGroup>(initialGroup);

  useEffect(() => {
    if (!open) return;

    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener(
      "keydown",
      handler,
    );

    return () =>
      window.removeEventListener(
        "keydown",
        handler,
      );
  }, [open, initialGroup, onClose]);

  const actions = useMemo(
    () =>
      G1_ACTIONS.filter(
        (action) =>
          action.group === group,
      ),
    [group],
  );

  if (!open) return null;

  return (
    <div
      className="action-drawer-backdrop"
      onPointerDown={(event) => {
        if (
          event.target ===
          event.currentTarget
        ) {
          onClose();
        }
      }}
    >
      <section className="action-drawer">
        <header className="action-drawer-header">
          <div>
            <span>UNITREE G1</span>
            <h2>Robot actions</h2>
            <p>
              Native capabilities and project
              interaction workflows.
            </p>
          </div>

          <button
            className="drawer-close"
            onClick={onClose}
            aria-label="Close robot actions"
          >
            <X size={20} />
          </button>
        </header>

        <nav className="action-tabs">
          {groups.map((item) => {
            const count =
              G1_ACTIONS.filter(
                (action) =>
                  action.group ===
                  item.id,
              ).length;

            return (
              <button
                key={item.id}
                className={
                  group === item.id
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setGroup(item.id)
                }
              >
                {item.label}
                <span>{count}</span>
              </button>
            );
          })}
        </nav>

        <div className="action-drawer-body">
          {actions.length ? (
            <div className="action-catalog">
              {actions.map((action) => (
                <button
                  key={action.id}
                  className={`action-tile ${
                    action.dangerous
                      ? "dangerous"
                      : ""
                  }`}
                  disabled={!armed}
                  onClick={() =>
                    onAction(action)
                  }
                >
                  <div className="action-tile-main">
                    <strong>
                      {action.label}
                    </strong>

                    <span>
                      {
                        sourceLabels[
                          action.source
                        ]
                      }
                    </span>
                  </div>

                  <span
                    className={`capability-tag ${action.availability}`}
                  >
                    {action.availability ===
                    "direct"
                      ? "AVAILABLE"
                      : "VERIFY"}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-action-group">
              <strong>
                No actions available
              </strong>

              <span>
                Robot-discovered actions
                will appear here.
              </span>
            </div>
          )}
        </div>

        <footer className="action-drawer-footer">
          <span>
            {armed
              ? "CONTROL ARMED"
              : "CONTROL LOCKED"}
          </span>

          <span>
            Final availability will come
            from the connected robot.
          </span>
        </footer>
      </section>
    </div>
  );
}
