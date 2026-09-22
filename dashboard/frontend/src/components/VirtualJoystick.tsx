import { useRef, useState } from "react";

export interface StickValue {
  x: number;
  y: number;
}

interface VirtualJoystickProps {
  label: string;
  description: string;
  value: StickValue;
  onChange: (value: StickValue) => void;
  disabled?: boolean;
}

export function VirtualJoystick({
  label,
  description,
  value,
  onChange,
  disabled = false,
}: VirtualJoystickProps) {
  const baseRef = useRef<HTMLDivElement>(null);
  const activePointer = useRef<number | null>(null);

  const [active, setActive] = useState(false);
  const [travel, setTravel] = useState(0);

  const updateFromPointer = (
    clientX: number,
    clientY: number,
  ) => {
    const element = baseRef.current;

    if (!element || disabled) return;

    const rect = element.getBoundingClientRect();

    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const maxRadius = rect.width * 0.30;

    setTravel(maxRadius);

    let dx = clientX - centerX;
    let dy = clientY - centerY;

    const distance = Math.hypot(dx, dy);

    if (distance > maxRadius) {
      const scale = maxRadius / distance;
      dx *= scale;
      dy *= scale;
    }

    let x = dx / maxRadius;
    let y = -dy / maxRadius;

    const deadZone = 0.06;

    if (Math.hypot(x, y) < deadZone) {
      x = 0;
      y = 0;
    }

    onChange({
      x: Math.max(-1, Math.min(1, x)),
      y: Math.max(-1, Math.min(1, y)),
    });
  };

  const release = () => {
    activePointer.current = null;
    setActive(false);

    onChange({
      x: 0,
      y: 0,
    });
  };

  return (
    <section
      className={`joystick-module ${
        disabled ? "disabled" : ""
      }`}
    >
      <div className="joystick-heading">
        <div>
          <span className="eyebrow">{label}</span>
          <p>{description}</p>
        </div>

        <span
          className={`touch-status ${
            active ? "active" : ""
          }`}
        >
          {active ? "ACTIVE" : "READY"}
        </span>
      </div>

      <div
        ref={baseRef}
        className={`joystick-base ${
          active ? "active" : ""
        }`}
        onPointerDown={(event) => {
          if (disabled) return;

          activePointer.current = event.pointerId;

          event.currentTarget.setPointerCapture(
            event.pointerId,
          );

          setActive(true);

          updateFromPointer(
            event.clientX,
            event.clientY,
          );
        }}
        onPointerMove={(event) => {
          if (
            activePointer.current !== event.pointerId
          ) {
            return;
          }

          updateFromPointer(
            event.clientX,
            event.clientY,
          );
        }}
        onPointerUp={release}
        onPointerCancel={release}
        onLostPointerCapture={release}
      >
        <div className="stick-axis axis-horizontal" />
        <div className="stick-axis axis-vertical" />

        <div className="stick-center" />

        <div
          className="joystick-knob"
          style={{
            transform: `translate(
              calc(-50% + ${value.x * travel}px),
              calc(-50% + ${-value.y * travel}px)
            )`,
          }}
        >
          <div className="knob-inner" />
        </div>
      </div>

      <div className="stick-readout">
        <div>
          <span>X</span>
          <strong>{value.x.toFixed(2)}</strong>
        </div>

        <div>
          <span>Y</span>
          <strong>{value.y.toFixed(2)}</strong>
        </div>
      </div>
    </section>
  );
}
