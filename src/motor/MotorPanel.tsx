import type { MotorCommand, MotorPose, MotorTrack } from "./types";
import { controllers } from "../controllers/registry";
import type { Dataset } from "../data";
export type MotorMode = "off" | "replay" | string;
export default function MotorPanel({
  physicsAvailable,
  dataset,
  mode,
  onMode,
  command,
  onCommand,
  track,
  pose,
  error,
  onExport,
}: {
  physicsAvailable?: boolean;
  dataset: Dataset;
  mode: MotorMode;
  onMode: (m: MotorMode) => void;
  command: MotorCommand;
  onCommand: (v: MotorCommand) => void;
  track?: MotorTrack;
  pose: MotorPose;
  error: string;
  onExport: () => void;
}) {
  return (
    <section className="motor-panel" aria-label="Motor simulation">
      <div className="motor-heading">
        <div>
          <span className="eyebrow">MOTOR LAB</span>
          <h3>Give the circuit a body.</h3>
        </div>
        <select
          aria-label="Motor controller"
          value={mode}
          onChange={(e) => onMode(e.target.value)}
        >
          <option value="off">Motor off</option>
          {physicsAvailable && <option value="physics">Physical fly replay</option>}
          {controllers.map((c) => (
            <option key={c.id} value={c.id} disabled={!!c.unsupported(dataset)}>
              {c.label}
            </option>
          ))}
          {dataset.motor && (
            <option value="replay">Imported motor replay</option>
          )}
        </select>
      </div>
      <p className="motor-provenance">
        {mode === "off"
          ? "Select a controller, then use Play below."
          : mode === "physics" ? "MuJoCo contact physics · experimental MaleCNS rate model · computed replay" : mode === "replay"
            ? `${track?.kind} · ${track?.model}`
            : "Servo dynamics + kinematic walking · no contact or flight physics"}
      </p>
      {(mode === "manual" || mode === "walking-circuit") && (
        <div className="motor-controls">
          {(
            [
              {
                key: "forward",
                label: "Forward",
                unit: "mm/s",
                min: -3,
                max: 3,
              },
              { key: "turn", label: "Turn", unit: "rad/s", min: -2, max: 2 },
              { key: "wing", label: "Wing spread", unit: "", min: 0, max: 1 },
            ] as const
          ).map((c) => (
            <label key={c.key}>
              {c.label}
              <output>
                {command[c.key].toFixed(2)} {c.unit}
              </output>
              <input
                aria-label={c.label}
                type="range"
                min={c.min}
                max={c.max}
                step=".05"
                value={command[c.key]}
                onChange={(e) =>
                  onCommand({ ...command, [c.key]: +e.target.value })
                }
              />
            </label>
          ))}
        </div>
      )}
      {mode === "neural-readout" && (
        <p className="motor-provenance">
          Six synthetic descending channels → left/right drive → gait. These are
          test mappings, not identified biological motor neurons.
        </p>
      )}
      {mode === "manual" && (
        <p className="motor-provenance">
          Direct motor commands do not simulate neural firing. On the synthetic
          dataset, select Walking neural circuit for a coupled activity demo.
        </p>
      )}
      {mode === "walking-circuit" && (
        <p className="motor-provenance">
          24 synthetic oscillator rates drive the forward command. The
          highlighted neuron trace records this controller's output. These are
          rates, not measured spikes.
        </p>
      )}
      {!dataset.activity && (
        <p className="motor-provenance">
          No neural activity loaded. Manual motion does not create brain
          activity.
        </p>
      )}
      {error && (
        <p role="alert" className="motor-error">
          {error}
        </p>
      )}
      <div className="motor-telemetry">
        <output data-testid="motor-position">
          x {pose.position[0].toFixed(2)} · z {pose.position[2].toFixed(2)} mm
        </output>
        <output>yaw {pose.heading.toFixed(2)} rad</output>
        {mode !== "physics" && <output data-testid="motor-joint">
          LF sweep {pose.joints["LF.sweep"].toFixed(3)} rad
        </output>}
        <button onClick={onExport} disabled={!track}>
          {mode === "physics" ? "Export full physics run ↗" : "Export motor run ↗"}
        </button>
      </div>
    </section>
  );
}
