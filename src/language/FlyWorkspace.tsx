import { useMemo, useRef, useState } from "react";
import Scene from "../Scene";
import type { Activity, Dataset } from "../data";
import { poseAt } from "../motor/engine";
import ChatPanel, { type LiveEvent } from "./ChatPanel";
import CircuitView, { type CircuitInspection } from "./CircuitView";

export default function FlyWorkspace({ dataset }: { dataset: Dataset }) {
  const [activity, setActivity] = useState<Activity>();
  const [values, setValues] = useState<Record<string, number>>();
  const [sequence, setSequence] = useState(0);
  const [running, setRunning] = useState(false);
  const [view, setView] = useState<CircuitInspection>();
  const [reset, setReset] = useState(0);
  const metadata = useRef<CircuitInspection | undefined>(undefined);
  const ids = useRef<string[]>([]);
  const display = useMemo(
    () => (activity ? { ...dataset, activity } : dataset),
    [dataset, activity],
  );
  const pose = useMemo(() => poseAt(undefined, 0), []);
  function onLive(event: LiveEvent) {
    if (event.type === "pending") {
      setValues(undefined);
      setRunning(true);
      setView(undefined);
    }
    if (event.type === "start") {
      if (
        event.datasetId !== dataset.id ||
        event.datasetVersion !== dataset.version
      )
        throw new Error("Model and anatomy dataset mismatch");
      ids.current = event.ids!;
      metadata.current = event.inspection;
      // Unit bounds allocate a stable overlay once; these are never displayed as samples.
      setActivity({
        kind: "simulation",
        unit: "signed hidden activation",
        times: [0],
        values: Object.fromEntries(ids.current.map((id) => [id, [1]])),
      });
    }
    if (event.type === "state") {
      setValues(
        Object.fromEntries(ids.current.map((id, i) => [id, event.values![i]])),
      );
      setSequence(event.sequence!);
      if (metadata.current)
        setView({ ...metadata.current, steps: [event.step!] });
    }
    if (event.type === "end") {
      setRunning(false);
      setValues(undefined);
    }
  }
  const scene = {
    dataset: display,
    pointBudget: 5000,
    activityVisible: running && !!values,
    liveValues: values,
    liveSequence: sequence,
    motorPose: pose,
    time: 0,
    selected: null,
    region: "All regions",
    threshold: 0,
    showEdges: false,
    showBody: true,
    reset,
    onSelect: () => {},
  };
  return (
    <main className="fly-workspace" aria-label="Live FlyGPT workspace">
      <div className="fly-workspace-bar">
        <span>
          {running ? "Live computation" : "Ready"} · {sequence} computed steps ·
          continuous states, not spikes
        </span>
        <button onClick={() => setReset((r) => r + 1)}>Reset views</button>
      </div>
      <div className="fly-live-views">
        <section className="fly-anatomy">
          <div className="live-view-title">
            5k-point overview · all 512 model cells
          </div>
          <Scene {...scene} mode="brain" />
        </section>
        <section className="fly-body">
          <div className="live-view-title">
            fly · illustrative neural placement
          </div>
          <Scene {...scene} mode="fly" />
        </section>
        <section className="fly-circuit" aria-label="Live neural circuit">
          <div className="live-view-title">
            {running ? "live pathways" : "last computed pathways"} · 24-cell
            detail
          </div>
          {view ? (
            <CircuitView data={view} embedded />
          ) : (
            <p className="circuit-empty">
              Send a message to see cell states and next-token predictions here.
            </p>
          )}
        </section>
      </div>
      <ChatPanel compact onLive={onLive} />
      <p className="live-note">
        Cell brightness follows absolute activation on a fixed 0–1 scale.
        Anatomy returns to idle when generation ends. Visible-step mode pauses
        computation between tokens; it does not replay stored activity. Tokens/s
        includes that delay.
      </p>
    </main>
  );
}
