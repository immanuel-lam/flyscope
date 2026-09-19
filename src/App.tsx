import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity as ActivityIcon,
  ArrowDownToLine,
  ArrowUpRight,
  Box,
  ChevronRight,
  CircleHelp,
  Dna,
  Expand,
  Eye,
  Focus,
  Layers3,
  Network,
  Pause,
  Play,
  RotateCcw,
  Search,
  Upload,
  X,
} from "lucide-react";
import Scene from "./Scene";
import type {
  CatalogInfo,
  Detail,
  ConnectionResult,
  GraphInfo,
  LargeMessage,
} from "./large/types";
import MotorPanel from "./motor/MotorPanel";
import { controllers } from "./controllers/registry";
import { simulateExperiment } from "./motor/experiment";
import { poseAt } from "./motor/engine";
import type { MotorCommand } from "./motor/types";
import {
  activityAt,
  activityRange,
  demoDataset,
  palette,
  parseDataset,
  parseSWC,
} from "./data";
import type { Dataset } from "./data";

import ChatPanel from "./language/ChatPanel";
import type { Activity } from "./data";
import PhysicsPanel from "./physics/PhysicsPanel";
import { physicsTelemetry, type PhysicsRun } from "./physics/types";
const demo = demoDataset();
function download(d: Dataset) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(d)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `${d.id}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export default function App() {
  const [dataset, setDataset] = useState<Dataset>(demo);
  const [catalogInfo, setCatalogInfo] = useState<CatalogInfo>();
  const [graphInfo, setGraphInfo] = useState<GraphInfo>();
  const [outgoingCounts, setOutgoingCounts] = useState<Uint32Array>();
  const [detail, setDetail] = useState<Detail>();
  const [connections, setConnections] = useState<ConnectionResult>();
  const [detailStatus, setDetailStatus] = useState("");
  const largeWorker = useRef<Worker | null>(null);
  const largeActive = useRef(false);
  const selectionRef = useRef<string | null>(null);
  useEffect(() => () => largeWorker.current?.terminate(), []);
  const [playing, setPlaying] = useState(false);
  const [inspectActivity, setInspectActivity] = useState(false);
  const [sidePanel, setSidePanel] = useState<"chat" | "neurons">("chat");
  const [chatActivity, setChatActivity] = useState<Activity>();
  const [physicsRun, setPhysicsRun] = useState<PhysicsRun>();
  const [motorMode, setMotorMode] = useState("off");
  const [motorCommand, setMotorCommand] = useState<MotorCommand>({
    forward: 1,
    turn: 0,
    wing: 0,
  });
  const motorResult = useMemo(() => {
    if (motorMode === "off") return {};
    if (motorMode === "physics" && physicsRun)
      return {
        track: physicsTelemetry(physicsRun),
        activity: physicsRun.activity,
      };
    if (motorMode === "replay") return { track: dataset.motor };
    try {
      const driver = controllers.find((c) => c.id === motorMode);
      if (!driver) throw new Error("Unknown motor controller.");
      const run = simulateExperiment(
        dataset,
        driver,
        motorCommand,
        Math.min(60, Math.max(dataset.activity?.times.at(-1) ?? 30, 0.01)),
      );
      return { track: run.motor, activity: run.activity };
    } catch (e) {
      return { error: (e as Error).message };
    }
  }, [dataset, motorMode, motorCommand, physicsRun]);
  const displayDataset = useMemo(
    () =>
      (chatActivity ?? motorResult.activity)
        ? { ...dataset, activity: chatActivity ?? motorResult.activity }
        : dataset,
    [dataset, motorResult.activity, chatActivity],
  );
  const [time, setTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [region, setRegion] = useState("All regions");
  const [selected, setSelected] = useState<string | null>("demo-0");
  const [query, setQuery] = useState("");
  const [threshold, setThreshold] = useState(0);
  const [showEdges, setShowEdges] = useState(false);
  const [showBody, setShowBody] = useState(true);
  const [reset, setReset] = useState(0);
  const [view, setView] = useState<"split" | "brain" | "fly">("split");
  const [panel, setPanel] = useState<"explore" | "sources">("explore");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const request = useRef(0);
  const regions = useMemo(
    () => [...new Set(dataset.neurons.map((n) => n.region))],
    [dataset],
  );
  const range = useMemo(() => activityRange(displayDataset), [displayDataset]);
  const duration = Math.max(
    displayDataset.activity?.times.at(-1) ?? 0,
    motorResult.track?.times.at(-1) ?? 0,
  );
  const motorPose = useMemo(
    () => poseAt(motorResult.track, time),
    [motorResult.track, time],
  );
  const changeMotor = (mode: string) => {
    setChatActivity(undefined);
    setMotorMode(mode);
    if (mode === "walking-circuit") setSelected("demo-5");
    setPlaying(false);
    setTime(0);
  };
  const filtered = useMemo(
    () =>
      dataset.neurons.filter(
        (n) =>
          (region === "All regions" || n.region === region) &&
          `${n.id} ${n.label} ${n.region}`
            .toLowerCase()
            .includes(query.toLowerCase()),
      ),
    [dataset, query, region],
  );
  selectionRef.current = selected;
  const neuronIndex = useMemo(
    () => new Map(dataset.neurons.map((n, i) => [n.id, i])),
    [dataset],
  );
  const neuron = selected
    ? dataset.neurons[neuronIndex.get(selected) ?? -1]
    : undefined;
  const regionCounts = useMemo(() => {
    const counts = new Map<string, number>();
    dataset.neurons.forEach((n) =>
      counts.set(n.region, (counts.get(n.region) ?? 0) + 1),
    );
    return counts;
  }, [dataset]);
  const isFull = dataset.id === "male-cns-v1-full";
  useEffect(() => {
    setDetail(undefined);
    setConnections(undefined);
    if (!isFull || !selected) return;
    setDetailStatus("Loading source skeleton…");
    largeWorker.current?.postMessage({ kind: "detail", id: selected });
    largeWorker.current?.postMessage({ kind: "connections", id: selected });
  }, [selected, isFull]);
  function loadFull() {
    ++request.current;
    largeActive.current = true;
    setLoading(true);
    setError("");
    largeWorker.current?.terminate();
    const worker = new Worker(
      new URL("./large/loader.worker.ts", import.meta.url),
      { type: "module" },
    );
    largeWorker.current = worker;
    worker.onmessage = ({ data }: { data: LargeMessage }) => {
      if (!largeActive.current || worker !== largeWorker.current) return;
      if (data.kind === "catalog") {
        apply(data.dataset, true);
        setCatalogInfo(data.info);
        setGraphInfo(data.graph);
        setOutgoingCounts(data.counts);
        setLoading(false);
      } else if (
        data.kind === "detail" &&
        data.detail.id === selectionRef.current
      ) {
        setDetail(data.detail);
        setDetailStatus(
          `${data.detail.segments.length / 6} / ${data.detail.totalSegments} segments · ${(data.detail.bytes / 1024).toFixed(0)} KB`,
        );
      } else if (
        data.kind === "connections" &&
        data.result.id === selectionRef.current
      )
        setConnections(data.result);
      else if (data.kind === "error") {
        if (data.request === "catalog") {
          setLoading(false);
          setError(data.message);
        } else if (data.id === selectionRef.current)
          setDetailStatus(data.message);
      }
    };
    worker.onerror = () => {
      setLoading(false);
      setError("Dataset worker failed. Check the prepared MaleCNS assets.");
    };
    worker.postMessage({ kind: "catalog" });
  }
  const value = neuron
    ? activityAt(displayDataset, neuron.id, time)
    : undefined;
  const active = displayDataset.activity
    ? dataset.neurons.reduce(
        (s, n) =>
          s +
          ((activityAt(displayDataset, n.id, time) ?? -Infinity) >
          range[0] + 0.5 * (range[1] - range[0])
            ? 1
            : 0),
        0,
      )
    : 0;
  const apply = (d: Dataset, full = false) => {
    largeActive.current = full;
    if (!full) {
      setCatalogInfo(undefined);
      setGraphInfo(undefined);
      setOutgoingCounts(undefined);
      setDetail(undefined);
      setConnections(undefined);
    }
    setDataset(d);
    setChatActivity(undefined);
    setPhysicsRun(undefined);
    setMotorMode(d.motor ? "replay" : "off");
    setTime(0);
    setPlaying(false);
    setSelected(d.neurons[0]?.id ?? null);
    setRegion("All regions");
    setQuery("");
    setThreshold(0);
    setReset((n) => n + 1);
    setError("");
  };
  useEffect(() => {
    if (!playing) return;
    let frame: number;
    let prev = performance.now();
    const tick = (now: number) => {
      const dt = Math.max(0, Math.min((now - prev) / 1000, 0.2));
      prev = now;
      setTime((t) => {
        const next = t + dt * speed;
        if (next >= duration) {
          setPlaying(false);
          return duration;
        }
        return next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, speed, duration]);
  async function loadReference() {
    const token = ++request.current;
    setLoading(true);
    setError("");
    try {
      const texts = await Promise.all(
        ["12781", "556329"].map(async (id) => {
          const r = await fetch(`/${id}.swc`);
          if (!r.ok)
            throw new Error("Could not load the bundled reference skeletons.");
          return r.text();
        }),
      );
      if (token !== request.current) return;
      apply(
        parseDataset({
          schemaVersion: 1,
          id: "malecns-dnge104",
          name: "MaleCNS · DNge104 pair",
          source:
            "HHMI Janelia, Cambridge, MRC LMB and Google Research. CC BY 4.0. https://male-cns.janelia.org/download/",
          version: "male-cns:v1.0",
          geometry: "reconstruction",
          coordinateSpace: "Male CNS EM",
          units: "8 nm voxels",
          neurons: texts.map((t, i) =>
            parseSWC(
              t,
              ["12781", "556329"][i],
              ["DNge104_R", "DNge104_L"][i],
              "Descending",
            ),
          ),
          edges: [],
        }),
      );
    } catch (e) {
      if (token === request.current) setError((e as Error).message);
    } finally {
      if (token === request.current) setLoading(false);
    }
  }
  async function importFile(f: File) {
    const token = ++request.current;
    setLoading(false);
    try {
      if (f.size > 30 * 1024 * 1024)
        throw new Error("File exceeds 30 MB. Export a smaller subset.");
      const text = await f.text();
      if (token !== request.current) return;
      if (f.name.endsWith(".swc")) {
        apply(
          parseDataset({
            schemaVersion: 1,
            id: f.name,
            name: f.name,
            source: "User-provided SWC; source and units not verified.",
            version: "local",
            geometry: "reconstruction",
            coordinateSpace: "Unspecified source space",
            units: "unspecified",
            neurons: [parseSWC(text, f.name)],
            edges: [],
          }),
        );
      } else apply(parseDataset(JSON.parse(text)));
    } catch (e) {
      if (token === request.current) setError((e as Error).message);
    }
  }
  const shared = {
    activityVisible: playing || inspectActivity,
    physics: motorMode === "physics" ? physicsRun : undefined,
    dataset: displayDataset,
    detail: isFull ? detail : undefined,
    connectionEdges: isFull ? connections?.edges : undefined,
    motorPose,
    time,
    selected,
    region,
    threshold,
    showEdges,
    showBody,
    reset,
    onSelect: setSelected,
  };
  useEffect(() => {
    if (new URLSearchParams(location.search).get("dataset") !== "demo")
      loadFull();
  }, []);
  return (
    <div className="app">
      <header className="topbar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPanel("explore");
          }}
        >
          <Dna size={25} />
          <span>
            flyscope<span className="brand-dot">.</span>
          </span>
        </a>
        <div className="top-divider" />
        <span className="top-description">Neural observatory</span>
        <span className="version">WORKBENCH / 0.1</span>
        <nav>
          <button
            className={panel === "explore" ? "nav-active" : ""}
            onClick={() => setPanel("explore")}
          >
            Explore
          </button>
          <button
            className={panel === "sources" ? "nav-active" : ""}
            onClick={() => setPanel("sources")}
          >
            Data & research <ArrowUpRight size={13} />
          </button>
        </nav>
        <button className="import-button" onClick={() => file.current?.click()}>
          <Upload size={14} /> Import dataset
        </button>
        <input
          ref={file}
          type="file"
          accept=".json,.swc"
          hidden
          onChange={(e) => {
            if (e.target.files?.[0]) void importFile(e.target.files[0]);
            e.target.value = "";
          }}
        />
      </header>
      <div className="workspace-heading">
        <div>
          <div className="eyebrow">
            DROSOPHILA MELANOGASTER <span>/</span> CONNECTOME EXPLORER
          </div>
          <h1>Explore the fruit fly nervous system.</h1>
        </div>
        <div className="dataset-status">
          <span
            className={`status-dot ${dataset.geometry === "reconstruction" ? "real" : ""}`}
          />
          <div>
            <strong>{dataset.name}</strong>
            <small>{`${dataset.geometry === "synthetic" ? "Synthetic anatomy" : "Reconstructed anatomy"} · ${displayDataset.activity ? displayDataset.activity.kind + " activity" : "structure only"}`}</small>
          </div>
        </div>
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
          <button aria-label="Dismiss error" onClick={() => setError("")}>
            <X size={16} />
          </button>
        </div>
      )}
      {panel === "sources" ? (
        <section className="research">
          <div>
            <div className="eyebrow">START WITH THE EVIDENCE</div>
            <h2>Wiring is the starting point.</h2>
            <p>
              A connectome describes structure. Neural activity needs a separate
              recording or model. Language learning needs an input encoding, a
              learning rule and an evaluation task.
            </p>
            <h3>Three independent layers</h3>
            <ol>
              <li>
                <strong>Structure</strong> — neuron IDs, skeletons, connections
                and coordinate space.
              </li>
              <li>
                <strong>Activity</strong> — per-neuron values with timestamps,
                units and provenance.
              </li>
              <li>
                <strong>Body</strong> — a separate 3D model. Brain placement
                here is illustrative; motion is not inferred from neural
                activity.
              </li>
            </ol>
            <h3>A testable learning experiment</h3>
            <p>
              Start with a small recurrent circuit constrained by the connection
              graph, plus a token encoder and decoder. Compare it against the
              same model with shuffled connections and a small standard
              baseline. A fly connectome is not evidence of language ability.
            </p>
            <button className="primary" onClick={() => setPanel("explore")}>
              Open the workbench <ChevronRight size={16} />
            </button>
          </div>
          <div className="source-list">
            <a
              href="https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/"
              target="_blank"
              rel="noreferrer"
            >
              <span>01 / GOOGLE RESEARCH</span>
              <h3>
                Male brain + nerve cord <ArrowUpRight />
              </h3>
              <p>
                September 2026 announcement. Over 166,000 neurons and 125
                million synaptic connections.
              </p>
            </a>
            <a
              href="https://male-cns.janelia.org/download/"
              target="_blank"
              rel="noreferrer"
            >
              <span>02 / MALECNS v1.0</span>
              <h3>
                Public source data <ArrowUpRight />
              </h3>
              <p>
                The bundled reference contains DNge104_R (12781) and DNge104_L
                (556329). Skeleton coordinates are in 8 nm units. CC BY 4.0.
              </p>
            </a>
            <a
              href="https://codex.flywire.ai/faq"
              target="_blank"
              rel="noreferrer"
            >
              <span>03 / FLYWIRE</span>
              <h3>
                Other fly datasets <ArrowUpRight />
              </h3>
              <p>
                FAFB, BANC and other datasets have separate versions and
                coordinate spaces. Use source-specific exports.
              </p>
            </a>
            <a
              href="https://github.com/philshiu/Drosophila_brain_model"
              target="_blank"
              rel="noreferrer"
            >
              <span>04 / SHIU ET AL.</span>
              <h3>
                A model you can connect <ArrowUpRight />
              </h3>
              <p>
                A leaky integrate-and-fire model provides spike times and rates
                keyed by FlyWire neuron IDs.
              </p>
            </a>
          </div>
        </section>
      ) : (
        <main className="workbench">
          <aside className="sidebar">
            <div className="section-label">
              <span>PROJECT DATA</span>
              <Layers3 size={14} />
            </div>
            <label className="field-label" htmlFor="dataset">
              Dataset
            </label>
            <select
              id="dataset"
              value={
                dataset.id === "male-cns-v1-full"
                  ? "full"
                  : dataset.id === "synthetic-workbench"
                    ? "demo"
                    : dataset.id === "malecns-dnge104"
                      ? "reference"
                      : "custom"
              }
              onChange={(e) => {
                if (e.target.value === "demo") {
                  ++request.current;
                  setLoading(false);
                  apply(demo);
                } else if (e.target.value === "reference") void loadReference();
                else if (e.target.value === "full") loadFull();
              }}
            >
              <option value="demo">Synthetic fly circuit</option>
              <option value="full">MaleCNS · full catalog</option>
              <option value="reference">MaleCNS · 2-neuron sample</option>
              {![
                "synthetic-workbench",
                "malecns-dnge104",
                "male-cns-v1-full",
              ].includes(dataset.id) && (
                <option value="custom">{dataset.name}</option>
              )}
            </select>
            <p className="subtle dataset-note">
              {loading
                ? "Loading dataset in background…"
                : dataset.geometry === "synthetic"
                  ? "A procedural test fixture. No measured anatomy."
                  : displayDataset.activity
                    ? `Reconstructed skeletons with ${displayDataset.activity.kind} activity.`
                    : isFull
                      ? "Source positions for all classified neurons. Select a cell for its skeleton."
                      : "Reconstructed skeletons. No activity loaded."}
            </p>
            <div className="dataset-count">
              <div>
                <strong>{dataset.neurons.length.toLocaleString()}</strong>
                <span>{isFull ? "classified neurons" : "neurons"}</span>
              </div>
              <div>
                <strong>
                  {(isFull
                    ? (graphInfo?.edges ?? 0)
                    : dataset.edges.length
                  ).toLocaleString()}
                </strong>
                <span>directed edges</span>
              </div>
            </div>
            <div className="section-label spaced">
              <span>{isFull ? "NEURON CLASSES" : "BRAIN REGIONS"}</span>
              <span>{regions.length}</span>
            </div>
            <button
              className={`region-row ${region === "All regions" ? "selected" : ""}`}
              onClick={() => setRegion("All regions")}
            >
              <Network size={14} />
              <span>All regions</span>
              <small>{dataset.neurons.length}</small>
            </button>
            {regions.map((r, i) => (
              <button
                className={`region-row ${region === r ? "selected" : ""}`}
                key={r}
                onClick={() => setRegion(r)}
              >
                <i style={{ background: palette[i % 6] }} />
                <span>{r}</span>
                <small>{regionCounts.get(r)}</small>
              </button>
            ))}
            <div className="section-label spaced">
              <span>DISPLAY LAYERS</span>
              <Eye size={14} />
            </div>
            <label className="toggle-row">
              <span>Connection graph</span>
              <input
                type="checkbox"
                checked={showEdges}
                disabled={!dataset.edges.length && !connections?.edges.length}
                onChange={(e) => setShowEdges(e.target.checked)}
              />
            </label>
            <label className="toggle-row">
              <span>Fly anatomy</span>
              <input
                type="checkbox"
                checked={showBody}
                onChange={(e) => setShowBody(e.target.checked)}
              />
            </label>
            <p className="subtle small">
              {isFull
                ? "Selected neuron: strongest 2,000 outgoing links at most."
                : "Connections show in the unfiltered view."}
            </p>
            <label className="threshold-label" htmlFor="threshold">
              Activity threshold <span>{Math.round(threshold * 100)}%</span>
            </label>
            <input
              id="threshold"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              disabled={!displayDataset.activity}
              onChange={(e) => setThreshold(+e.target.value)}
            />
            <p className="subtle small">Below-threshold neurons are dimmed.</p>
            <button
              className="export"
              disabled={isFull}
              title={
                isFull
                  ? "Use the source files in data/malecns and the prepared catalog; bulk JSON export omits the graph."
                  : undefined
              }
              onClick={() => download(dataset)}
            >
              <ArrowDownToLine size={14} /> Export current dataset
            </button>
            <div className="sidebar-foot">
              <CircleHelp size={15} />
              <span>
                Bring your own circuit.
                <br />
                JSON datasets or SWC skeletons.
              </span>
            </div>
          </aside>
          <section className="visual-area">
            {isFull && catalogInfo && (
              <div className="large-status">
                <strong>
                  {catalogInfo.total.toLocaleString()} cells ·{" "}
                  {catalogInfo.positioned.toLocaleString()} source positions
                </strong>
                <span>
                  Skeletons on selection · 20k points while orbiting; all cells
                  at rest.
                </span>
                <small>
                  Catalog {(catalogInfo.bytes / 1048576).toFixed(1)} MB · worker{" "}
                  {catalogInfo.loadMs.toFixed(0)} ms ·{" "}
                  {connections
                    ? `${connections.total.toLocaleString()} outgoing partners for selected cell`
                    : "Loading selected cell…"}
                </small>
              </div>
            )}
            <div className="view-toolbar">
              <div className="segmented">
                <button
                  className={view === "split" ? "active" : ""}
                  onClick={() => setView("split")}
                >
                  <Layers3 size={14} /> Split view
                </button>
                <button
                  className={view === "brain" ? "active" : ""}
                  onClick={() => setView("brain")}
                >
                  <Network size={14} /> Neurons
                </button>
                <button
                  className={view === "fly" ? "active" : ""}
                  onClick={() => setView("fly")}
                >
                  <Box size={14} /> Fly
                </button>
              </div>
              <button
                className="icon-button"
                aria-label="Reset cameras"
                onClick={() => setReset((n) => n + 1)}
              >
                <Focus size={17} />
              </button>
            </div>
            <div className={`viewports view-${view}`}>
              {view !== "fly" && (
                <section className="viewport brain-viewport">
                  <div className="viewport-heading">
                    <span>
                      <i className="tiny-dot" />
                      NEURAL STRUCTURE
                    </span>
                    <button
                      aria-label="Expand neuron view"
                      onClick={() =>
                        setView(view === "brain" ? "split" : "brain")
                      }
                    >
                      <Expand size={14} />
                    </button>
                  </div>
                  <Scene {...shared} mode="brain" />
                  <div className="viewport-caption">
                    <span>
                      {isFull
                        ? "Full catalog · source position overview"
                        : dataset.geometry === "synthetic"
                          ? "Illustrative morphology"
                          : "Source reconstruction"}
                    </span>
                    <small>
                      {dataset.neurons.length.toLocaleString()} neurons ·{" "}
                      {dataset.units}
                    </small>
                  </div>
                  <div className="axes">
                    <span>Y</span>
                    <span>Z</span>
                    <span>X</span>
                  </div>
                </section>
              )}
              {view !== "brain" && (
                <section className="viewport fly-viewport">
                  <div className="viewport-heading">
                    <span>
                      <i className="tiny-dot" />
                      EMBODIED VIEW
                    </span>
                    <button
                      aria-label="Expand fly view"
                      onClick={() => setView(view === "fly" ? "split" : "fly")}
                    >
                      <Expand size={14} />
                    </button>
                  </div>
                  <Scene {...shared} mode="fly" />
                  <div className="body-note">
                    <span className="status-dot" />{" "}
                    {displayDataset.activity
                      ? "Shared neural activity"
                      : "Structure only"}
                  </div>
                  <div className="viewport-caption">
                    <span>
                      {motorMode === "physics"
                        ? "NeuroMechFly · actual physical rig"
                        : "Procedural fly model"}
                    </span>
                    <small>
                      {motorMode === "physics"
                        ? "MuJoCo body transforms · neural placement illustrative"
                        : motorResult.track
                          ? "Motor pose replay · illustrative anatomy"
                          : isFull
                            ? "20k-point CNS overview · illustrative placement"
                            : "Illustrative placement · motor off"}
                    </small>
                  </div>
                </section>
              )}
            </div>
            <label className="activity-inspection">
              <input
                type="checkbox"
                checked={inspectActivity}
                onChange={(e) => setInspectActivity(e.target.checked)}
              />{" "}
              Show activity while paused
            </label>
            <div className="canvas-bottom">
              <span>
                Drag to orbit <b>·</b> Scroll to zoom <b>·</b> Click a neuron to
                inspect
              </span>
              <div>
                <i className="legend-swatch" />
                Low <span className="heat-legend" /> High{" "}
                <small>
                  {motorMode === "physics"
                    ? "Point cloud retained · glow: |state| / cell peak"
                    : (displayDataset.activity?.unit ?? "No activity")}
                </small>
              </div>
            </div>
            {isFull && (
              <PhysicsPanel
                run={physicsRun}
                time={time}
                onRun={(run) => {
                  if (
                    run.datasetId !== dataset.id ||
                    run.datasetVersion !== dataset.version
                  ) {
                    setError("Physics run does not match this dataset");
                    return;
                  }
                  setChatActivity(undefined);
                  setPhysicsRun(run);
                  setMotorMode("physics");
                  setTime(0);
                  setPlaying(false);
                  const id = Object.keys(run.activity.values).sort((a, b) => {
                    const spread = (id: string) => {
                      const values = run.activity.values[id].slice(
                        Math.floor(run.times.length * 0.2),
                      );
                      return Math.max(...values) - Math.min(...values);
                    };
                    return spread(b) - spread(a);
                  })[0];
                  if (id) setSelected(id);
                }}
              />
            )}
            <MotorPanel
              dataset={displayDataset}
              physicsAvailable={!!physicsRun}
              mode={motorMode}
              onMode={changeMotor}
              command={motorCommand}
              onCommand={(command) => {
                setMotorCommand(command);
                setPlaying(false);
                setTime(0);
              }}
              track={motorResult.track}
              pose={motorPose}
              error={motorResult.error ?? ""}
              onExport={() =>
                motorMode === "physics" && physicsRun
                  ? (() => {
                      const link = document.createElement("a");
                      const url = URL.createObjectURL(
                        new Blob([JSON.stringify(physicsRun)], {
                          type: "application/json",
                        }),
                      );
                      link.href = url;
                      link.download = "flyscope-physics-run.json";
                      link.click();
                      setTimeout(() => URL.revokeObjectURL(url), 1000);
                    })()
                  : download({
                      ...dataset,
                      name: isFull
                        ? "MaleCNS selected-neuron motor export"
                        : dataset.name,
                      neurons: isFull && neuron ? [neuron] : dataset.neurons,
                      motor: motorResult.track,
                      activity: displayDataset.activity,
                    })
              }
            />
            <div className="timeline">
              <div className="timeline-top">
                <span>
                  <ActivityIcon size={14} /> ACTIVITY + MOTOR TIME
                </span>
                <span className="activity-badge">
                  {displayDataset.activity
                    ? `${displayDataset.activity?.kind} signal`
                    : "No activity loaded"}
                </span>
                <span className="time-code">
                  {time.toFixed(2)} <small>/ {duration.toFixed(2)} s</small>
                </span>
              </div>
              <div className="transport">
                <button
                  className="play"
                  aria-label={playing ? "Pause activity" : "Play activity"}
                  disabled={duration === 0}
                  onClick={() => {
                    if (time >= duration) setTime(0);
                    setPlaying((v) => !v);
                  }}
                >
                  {playing ? <Pause size={17} /> : <Play size={17} />}
                </button>
                <button
                  className="icon-button"
                  aria-label="Rewind activity"
                  onClick={() => {
                    setTime(0);
                    setPlaying(false);
                  }}
                >
                  <RotateCcw size={15} />
                </button>
                <div className="scrubber">
                  <input
                    aria-label="Activity time"
                    type="range"
                    min="0"
                    max={duration || 1}
                    step="0.01"
                    value={time}
                    disabled={duration === 0}
                    onChange={(e) => setTime(+e.target.value)}
                  />
                  <div>
                    <span>0 s</span>
                    <span>{(duration / 4).toFixed(1)}</span>
                    <span>{(duration / 2).toFixed(1)}</span>
                    <span>{(duration * 0.75).toFixed(1)}</span>
                    <span>{duration.toFixed(1)} s</span>
                  </div>
                </div>
                <select
                  aria-label="Playback speed"
                  value={speed}
                  onChange={(e) => setSpeed(+e.target.value)}
                >
                  {[0.25, 0.5, 1, 2, 4].map((s) => (
                    <option key={s} value={s}>
                      {s}×
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </section>
          <aside className="inspector">
            {isFull && (
              <div className="side-tabs" role="tablist" aria-label="Side panel">
                <button
                  role="tab"
                  aria-selected={sidePanel === "chat"}
                  onClick={() => setSidePanel("chat")}
                >
                  FlyGPT
                </button>
                <button
                  role="tab"
                  aria-selected={sidePanel === "neurons"}
                  onClick={() => setSidePanel("neurons")}
                >
                  Neuron inspector
                </button>
              </div>
            )}
            <div hidden={isFull && sidePanel !== "chat"}>
              {" "}
              {isFull && (
                <ChatPanel />
              )}
            </div>
            <div
              className="inspector-content"
              hidden={isFull && sidePanel !== "neurons"}
            >
              <div className="section-label">
                <span>NEURON INSPECTOR</span>
                <Focus size={14} />
              </div>
              <div className="search-field">
                <Search size={14} />
                <input
                  aria-label="Find neurons"
                  placeholder="Find neuron or ID…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                {query && (
                  <button
                    aria-label="Clear search"
                    onClick={() => setQuery("")}
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
              <div className="neuron-list" aria-label="Neurons">
                {filtered.slice(0, 80).map((n) => (
                  <button
                    key={n.id}
                    className={selected === n.id ? "chosen" : ""}
                    onClick={() => setSelected(n.id)}
                  >
                    <i
                      style={{
                        background: palette[regions.indexOf(n.region) % 6],
                      }}
                    />
                    <span>{n.label}</span>
                    <ChevronRight size={12} />
                  </button>
                ))}
                {!filtered.length && (
                  <p className="subtle">No neurons match this search.</p>
                )}
              </div>
              <p className="list-count">
                {Math.min(filtered.length, 80)} of{" "}
                {filtered.length.toLocaleString()} matches
              </p>
              {neuron && (
                <div className="neuron-detail">
                  {isFull && (
                    <>
                      <p className="subtle small">{detailStatus}</p>
                      <p className="subtle small">
                        Position: {neuron.positionKind}. Status: {neuron.status}
                        .
                      </p>
                    </>
                  )}
                  <span className="eyebrow">SELECTED NEURON</span>
                  <h2>{neuron.label}</h2>
                  <code>{neuron.id}</code>
                  <dl>
                    <div>
                      <dt>Region</dt>
                      <dd>{neuron.region}</dd>
                    </div>
                    <div>
                      <dt>Geometry</dt>
                      <dd>
                        {dataset.geometry === "synthetic"
                          ? "Synthetic"
                          : "Reconstructed"}
                      </dd>
                    </div>
                    <div>
                      <dt>Segments</dt>
                      <dd>
                        {(isFull
                          ? (detail?.totalSegments ?? 0)
                          : (neuron.skeleton?.length ?? 0) / 6
                        ).toLocaleString()}
                      </dd>
                    </div>
                    <div>
                      <dt>Outgoing edges</dt>
                      <dd>
                        {isFull
                          ? (outgoingCounts?.[
                              neuronIndex.get(neuron.id) ?? -1
                            ] ?? 0)
                          : dataset.edges.filter((e) => e.source === neuron.id)
                              .length}
                      </dd>
                    </div>
                  </dl>
                  <div className="signal-title">
                    <span>Activity</span>
                    <strong>
                      {value === undefined ? "—" : value.toFixed(3)}{" "}
                      <small>{displayDataset.activity?.unit}</small>
                    </strong>
                  </div>
                  <Trace
                    dataset={displayDataset}
                    id={neuron.id}
                    time={time}
                    range={range}
                  />
                  <div className="signal-axis">
                    <span>0 s</span>
                    <span>{duration.toFixed(1)} s</span>
                  </div>
                  <p className="subtle small">
                    {displayDataset.activity
                      ? motorMode === "physics"
                        ? "Glow shows each recorded cell relative to its own peak. Trace values are unchanged. Rates are not spikes."
                        : `${displayDataset.activity.kind} values. Color uses the dataset’s full value range.`
                      : "Structure only. Import a JSON dataset with activity to play a recording or simulation."}
                  </p>
                </div>
              )}
              <div className="scope-note">
                <span className="eyebrow">IN THIS VIEW</span>
                <strong>
                  {displayDataset.activity
                    ? `${active} / ${Object.keys(displayDataset.activity.values).length}`
                    : "Structure only"}
                </strong>
                <span>
                  {displayDataset.activity
                    ? "recorded neurons above 50% of the activity range"
                    : "No firing activity is inferred"}
                </span>
              </div>
            </div>
          </aside>
        </main>
      )}
      <footer>
        <span>
          <span className="status-dot real" /> Local workspace <b>/</b>{" "}
          {dataset.geometry === "synthetic"
            ? "Synthetic fixture"
            : dataset.version}
        </span>
        <span>
          Structure ≠ activity <b>·</b> Model outputs need validation
        </span>
      </footer>
    </div>
  );
}
function Trace({
  dataset,
  id,
  time,
  range,
}: {
  dataset: Dataset;
  id: string;
  time: number;
  range: [number, number];
}) {
  const a = dataset.activity;
  const vals = a?.values[id];
  if (!a || !vals) return <div className="empty-trace">No activity data</div>;
  const [min, max] = range;
  const duration = a.times.at(-1) || 1;
  const step = Math.max(1, Math.ceil(vals.length / 400));
  const points = vals
    .map((v, i) =>
      i % step === 0
        ? `${(a.times[i] / duration) * 230},${58 - ((v - min) / (max - min)) * 50}`
        : null,
    )
    .filter(Boolean)
    .join(" ");
  return (
    <svg
      className="trace"
      viewBox="0 0 230 64"
      role="img"
      aria-label={`Activity for ${id}; ${a.unit}`}
    >
      <path d="M0 16H230M0 37H230M0 58H230" stroke="#313a31" strokeWidth=".5" />
      <polyline
        points={points}
        fill="none"
        stroke="#bbd780"
        strokeWidth="1.5"
      />
      <line
        x1={(time / duration) * 230}
        x2={(time / duration) * 230}
        y1="0"
        y2="64"
        stroke="#e8dabb"
        strokeWidth="1"
      />
    </svg>
  );
}
