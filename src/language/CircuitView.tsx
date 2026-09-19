import { useEffect, useRef, useState } from "react";
export interface CircuitInspection {
  cells: { id: string; input: boolean; retention: number }[];
  edges: { from: number; to: number; weight: number }[];
  steps: {
    token: string;
    tokenId: number;
    states: number[][];
    input: number[];
    predictions: { token: string; probability: number; weights: number[] }[];
  }[];
  promptSteps: number;
  selection: string;
}
export default function CircuitView({
  data,
  embedded = false,
}: {
  data: CircuitInspection;
  embedded?: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [index, setIndex] = useState(0);
  useEffect(() => setIndex(0), [data]);
  const step = data.steps[Math.min(index, data.steps.length - 1)];
  if (!step) return null;
  const color = (v: number) => (v >= 0 ? "#86d8cd" : "#ef9aa9");
  const y = (i: number) => 80 + i * 22;
  const content = (
    <>
      <header>
        <div>
          <span className="eyebrow">FLYGPT · COMPUTED STATES</span>
          <h2 id="circuit-title">Inside the neural circuit</h2>
        </div>
        {!embedded && (
          <button onClick={() => dialog.current?.close()}>
            Close circuit view
          </button>
        )}
      </header>
      <p>
        Three columns show the same cells before and after two recurrent
        updates. These are continuous activations, not spikes. This is a
        computed snapshot, not a reasoning trace or an animation.
      </p>
      {!embedded && (
        <label>
          Computation step{" "}
          <select
            aria-label="Computation step"
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
          >
            {data.steps.map((s, i) => (
              <option key={i} value={i}>
                {i + 1} · {i < data.promptSteps ? "prompt" : "generated"} ·{" "}
                {JSON.stringify(s.token)}
              </option>
            ))}
          </select>
        </label>
      )}
      <p>
        Input token <code>{JSON.stringify(step.token)}</code> · token id{" "}
        {step.tokenId}. Embeddings enter the marked input cells at both updates.
        State retention and biases also contribute.
      </p>
      <div className="circuit-scroll">
        <svg
          viewBox="0 0 1040 645"
          role="img"
          aria-label="Computed recurrent cell states and next-token probabilities"
        >
          {[130, 410, 690].map((x, k) => (
            <text key={x} x={x} y={30} textAnchor="middle" fill="currentColor">
              {["previous state", "update 1", "update 2"][k]}
            </text>
          ))}
          <text x={930} y={30} textAnchor="middle" fill="currentColor">
            next-token prediction
          </text>
          {[0, 1].flatMap((pass) =>
            data.edges.map((edge, i) => {
              const contribution = edge.weight * step.states[pass][edge.from];
              return (
                <line
                  key={`${pass}-${i}`}
                  x1={130 + 280 * pass}
                  y1={y(edge.from)}
                  x2={410 + 280 * pass}
                  y2={y(edge.to)}
                  stroke={color(contribution)}
                  strokeOpacity={Math.min(0.7, Math.abs(contribution) * 2)}
                  strokeWidth={1}
                >
                  <title>
                    {data.cells[edge.from].id} → {data.cells[edge.to].id};
                    weight {edge.weight.toFixed(5)}; contribution{" "}
                    {contribution.toFixed(5)}
                  </title>
                </line>
              );
            }),
          )}
          {step.predictions.flatMap((p, j) =>
            p.weights.map((weight, i) => {
              const v = weight * step.states[2][i];
              return (
                weight !== 0 && (
                  <line
                    key={`${j}-${i}`}
                    x1={690}
                    y1={y(i)}
                    x2={915}
                    y2={140 + j * 90}
                    stroke={color(v)}
                    strokeOpacity={Math.min(0.65, Math.abs(v))}
                  >
                    <title>Learned readout contribution {v.toFixed(5)}</title>
                  </line>
                )
              );
            }),
          )}
          {step.states.flatMap((values, k) =>
            values.map((v, i) => (
              <g key={`${k}-${i}`}>
                <circle
                  cx={130 + 280 * k}
                  cy={y(i)}
                  r={7}
                  fill={color(v)}
                  fillOpacity={Math.abs(v)}
                  stroke={data.cells[i].input ? "#e5d58c" : "#9aaba6"}
                >
                  <title>
                    Cell {data.cells[i].id}; activation {v.toFixed(6)}; injected
                    embedding {step.input[i].toFixed(6)}; retention{" "}
                    {data.cells[i].retention.toFixed(4)}
                  </title>
                </circle>
                {k === 0 && (
                  <text
                    x={112}
                    y={y(i) + 3}
                    textAnchor="end"
                    fontSize={9}
                    fill="currentColor"
                  >
                    {data.cells[i].id}
                  </text>
                )}
              </g>
            )),
          )}
          {step.predictions.map((p, i) => (
            <g key={i}>
              <circle
                cx={915}
                cy={140 + i * 90}
                r={10}
                fill="#e5d58c"
                fillOpacity={p.probability}
                stroke="#e5d58c"
              />
              <text x={937} y={135 + i * 90} fill="currentColor" fontSize={12}>
                {JSON.stringify(p.token)}
              </text>
              <text x={937} y={153 + i * 90} fill="currentColor" fontSize={11}>
                {(p.probability * 100).toFixed(2)}%
              </text>
            </g>
          ))}
        </svg>
      </div>
      <p>
        24 of 512 cells shown · {data.edges.length} source connections between
        these cells. Teal = positive, pink = negative; brightness = magnitude.
        Yellow outlines mark input cells. Output lines are learned readout
        weights, not biological synapses. Probabilities cover the full
        vocabulary before generation masks.
      </p>
      <p>
        {data.selection} Hidden cells and their connections still contribute to
        every update. Hover a node or line for its value. Column order is
        computational, not anatomical.
      </p>
    </>
  );
  if (embedded)
    return <section className="circuit-embedded">{content}</section>;
  return (
    <>
      <button onClick={() => dialog.current?.showModal()}>
        Inspect model computation
      </button>
      <dialog
        ref={dialog}
        className="circuit-dialog"
        aria-labelledby="circuit-title"
      >
        {content}
      </dialog>
    </>
  );
}
