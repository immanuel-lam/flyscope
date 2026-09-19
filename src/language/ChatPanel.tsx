import { useEffect, useRef, useState } from "react";
import CircuitView, { type CircuitInspection } from "./CircuitView";
import type { Activity } from "../data";
type Message = { role: "user" | "assistant"; content: string };
export interface ChatReply {
  inspection?: CircuitInspection;
  text: string;
  tokens: string[];
  activity: Activity;
  modelId: string;
  neurons: number;
  edges: number;
  elapsedSeconds: number;
  datasetId: string;
  datasetVersion: string;
  ablated: boolean;
}
export type LiveEvent = {
  type: string;
  inspection?: CircuitInspection;
  ids?: string[];
  datasetId?: string;
  datasetVersion?: string;
  step?: CircuitInspection["steps"][number];
  values?: number[];
  sequence?: number;
  text?: string;
  tokenCount?: number;
  elapsedSeconds?: number;
  error?: string;
  result?: ChatReply;
};
export default function ChatPanel({
  onLive,
  compact = false,
}: {
  onLive?: (event: LiveEvent) => void;
  compact?: boolean;
}) {
  const [messages, setMessages] = useState<Message[]>([]),
    [input, setInput] = useState(""),
    [busy, setBusy] = useState(false),
    [ready, setReady] = useState(false),
    [error, setError] = useState(""),
    [ablate, setAblate] = useState(false),
    [last, setLast] = useState<ChatReply>();
  const mounted = useRef(true);
  const abort = useRef<AbortController | null>(null);
  const [visibleSteps, setVisibleSteps] = useState(true);
  const [rate, setRate] = useState(0);
  useEffect(() => {
    mounted.current = true;
    fetch("/api/chat/status")
      .then((r) => r.json())
      .then((s) => {
        if (mounted.current) setReady(s.ready);
      })
      .catch(() => {});
    return () => {
      mounted.current = false;
      abort.current?.abort();
    };
  }, []);
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!input.trim() || busy) return;
    const text = input.trim();
    setInput("");
    setBusy(true);
    setRate(0);
    onLive?.({ type: "pending" });
    abort.current = new AbortController();
    setError("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    try {
      const response = await fetch("/api/chat/generate", {
        method: "POST",
        signal: abort.current.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: messages.slice(-4),
          ablated: ablate,
          stream: true,
          visibleSteps,
        }),
      });
      if (!response.ok) throw new Error((await response.json()).error);
      if (!response.body) throw new Error("Missing inference stream");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let result: ChatReply | undefined;
      setMessages((m) => [...m, { role: "assistant", content: "" }]);
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let newline: number;
        while ((newline = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          if (!line.trim()) continue;
          const event = JSON.parse(line) as LiveEvent;
          if (event.type === "error") throw new Error(event.error);
          if (!mounted.current) return;
          if (event.type === "token") {
            setMessages((m) => [
              ...m.slice(0, -1),
              { role: "assistant", content: event.text ?? "" },
            ]);
            setRate(
              (event.tokenCount ?? 0) /
                Math.max(event.elapsedSeconds ?? 0, 0.001),
            );
          }
          if (event.type === "result") result = event.result;
          onLive?.(event);
        }
        if (done) break;
      }
      if (!result) throw new Error("Inference stream ended before completion");
      setLast(result);
      setRate(result.tokens.length / Math.max(result.elapsedSeconds, 0.001));
      setMessages((m) => [
        ...m.slice(0, -1),
        {
          role: "assistant",
          content: result!.text || "[Model produced an end token without text]",
        },
      ]);
    } catch (e) {
      if (mounted.current) setError((e as Error).message);
    } finally {
      onLive?.({ type: "end" });
      if (mounted.current) setBusy(false);
    }
  }
  return (
    <section
      className={`chat-panel motor-panel ${compact ? "compact-chat" : ""}`}
      aria-label="FlyGPT chat"
    >
      <div className="motor-heading">
        <div>
          <span className="eyebrow">MALECNS · EXPERIMENTAL CHAT</span>
          <h3>FlyGPT</h3>
        </div>
        <button
          onClick={() => {
            setMessages([]);
            setLast(undefined);
          }}
          disabled={busy}
        >
          New conversation
        </button>
      </div>
      <p className="motor-provenance">
        512 source neuron IDs · 28,452 real directed connections. Trained from
        scratch; no external language model answers for it. Experimental,
        limited text generation—not a general assistant.
      </p>
      <div className="chat-messages" role="log" aria-live="polite">
        {messages.length ? (
          messages.map((m, i) => (
            <div className={`chat-message ${m.role}`} key={i}>
              <strong>{m.role === "user" ? "You" : "FlyGPT"}</strong>
              <p>{m.content}</p>
            </div>
          ))
        ) : (
          <p>
            Send any short message. The model generates tokens through its
            trained MaleCNS connections. Replies can be incorrect or unreadable.
          </p>
        )}
        {busy && (
          <p role="status">
            Thinking… {rate.toFixed(1)} tokens/s{" "}
            <button onClick={() => abort.current?.abort()}>Stop</button>
          </p>
        )}
      </div>
      <form onSubmit={submit}>
        <label htmlFor="fly-chat">Message</label>
        <div className="chat-input">
          <input
            id="fly-chat"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            maxLength={1000}
            placeholder="Message FlyGPT…"
            disabled={busy}
          />
          <button disabled={busy || !ready || !input.trim()} type="submit">
            Send
          </button>
        </div>
      </form>
      <label className="physics-options">
        <input
          type="checkbox"
          checked={visibleSteps}
          disabled={busy}
          onChange={(e) => setVisibleSteps(e.target.checked)}
        />{" "}
        Visible computation steps (20 steps/s limit)
      </label>
      <label className="physics-options">
        <input
          type="checkbox"
          checked={ablate}
          onChange={(e) => setAblate(e.target.checked)}
          disabled={busy}
        />{" "}
        Disable connections for this reply (ablation)
      </label>
      {!ready && (
        <p className="motor-provenance">
          Checkpoint not available yet. Training must finish before chat is
          enabled.
        </p>
      )}
      {last && (
        <p className="motor-provenance" data-testid="chat-provenance">
          {last.modelId} · {last.tokens.length} generated tokens ·{" "}
          {rate.toFixed(1)} tokens/s · {last.elapsedSeconds.toFixed(2)} s ·{" "}
          {last.ablated ? "connections disabled" : "real graph enabled"}.
          Artificial continuous cell states; no biological reasoning claim.
        </p>
      )}
      {!compact && last?.inspection && <CircuitView data={last.inspection} />}
      {error && (
        <p role="alert" className="motor-error">
          {error}
        </p>
      )}
    </section>
  );
}
