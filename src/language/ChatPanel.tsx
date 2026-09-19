import { useEffect, useRef, useState } from "react";
import type { Activity } from "../data";
type Message = { role: "user" | "assistant"; content: string };
export interface ChatReply {
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
export default function ChatPanel({
  onActivity,
}: {
  onActivity: (reply: ChatReply) => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]),
    [input, setInput] = useState(""),
    [busy, setBusy] = useState(false),
    [ready, setReady] = useState(false),
    [error, setError] = useState(""),
    [ablate, setAblate] = useState(false),
    [last, setLast] = useState<ChatReply>();
  const mounted = useRef(true);
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
    };
  }, []);
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!input.trim() || busy) return;
    const text = input.trim();
    setInput("");
    setBusy(true);
    setError("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    try {
      const response = await fetch("/api/chat/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: messages.slice(-4),
          ablated: ablate,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      if (!mounted.current) return;
      setLast(result);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: result.text || "[Model produced an end token without text]",
        },
      ]);
      onActivity(result);
    } catch (e) {
      if (mounted.current) setError((e as Error).message);
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  return (
    <section className="chat-panel motor-panel" aria-label="FlyGPT chat">
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
            Start with “Hi”, then ask a short question. Each reply plays the
            cell states used to generate its tokens.
          </p>
        )}
        {busy && <p role="status">Running the neural circuit…</p>}
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
          {last.elapsedSeconds.toFixed(2)} s ·{" "}
          {last.ablated ? "connections disabled" : "real graph enabled"}. Glow
          shows hidden-state magnitude; these are continuous states, not spikes.
        </p>
      )}
      {error && (
        <p role="alert" className="motor-error">
          {error}
        </p>
      )}
    </section>
  );
}
