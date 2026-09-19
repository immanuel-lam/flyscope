import type { Activity, Neuron } from "../data.ts";
/** Only relative display gain; never changes the recorded values or invents events. */
export function recordedPeaks(activity?: Activity): Map<string, number> {
  return new Map(
    Object.entries(activity?.values ?? {}).map(([id, values]) => [
      id,
      values.reduce((peak, v) => Math.max(peak, Math.abs(v)), 0),
    ]),
  );
}
export function relativeRate(value: number | undefined, peak: number): number {
  return value === undefined || peak <= 0
    ? 0
    : Math.min(1, Math.abs(value) / peak);
}
export function activityPointOrder(
  neurons: Neuron[],
  recorded: Map<string, number>,
  budget: number,
): number[] {
  const chosen = new Set<number>();
  neurons.forEach((n, i) => {
    if (recorded.has(n.id)) chosen.add(i);
  });
  for (let i = 0; i < Math.min(budget, neurons.length); i++)
    chosen.add(
      Math.floor((i * neurons.length) / Math.min(budget, neurons.length)),
    );
  const order = [...chosen];
  for (let i = 0; i < neurons.length; i++) if (!chosen.has(i)) order.push(i);
  return order;
}
