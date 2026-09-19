import type { MotorController } from "../../motor/types.ts";
/** These six mappings are an engineering fixture, NOT biological motor identities. */
export const LEFT_IDS = ["demo-5", "demo-17", "demo-29"];
export const RIGHT_IDS = ["demo-11", "demo-23", "demo-35"];
export const controller: MotorController = {
  id: "neural-readout",
  label: "Demo neural readout",
  description: "Arbitrary synthetic-neuron mapping; not a biological decoder.",
  unsupported: (d) =>
    d.id !== "synthetic-workbench" ||
    d.version !== "1.0" ||
    d.geometry !== "synthetic" ||
    !d.activity ||
    ![...LEFT_IDS, ...RIGHT_IDS].every((id) => d.activity?.values[id])
      ? "Requires the synthetic fixture and its six mapped activity channels. No automatic mapping of real neurons."
      : null,
  create: () => ({
    step: ({ activity }) => {
      const mean = (ids: string[]) =>
        ids.reduce(
          (s, id) => s + Math.max(0, Math.min(1, activity(id) ?? 0)),
          0,
        ) / ids.length;
      const l = mean(LEFT_IDS),
        r = mean(RIGHT_IDS);
      return { forward: (l + r) * 2.5, turn: (r - l) * 2, wing: 0 };
    },
  }),
};
