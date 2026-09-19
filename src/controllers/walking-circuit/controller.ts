import type { MotorController } from "../../motor/types.ts";
export const WALK_IDS = Array.from(
  { length: 24 },
  (_, i) => `demo-${5 + i * 6}`,
);
export const controller: MotorController = {
  id: "walking-circuit",
  label: "Walking neural circuit",
  description:
    "Synthetic oscillator population and motor readout; not a biological connectome model.",
  activity: { kind: "synthetic", unit: "normalized rate" },
  unsupported: (d) =>
    d.id !== "synthetic-workbench" ||
    !WALK_IDS.every((id) => d.neurons.some((n) => n.id === id))
      ? "Load the synthetic fly circuit. Real MaleCNS cells have no validated mapping for this toy controller."
      : null,
  create: ({ parameters }) => ({
    step: ({ time }) => {
      const drive = Math.min(1, Math.abs(parameters.forward) / 3);
      const neural: Record<string, number> = {};
      WALK_IDS.forEach((id, i) => {
        neural[id] =
          drive *
          (0.5 +
            0.5 *
              Math.sin(
                time * 2 * Math.PI * 3 +
                  (i % 2) * Math.PI +
                  Math.floor(i / 4) * Math.PI,
              ));
      });
      const mean =
        WALK_IDS.reduce((sum, id) => sum + neural[id], 0) / WALK_IDS.length;
      return {
        forward: Math.sign(parameters.forward) * mean * 6,
        turn: parameters.turn,
        wing: parameters.wing,
        neural,
      };
    },
  }),
};
