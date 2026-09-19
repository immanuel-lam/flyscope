import type { Activity, Dataset } from "../data.ts";
import type { MotorController, MotorCommand } from "./types.ts";
import { simulateMotor } from "./engine.ts";
/** Record controller-emitted values on the same clock as motor integration. */
export function simulateExperiment(
  dataset: Dataset,
  controller: MotorController,
  parameters: MotorCommand,
  duration: number,
) {
  const samples: { time: number; values: Record<string, number> }[] = [];
  const ids = new Set(dataset.neurons.map((n) => n.id));
  const wrapped: MotorController = {
    ...controller,
    create: (context) => {
      const instance = controller.create(context);
      return {
        step: (observation) => {
          const out = instance.step(observation);
          if (controller.activity) {
            if (!out.neural || !Object.keys(out.neural).length)
              throw new Error("Coupled controller omitted neural output.");
            for (const [id, v] of Object.entries(out.neural))
              if (!ids.has(id) || !Number.isFinite(v))
                throw new Error(
                  "Coupled controller returned invalid neuron activity.",
                );
            samples.push({ time: observation.time, values: { ...out.neural } });
          }
          return out;
        },
      };
    },
  };
  const motor = simulateMotor(dataset, wrapped, parameters, duration);
  let activity: Activity | undefined;
  if (controller.activity && samples.length) {
    const keys = Object.keys(samples[0].values);
    if (
      samples.some(
        (s) =>
          Object.keys(s.values).length !== keys.length ||
          keys.some((k) => s.values[k] === undefined),
      )
    )
      throw new Error("Coupled activity channel set changed during the run.");
    const last = samples.at(-1)!;
    samples.push({ time: duration, values: last.values });
    activity = {
      ...controller.activity,
      times: samples.map((s) => s.time),
      values: Object.fromEntries(
        keys.map((id) => [id, samples.map((s) => s.values[id])]),
      ),
    };
  }
  return { motor, activity };
}
