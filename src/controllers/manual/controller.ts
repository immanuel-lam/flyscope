import type { MotorController } from "../../motor/types.ts";
export const controller: MotorController = {
  id: "manual",
  label: "Manual walking",
  description: "User commands drive a synthetic alternating-tripod gait.",
  unsupported: () => null,
  create: ({ parameters }) => ({ step: () => ({ ...parameters }) }),
};
