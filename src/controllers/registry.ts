import type { MotorController } from "../motor/types";
// Vite discovers checked-in controller modules. JSON uploads never execute code.
const modules = import.meta.glob<{ controller: MotorController }>(
  "./*/controller.ts",
  { eager: true },
);
export const controllers = Object.values(modules)
  .map((m) => m.controller)
  .sort((a, b) =>
    a.id === "manual" ? -1 : b.id === "manual" ? 1 : a.id.localeCompare(b.id),
  );
const ids = new Set<string>();
for (const c of controllers) {
  if (
    !c ||
    !c.id ||
    ids.has(c.id) ||
    typeof c.create !== "function" ||
    typeof c.unsupported !== "function"
  )
    throw new Error("Invalid or duplicate motor controller.");
  ids.add(c.id);
}
