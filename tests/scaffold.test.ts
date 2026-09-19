import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
test("agent scaffold produces a discoverable module and test, refuses overwrite and unsafe names", () => {
  const root = mkdtempSync(join(tmpdir(), "flyscope-scaffold-"));
  const script = resolve("scripts/new-controller.mjs");
  try {
    const run = (id: string) =>
      spawnSync(process.execPath, [script, id], {
        cwd: root,
        encoding: "utf8",
      });
    assert.equal(run("test-driver").status, 0);
    assert.match(
      readFileSync(
        join(root, "src/controllers/test-driver/controller.ts"),
        "utf8",
      ),
      /export const controller/,
    );
    assert.match(
      readFileSync(join(root, "tests/test-driver.test.ts"), "utf8"),
      /simulateMotor/,
    );
    assert.equal(run("test-driver").status, 1);
    assert.equal(run("../../unsafe").status, 1);
    assert.equal(run("replay").status, 1);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
