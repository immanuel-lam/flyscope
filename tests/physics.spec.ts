import { test, expect } from "@playwright/test";
import { existsSync } from "node:fs";
test("physical body replay shares neural time, rewinds, and clears on dataset switch", async ({
  page,
}) => {
  test.skip(
    !existsSync("data/physics/latest.json"),
    "Generate a physics run first",
  );
  test.setTimeout(60000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/?dataset=malecns");
  await page.getByRole("button", { name: "Load latest run" }).click();
  await expect(page.getByLabel("Motor controller")).toHaveValue("physics");
  await expect(page.getByTestId("physics-metrics")).toContainText(
    "mm displacement",
  );
  await expect(page.locator(".fly-viewport")).toContainText(
    "actual physical rig",
  );
  const initial = await page.getByTestId("motor-position").innerText();
  const canvas = page.locator(".fly-viewport canvas");
  await expect(canvas).toHaveAttribute("data-recorded-cells", "756");
  const initialStrength = await canvas.getAttribute("data-activity-strength");
  await page.getByLabel("Activity time", { exact: true }).fill("1");
  await expect(page.getByTestId("motor-position")).not.toHaveText(initial);
  await expect(canvas).not.toHaveAttribute(
    "data-activity-strength",
    initialStrength!,
  );
  await expect(page.locator(".neuron-detail")).toContainText("normalized rate");
  await page.getByRole("button", { name: "Rewind activity" }).click();
  await expect(page.getByTestId("motor-position")).toHaveText(initial);
  await page
    .getByRole("button", { name: "Play activity", exact: true })
    .click();
  await expect(
    page.getByLabel("Activity time", { exact: true }),
  ).not.toHaveValue("0");
  await page.getByLabel("Dataset", { exact: true }).selectOption("demo");
  await expect(page.getByLabel("Motor controller")).toHaveValue("off");
  await expect(
    page.getByRole("region", { name: "Physical fly simulation" }),
  ).toHaveCount(0);
  expect(errors).toEqual([]);
});
