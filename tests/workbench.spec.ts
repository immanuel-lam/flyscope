import { test, expect } from "@playwright/test";
test("viewer, playback, filtering, source reconstruction, import and responsive layout", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/?dataset=demo");
  await expect(page.locator("canvas")).toHaveCount(2);
  await page.screenshot({ path: "tests/desktop.png", fullPage: true });
  await page
    .getByRole("button", { name: "Play activity", exact: true })
    .click();
  await page.waitForTimeout(450);
  await page
    .getByRole("button", { name: "Pause activity", exact: true })
    .click();
  expect(
    +(await page.getByRole("slider", { name: "Activity time" }).inputValue()),
  ).toBeGreaterThan(0);
  await page
    .getByRole("button", { name: "Rewind activity", exact: true })
    .click();
  await expect(page.getByRole("slider", { name: "Activity time" })).toHaveValue(
    "0",
  );
  await page.getByRole("textbox", { name: "Find neurons" }).fill("OL-L 002");
  await expect(page.locator(".neuron-list button")).toHaveCount(1);
  await page.locator(".neuron-list button").click();
  await expect(page.locator(".neuron-detail h2")).toHaveText("OL-L 002");
  await page
    .getByRole("textbox", { name: "Find neurons" })
    .fill("nothing matches");
  await expect(page.getByText("No neurons match this search.")).toBeVisible();
  await page.getByRole("button", { name: "Clear search" }).click();
  await page.getByLabel("Dataset", { exact: true }).selectOption("reference");
  await expect(page.locator(".dataset-status strong")).toHaveText(
    "MaleCNS · DNge104 pair",
  );
  await expect(
    page.getByRole("button", { name: "Play activity", exact: true }),
  ).toBeDisabled();
  await expect(page.locator(".neuron-list button")).toHaveCount(2);
  await expect(page.locator(".neuron-detail h2")).toHaveText("DNge104_R");
  await page.screenshot({ path: "tests/reference.png", fullPage: true });
  await page.locator("input[type=file]").setInputFiles({
    name: "bad.json",
    mimeType: "application/json",
    buffer: Buffer.from("{}"),
  });
  await expect(page.getByRole("alert")).toContainText("schemaVersion");
  await page.getByRole("button", { name: "Dismiss error" }).click();
  const fixture = {
    schemaVersion: 1,
    id: "custom",
    name: "Test recording",
    version: "1",
    source: "test fixture",
    geometry: "synthetic",
    coordinateSpace: "test",
    units: "um",
    neurons: [
      {
        id: "9007199254740999",
        label: "Test neuron",
        region: "Test",
        position: [0, 0, 0],
      },
    ],
    edges: [],
    activity: {
      kind: "recording",
      unit: "mV",
      times: [0, 1, 2],
      values: { "9007199254740999": [-70, -50, -65] },
    },
  };
  await page.locator("input[type=file]").setInputFiles({
    name: "recording.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(fixture)),
  });
  await expect(page.locator(".dataset-status strong")).toHaveText(
    "Test recording",
  );
  await expect(page.locator(".neuron-detail code")).toHaveText(
    "9007199254740999",
  );
  await expect(
    page.getByRole("button", { name: "Play activity", exact: true }),
  ).toBeEnabled();
  await page.getByLabel("Dataset", { exact: true }).selectOption("demo");
  await page.getByRole("button", { name: "Data & research" }).click();
  await expect(
    page.getByRole("heading", { name: "Wiring is the starting point." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Open the workbench" }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "tests/mobile.png", fullPage: true });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});
