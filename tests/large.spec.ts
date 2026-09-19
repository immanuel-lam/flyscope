import { test, expect } from "@playwright/test";
import { existsSync } from "node:fs";
test("full catalog, selected source geometry, graph and stale-request isolation", async ({
  page,
}) => {
  test.skip(
    !existsSync("public/malecns/catalog.json"),
    "Run prepare:malecns first.",
  );
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/?dataset=malecns");
  await expect(page.locator(".large-status")).toContainText("166,700 cells", {
    timeout: 60000,
  });
  await expect(page.locator(".dataset-count")).toContainText("25,582,938");
  await page.getByRole("textbox", { name: "Find neurons" }).fill("12781");
  await page
    .locator(".neuron-list button")
    .filter({ hasText: "DNge104_R" })
    .click();
  await expect(page.locator(".neuron-detail")).toContainText(
    "15942 / 15942 segments",
    { timeout: 45000 },
  );
  await expect(page.locator(".large-status")).toContainText("1,426 outgoing");
  await page.getByLabel("Connection graph").check();
  await expect
    .poll(() =>
      page.locator(".brain-viewport canvas").getAttribute("data-lines"),
    )
    .not.toBe("16998");
  const frames = await page
    .locator(".brain-viewport canvas")
    .getAttribute("data-rendered-frames");
  await page.waitForTimeout(300);
  await expect(page.locator(".brain-viewport canvas")).toHaveAttribute(
    "data-rendered-frames",
    frames!,
  );
  await page.getByLabel("Dataset", { exact: true }).selectOption("demo");
  await expect(page.locator(".dataset-status")).toContainText(
    "Synthetic fly circuit",
  );
  await page.waitForTimeout(250);
  await expect(page.locator(".large-status")).toHaveCount(0);
  expect(errors).toEqual([]);
});
