import { test, expect } from "@playwright/test";
async function seek(page: import("@playwright/test").Page, time: number) {
  await page
    .getByRole("slider", { name: "Activity time" })
    .evaluate((element, time) => {
      const setter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!;
      setter.call(element, String(time));
      element.dispatchEvent(new Event("input", { bubbles: true }));
    }, time);
}
test("motor rig moves, rewind is exact, run export replays, and real neurons require explicit control", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/?dataset=demo");
  await page.getByLabel("Motor controller").selectOption("manual");
  const rest = await page.getByTestId("motor-position").textContent();
  await seek(page, 1);
  await expect(page.getByTestId("motor-position")).not.toHaveText(rest!);
  const atOne = await page.getByTestId("motor-position").textContent();
  await seek(page, 2);
  await seek(page, 1);
  await expect(page.getByTestId("motor-position")).toHaveText(atOne!);
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export motor run" }).click();
  const file = await downloadEvent;
  const filePath = testInfo.outputPath("motor-run.json");
  await file.saveAs(filePath);
  await page.locator("input[type=file]").setInputFiles(filePath);
  await expect(page.getByLabel("Motor controller")).toHaveValue("replay");
  await seek(page, 1);
  await expect(page.getByTestId("motor-position")).toHaveText(atOne!);
  await page.getByLabel("Dataset", { exact: true }).selectOption("reference");
  await expect(page.getByLabel("Motor controller")).toHaveValue("off");
  await expect(page.locator('option[value="neural-readout"]')).toBeDisabled();
  await page.getByLabel("Motor controller").selectOption("manual");
  await expect(
    page.getByRole("button", { name: "Play activity", exact: true }),
  ).toBeEnabled();
  await expect(page.locator("#threshold")).toBeDisabled();
  const before = await page.locator(".fly-viewport canvas").screenshot();
  await seek(page, 1.23);
  const after = await page.locator(".fly-viewport canvas").screenshot();
  expect(before.equals(after)).toBeFalsy();
  await page.getByRole("button", { name: "Rewind activity" }).click();
  await expect(page.getByTestId("motor-position")).toHaveText(
    "x 0.00 · z 0.00 mm",
  );
  await page.getByLabel("Dataset", { exact: true }).selectOption("demo");
  await page.getByLabel("Motor controller").selectOption("neural-readout");
  await seek(page, 3);
  await expect(page.getByTestId("motor-position")).not.toHaveText(
    "x 0.00 · z 0.00 mm",
  );
  await page.screenshot({ path: "tests/motor-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "tests/motor-mobile.png", fullPage: true });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});

test("coupled walking shows the controller neuron rates", async ({ page }) => {
  await page.goto("/?dataset=demo");
  await page.getByLabel("Motor controller").selectOption("walking-circuit");
  await expect(page.locator(".neuron-detail code")).toHaveText("demo-5");
  await seek(page, 0.1);
  await expect(page.locator(".signal-title")).toContainText("normalized rate");
  await expect(page.getByTestId("motor-position")).not.toHaveText(
    "x 0.00 · z 0.00 mm",
  );
});
