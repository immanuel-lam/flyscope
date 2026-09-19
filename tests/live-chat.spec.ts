import { test, expect } from "@playwright/test";
import { existsSync } from "node:fs";
test("FlyGPT streams actual neural states while generating and clears glow at completion", async ({
  page,
}) => {
  test.skip(!existsSync("public/malecns/catalog.json"));
  await page.goto("/");
  const button = page.getByRole("button", { name: "FlyGPT", exact: true });
  await expect(button).toBeEnabled();
  await button.click();
  await page.getByLabel("Message", { exact: true }).fill("Describe a forest.");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  const canvas = page.locator(".fly-anatomy canvas");
  await expect(canvas).toHaveAttribute("data-activity-visible", "true");
  await expect(canvas).toHaveAttribute("data-points", "5512");
  await expect
    .poll(async () =>
      Number(await canvas.getAttribute("data-activity-strength")),
    )
    .toBeGreaterThan(0);
  await expect(page.getByLabel("Live neural circuit")).toContainText(
    "update 1",
  );
  await expect(page.locator(".chat-message.assistant p")).not.toBeEmpty();
  await expect(page.getByTestId("chat-provenance")).toContainText("tokens/s", {
    timeout: 20000,
  });
  await expect(canvas).toHaveAttribute("data-activity-visible", "false");
  await expect(page.getByLabel("Live neural circuit")).toContainText(
    "last computed pathways",
  );
});
