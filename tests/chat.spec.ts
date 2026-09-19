import { test, expect } from "@playwright/test";
import { existsSync } from "node:fs";
test("Explore is anatomy only; chat is in the dedicated FlyGPT workspace", async ({
  page,
}) => {
  test.skip(!existsSync("public/malecns/catalog.json"));
  await page.goto("/");
  await expect(page.getByLabel("Dataset", { exact: true })).toHaveValue("full");
  await expect(page.getByLabel("Find neurons")).toBeVisible();
  await expect(page.getByLabel("Message", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "FlyGPT", exact: true }).click();
  await expect(page.getByLabel("Message", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Find neurons")).toHaveCount(0);
  await expect(page.locator(".compact-chat .chat-messages")).toHaveCSS(
    "min-height",
    "220px",
  );
  await page.getByRole("button", { name: "Explore", exact: true }).click();
  await expect(page.getByLabel("Find neurons")).toBeVisible();
  await expect(page.getByLabel("Message", { exact: true })).toHaveCount(0);
});
