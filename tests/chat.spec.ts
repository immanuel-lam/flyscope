import { test, expect } from "@playwright/test";
import { existsSync } from "node:fs";
test("MaleCNS opens by default with working FlyGPT side chat without state replay", async ({
  page,
}) => {
  test.skip(
    !existsSync("models/malecns-chat/runtime.npz") ||
      !existsSync("public/malecns/catalog.json"),
    "Prepare data and train the chat model first",
  );
  test.setTimeout(60000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", {
      name: "Explore the fruit fly nervous system.",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByLabel("Dataset", { exact: true })).toHaveValue("full");
  await expect(
    page.getByRole("tab", { name: "FlyGPT", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await page.getByLabel("Message", { exact: true }).fill("Hi");
  await page.route("**/api/chat/generate", async (route) => {
    const response = await route.fetch();
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.fulfill({ response });
  });
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Thinking…" }),
  ).toBeVisible();
  await expect(page.locator(".chat-message.assistant")).toBeVisible({
    timeout: 20000,
  });
  await expect(
    page.getByRole("status").filter({ hasText: "Thinking…" }),
  ).toHaveCount(0);
  await expect(page.getByTestId("chat-provenance")).toContainText(
    "real graph enabled",
  );
  await expect(page.locator(".brain-viewport canvas")).toHaveAttribute(
    "data-activity-visible",
    "false",
  );
  await page.getByRole("button", { name: "Inspect model computation" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("img", {
      name: "Computed recurrent cell states and next-token probabilities",
    }),
  ).toBeVisible();
  await page.getByLabel("Computation step").selectOption("1");
  await expect(page.getByRole("dialog")).toContainText("24 of 512 cells shown");
  await page.getByRole("button", { name: "Close circuit view" }).click();
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page
    .getByRole("tab", { name: "Neuron inspector", exact: true })
    .click();
  await expect(page.getByLabel("Find neurons")).toBeVisible();
  await page.getByRole("tab", { name: "FlyGPT", exact: true }).click();
  await expect(page.locator(".chat-message.assistant")).toBeVisible();
  expect(errors).toEqual([]);
});
