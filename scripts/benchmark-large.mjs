import { chromium } from "@playwright/test";
import { writeFileSync, mkdirSync } from "node:fs";
const browser = await chromium.launch({
  headless: true,
  args: [
    "--enable-webgl",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.goto("http://127.0.0.1:5174/");
await page.evaluate(() => {
  window.__long = [];
  new PerformanceObserver((l) =>
    window.__long.push(...l.getEntries().map((e) => e.duration)),
  ).observe({ type: "longtask", buffered: true });
});
const start = performance.now();
await page.getByLabel("Dataset", { exact: true }).selectOption("full");
await page.locator(".large-status").waitFor({ timeout: 60000 });
const loadMs = performance.now() - start;
await page.waitForTimeout(2000);
await page.getByRole("textbox", { name: "Find neurons" }).fill("12781");
await page.locator(".neuron-list button").first().click();
await page.waitForFunction(
  () =>
    document
      .querySelector(".neuron-detail")
      ?.textContent.includes("segments ·"),
  { timeout: 30000 },
);
const sample = await page.evaluate(async () => {
  const frame = [];
  let last = performance.now();
  await new Promise((resolve) => {
    function tick(now) {
      frame.push(now - last);
      last = now;
      if (frame.length < 180) requestAnimationFrame(tick);
      else resolve();
    }
    requestAnimationFrame(tick);
  });
  frame.sort((a, b) => a - b);
  return {
    rafMedianMs: frame[90],
    rafP95Ms: frame[171],
    longTasks: window.__long,
    heap: performance.memory
      ? {
          used: performance.memory.usedJSHeapSize,
          total: performance.memory.totalJSHeapSize,
        }
      : null,
    canvases: [...document.querySelectorAll("canvas")].map((c) => ({
      ...c.dataset,
    })),
    status: document.querySelector(".large-status")?.textContent,
    detail: document.querySelector(".neuron-detail")?.textContent,
  };
});
await page.screenshot({ path: "tests/full-model.png", fullPage: true });
const bounds = await page.locator(".brain-viewport canvas").boundingBox();
await page.evaluate(() => {
  window.__orbit = [];
  window.__orbitRun = true;
  let prev = performance.now();
  function frame(t) {
    window.__orbit.push(t - prev);
    prev = t;
    if (window.__orbitRun) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
});
await page.mouse.move(
  bounds.x + bounds.width * 0.5,
  bounds.y + bounds.height * 0.5,
);
await page.mouse.down();
await page.mouse.move(
  bounds.x + bounds.width * 0.8,
  bounds.y + bounds.height * 0.55,
  { steps: 60 },
);
await page.mouse.up();
const orbit = await page.evaluate(() => {
  window.__orbitRun = false;
  const a = window.__orbit.sort((a, b) => a - b);
  return {
    samples: a.length,
    medianMs: a[Math.floor(a.length * 0.5)],
    p95Ms: a[Math.floor(a.length * 0.95)],
    renderedPoints: document.querySelector(".brain-viewport canvas")?.dataset
      .points,
  };
});

const searchStart = performance.now();
await page.getByRole("textbox", { name: "Find neurons" }).fill("DNge104");
await page.locator(".neuron-list button").first().waitFor();
const searchMs = performance.now() - searchStart;
const result = {
  timestamp: new Date().toISOString(),
  environment:
    "Headless Chromium with SwiftShader software rendering, 1440x1000; local Vite dev server. RAF timings are scheduling, not a physical-GPU FPS claim.",
  loadMs,
  searchMs,
  orbit,
  ...sample,
  errors,
};
mkdirSync("docs/performance", { recursive: true });
writeFileSync("docs/performance/latest.json", JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
await browser.close();
