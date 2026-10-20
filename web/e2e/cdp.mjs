// Минимальный клиент Chrome DevTools Protocol для проверок интерфейса без сторонних пакетов.
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export async function launch(port = 9333, size = [1440, 900]) {
  const dir = mkdtempSync(join(tmpdir(), "cdp-"));
  const proc = spawn(process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`, `--window-size=${size[0]},${size[1]}`,
     "--no-first-run", "--disable-gpu", "about:blank"], { stdio: "ignore" });
  let targets;
  for (let i = 0; i < 50; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (targets.some(t => t.type === "page")) break; } catch {}
    await new Promise(r => setTimeout(r, 200));
  }
  const ws = new WebSocket(targets.find(t => t.type === "page").webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener("open", r));
  let id = 0; const pending = new Map(); const listeners = [];
  ws.addEventListener("message", m => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    else listeners.forEach(l => l(d));
  });
  const send = (method, params = {}) => new Promise((res, rej) => {
    const i = ++id; pending.set(i, d => d.error ? rej(new Error(method + ": " + JSON.stringify(d.error))) : res(d.result));
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const api = {
    send, sleep, on: l => listeners.push(l),
    async ev(expr) {
      const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) { console.error("ev error:", (r.exceptionDetails.exception?.description || "").split("\n")[0], "::", expr.slice(0, 90)); return null; }
      return r.result.value;
    },
    async goto(url, wait = 800) { await send("Page.navigate", { url }); await sleep(wait); },
    async size(w, h) { await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile: false }); await sleep(200); },
    async press(key, code = key, vk = 0) {
      const p = { key, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk };
      await send("Input.dispatchKeyEvent", { type: "rawKeyDown", ...p });
      if (key === "Enter") await send("Input.dispatchKeyEvent", { type: "char", text: "\r", ...p });
      if (key === " ") await send("Input.dispatchKeyEvent", { type: "char", text: " ", ...p });
      await send("Input.dispatchKeyEvent", { type: "keyUp", ...p });
      await sleep(80);
    },
    tab: (shift) => send("Input.dispatchKeyEvent", { type: "rawKeyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9, modifiers: shift ? 8 : 0 })
      .then(() => send("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 })).then(() => sleep(60)),
    enter: () => api.press("Enter", "Enter", 13),
    esc: () => api.press("Escape", "Escape", 27),
    space: () => api.press(" ", "Space", 32),
    /** Центр элемента по JS-выражению, возвращающему элемент */
    async clickBy(finder) {
      const pt = await api.ev(`(()=>{const e=(${finder});if(!e)return null;e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);
      if (!pt) throw new Error("не найден элемент: " + finder);
      await send("Input.dispatchMouseEvent", { type: "mousePressed", x: pt.x, y: pt.y, button: "left", clickCount: 1 });
      await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: pt.x, y: pt.y, button: "left", clickCount: 1 });
      await sleep(250);
    },
    clickText(sel, text) { return api.clickBy(`[...document.querySelectorAll(${JSON.stringify(sel)})].find(e=>e.textContent.includes(${JSON.stringify(text)}))`); },
    async shot(path) { const r = await send("Page.captureScreenshot", { format: "png" }); (await import("node:fs")).writeFileSync(path, Buffer.from(r.data, "base64")); },
    async close() { try { await send("Browser.close"); } catch {} proc.kill(); },
  };
  await send("Page.enable"); await send("Runtime.enable"); await send("DOM.enable"); await send("Network.enable");
  await api.size(size[0], size[1]);
  return api;
}
