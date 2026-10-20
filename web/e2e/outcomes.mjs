// Исходы реальных запусков: отказ API внутри попытки, Optuna и выбранный финалист.
// Запуск: node e2e/outcomes.mjs <work> <run_id отказа API> <run_id Optuna>
import { launch } from "./cdp.mjs";
const [SP, failId, optId] = process.argv.slice(2); (await import("node:fs")).mkdirSync(`${SP}/shots`, { recursive: true });
let failed = 0; const ok = (n, c, x = "") => { console.log(`${c ? "PASS" : "FAIL"} ${n} ${c ? "" : x}`); if (!c) failed++; };
const b = await launch(9341);
await b.goto("http://127.0.0.1:5173/runs/" + failId, 1500);
const t = await b.ev(`document.body.textContent`);
ok("отказ API виден в ленте внутри попытки", /запрос Qwen/.test(t) && /APIConnectionError/.test(t));
ok("примечание: api_error и baseline в CSV", /api_error/.test(t) && /вошёл baseline/.test(t));
ok("неопределённость usage показана", /Usage части запросов неизвестен/.test(t));
ok("итоговый и текущий CSV доступны", await b.ev(`[...document.querySelectorAll('.actions button')].filter(x=>/CSV/.test(x.textContent)).every(x=>!x.disabled)`));
await b.clickBy(`[...document.querySelectorAll('.feed-item')].find(e=>e.textContent.includes('Запрос к Qwen'))`);
ok("панель попытки показывает ошибку", await b.ev(`document.querySelector('.drawer').textContent.includes('APIConnectionError')`));
await b.shot(`${SP}/shots/api_fail_drawer.png`); await b.esc();
await b.shot(`${SP}/shots/api_fail.png`);
await b.goto("http://127.0.0.1:5173/runs/" + optId, 1500);
const o = await b.ev(`({best:document.querySelector('.stat-accent .stat-value')?.textContent, note:document.querySelector('.alert')?.textContent, rows:document.querySelectorAll('.feed-item').length})`);
ok("Optuna: лучший = выбранный финалист 0.9111", o.best === "0.9111", JSON.stringify(o));
ok("Optuna: примечание об ансамбле", /ensemble:baseline\+numeric-00019/.test(o.note || ""), o.note);
await b.clickBy(`[...document.querySelectorAll('.feed-item')].find(e=>e.textContent.includes('numeric-00019'))`);
ok("Optuna: панель показывает найденные параметры", await b.ev(`/num_leaves/.test(document.querySelector('.drawer').textContent)`));
await b.shot(`${SP}/shots/api_optuna_drawer.png`);
console.log(failed ? `FAILED: ${failed}` : "ALL PASSED"); await b.close(); process.exit(failed ? 1 : 0);
