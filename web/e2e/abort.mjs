// Прерывание реального запуска из браузера. Запуск: node e2e/abort.mjs <work> <run_id>
import { launch } from "./cdp.mjs";
const SP = process.argv[2]; (await import("node:fs")).mkdirSync(`${SP}/shots`, { recursive: true }); const B = "http://127.0.0.1:5173";
let failed = 0; const ok = (n, c, x = "") => { console.log(`${c ? "PASS" : "FAIL"} ${n} ${c ? "" : x}`); if (!c) failed++; };
const id = process.argv[3];
const b = await launch(9337);
await b.goto(B + "/runs/" + id, 1200);
ok("запуск выполняется", await b.ev(`document.querySelector('.run-head .badge')?.textContent.includes('Выполняется')`));
ok("текущий CSV уже доступен во время выполнения", await b.ev(`![...document.querySelectorAll('.actions button')].find(x=>x.textContent.includes('текущий')).disabled`));
await b.clickText("button", "Прервать запуск");
ok("диалог подтверждения", await b.ev(`!!document.querySelector('.dialog')`));
await b.clickText(".dialog button", "Прервать");
for (let i = 0; i < 20; i++) { if (await b.ev(`document.querySelector('.run-head .badge')?.textContent.includes('Остановлено')`)) break; await b.sleep(500); }
ok("статус «Остановлено» подтверждён сервером", await b.ev(`document.querySelector('.run-head .badge')?.textContent.includes('Остановлено')`));
ok("текущий CSV остаётся доступным, итоговый — нет", await b.ev(`(()=>{const bs=[...document.querySelectorAll('.actions button')];const g=t=>bs.find(x=>x.textContent.includes(t));return !g('текущий').disabled&&g('итоговый').disabled})()`));
const dl = await b.ev(`fetch('/api/runs/${id}/download/current').then(r=>r.status+':'+r.headers.get('content-type'))`);
ok("текущий CSV скачивается", dl.startsWith("200:text/csv"), dl);
await b.shot(`${SP}/shots/api_stopped.png`);
console.log(failed ? `FAILED: ${failed}` : "ALL PASSED"); await b.close(); process.exit(failed ? 1 : 0);
