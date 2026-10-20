// Подключённая версия: запуск из браузера через solution.py, опрос, перезагрузка, скачивание.
// Нужны: `make web-ui`, web_api с WEB_API_RUNS=<work>/runs-web и OPENAI_BASE_URL на web_api/tests/fake_llm.py.
// Запуск: node e2e/api.mjs <work>
import { launch } from "./cdp.mjs";
const SP = process.argv[2];
// Данные задачи: по умолчанию синтетика из make-data.mjs; E2E_DATA/E2E_TASK — реальная задача из data/
const DATA = (await import("node:path")).resolve(process.env.E2E_DATA ?? `${SP}/syn`); // DevTools требует абсолютные пути
const DESCRIPTION = (await import("node:fs")).readFileSync(process.env.E2E_TASK ?? `${SP}/syn/task.txt`, "utf8");
const TITLE = process.env.E2E_TITLE ?? "Из браузера, fake Qwen";
const MINUTES = Number(process.env.E2E_MINUTES ?? 10); (await import("node:fs")).mkdirSync(`${SP}/shots`, { recursive: true }); const B = "http://127.0.0.1:5173";
let failed = 0; const ok = (n, c, x = "") => { console.log(`${c ? "PASS" : "FAIL"} ${n} ${c ? "" : x}`); if (!c) failed++; };
const b = await launch(9336);
const polls = []; b.on(d => { if (d.method === "Network.requestWillBeSent" && /\/api\/runs\/[0-9a-f]{32}$/.test(d.params.request.url) && d.params.request.method === "GET") polls.push(Date.now()); });
const errors = []; b.on(d => { if (d.method === "Runtime.exceptionThrown") errors.push(JSON.stringify(d.params.exceptionDetails).slice(0, 200)); });
await b.goto(B + "/runs");
ok("баннер демо отсутствует в API-режиме", await b.ev(`!document.body.textContent.includes('Демонстрационные данные')`));
ok("пустая история от реального API", await b.ev(`document.querySelector('.empty')!==null`));
await b.goto(B + "/runs/new");
await b.ev(`document.querySelector('input:not([type])').focus()`); await b.send("Input.insertText", { text: TITLE });
await b.ev(`document.querySelector('textarea').focus()`);
await b.send("Input.insertText", { text: DESCRIPTION });
const { root } = await b.send("DOM.getDocument");
const ids = (await b.send("DOM.querySelectorAll", { nodeId: root.nodeId, selector: "input[type=file]" })).nodeIds.slice(1);
const F = ["train", "test", "data_dictionary", "sample_submission"].map(n => `${DATA}/${n}.csv`);
for (let i = 0; i < 4; i++) await b.send("DOM.setFileInputFiles", { nodeId: ids[i], files: [F[i]] });
for (let i = 0; i < 20 && (await b.ev(`document.querySelectorAll(".file-loading").length`)); i++) await b.sleep(300);
await b.clickText("button", "Далее");
// время 10 минут, потоки 2
const setNum = async (idx, val) => { await b.ev(`document.querySelectorAll('input[type=number]')[${idx}].select()`); await b.send("Input.insertText", { text: String(val) }); };
await setNum(0, MINUTES); await setNum(2, 2);
ok("доступность Qwen из конфигурации сервера", await b.ev(`document.querySelector('.summary-box').textContent.includes('Доступна')&&document.querySelector('.summary-box').textContent.includes('fake-qwen')`));
// двойное нажатие «Запустить»
const pt = await b.ev(`(()=>{const e=[...document.querySelectorAll('button')].find(x=>x.textContent.includes('Запустить'));const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);
for (let i = 0; i < 2; i++) { await b.send("Input.dispatchMouseEvent", { type: "mousePressed", x: pt.x, y: pt.y, button: "left", clickCount: 1 }); await b.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: pt.x, y: pt.y, button: "left", clickCount: 1 }); }
for (let i = 0; i < 60 && (await b.ev(`location.pathname`)) === "/runs/new"; i++) await b.sleep(500); // загрузка больших CSV
const path = await b.ev(`location.pathname`);
ok("после запуска открыта страница реального запуска", /^\/runs\/[0-9a-f]{32}$/.test(path), path);
const id = path.split("/").pop();
const list = await (await fetch("http://127.0.0.1:8765/api/runs", { headers: { Host: "127.0.0.1:8765" } })).json();
ok("двойное нажатие не создало второй запуск", list.length === 1, JSON.stringify(list.map(r => r.id)));
const t0 = Date.now(); polls.length = 0;
// перезагрузка страницы посреди выполнения
await b.sleep(1200); await b.send("Page.reload"); polls.length = 0; await b.sleep(1500);
ok("после перезагрузки исследование на месте", await b.ev(`document.querySelector('.run-head h1')?.textContent===${JSON.stringify(TITLE)}`));
// опрос раз в 2 с: ждём завершения
let status = "";
for (let i = 0; i < Number(process.env.E2E_WAIT_SEC ?? 60); i++) { status = await b.ev(`document.querySelector('.run-head .badge')?.textContent||''`); if (!status.includes("Выполняется")) break; await b.sleep(1000); }
ok("запуск дошёл до «Завершено» без ручного обновления", status.includes("Завершено"), status);
await b.sleep(2500);
const afterDone = polls.length; await b.sleep(5000);
ok("опрос прекращается после завершения", polls.length === afterDone, `${afterDone} -> ${polls.length}`);
const steady = polls.filter((t, i) => i === 0 || t - polls[i - 1] > 300); const gaps = steady.slice(1, 6).map((t, i) => t - steady[i]);
ok("интервал опроса ≈2 с", gaps.length > 0 && gaps.every(g => g > 1500 && g < 3500), JSON.stringify(gaps));
const info = await b.ev(`({exps:document.querySelectorAll('.feed-item').length,best:document.querySelector('.stat-accent .stat-value')?.textContent,base:document.querySelectorAll('.stat-value')[0]?.textContent,note:document.querySelector('.alert')?.textContent})`);
ok("в ленте есть измеренные эксперименты Qwen", info.exps >= 3, JSON.stringify(info));
const harness = JSON.parse((await import("node:fs")).readFileSync(`${SP}/runs-web/${id}/harness/baseline.json`, "utf8"));
ok("baseline в интерфейсе = baseline.json harness", info.base === harness.score.value.toFixed(/rmse|mae|mse/i.test(harness.score.metric) ? 2 : 4), `${info.base} vs ${harness.score.value}`);
await b.shot(`${SP}/shots/api_full.png`);
await b.clickBy(`document.querySelectorAll(".feed-item")[1]`);
ok("панель показывает предложенные и запущенные параметры реального эксперимента", await b.ev(`(()=>{const t=document.querySelector('.drawer').textContent;return /depth/.test(t)&&t.includes('Предложено')&&t.includes('Запущено')})()`));
await b.shot(`${SP}/shots/api_drawer.png`); await b.esc();
// скачивание: итоговый CSV именно этого запуска
const dl = await b.ev(`fetch('/api/runs/${id}/download/final').then(r=>r.text())`);
const disk = (await import("node:fs")).readFileSync(`${SP}/runs-web/${id}/inputs/submission.csv`, "utf8");
ok("итоговый CSV = файл этого запуска", dl === disk && dl.startsWith("id,"));
ok("нет исключений JS", errors.length === 0, errors.join("\n"));
console.log(failed ? `FAILED: ${failed}` : "ALL PASSED"); await b.close(); process.exit(failed ? 1 : 0);
