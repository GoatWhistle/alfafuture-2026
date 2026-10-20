// Демо-режим в Chrome: клавиатура, диалоги, все сценарии, раскладка.
// Запуск: `make web-demo`, затем `node e2e/make-data.mjs <work> && node e2e/demo.mjs <work>`
import { mkdirSync } from "node:fs";
import { launch } from "./cdp.mjs";
const WORK = process.argv[2]; mkdirSync(`${WORK}/shots`, { recursive: true });
const results = []; let failed = 0;
const ok = (name, cond, extra = "") => { console.log(`${cond ? "PASS" : "FAIL"} ${name} ${cond ? "" : extra}`); if (!cond) failed++; };
const B = "http://127.0.0.1:5173";
const b = await launch(9333);
const errors = []; b.on(d => { if (d.method === "Runtime.exceptionThrown") errors.push(JSON.stringify(d.params.exceptionDetails).slice(0, 200)); });
const describe = `(()=>{const a=document.activeElement;return a&&a!==document.body?(a.tagName+':'+(a.getAttribute('aria-label')||a.textContent||'').trim().slice(0,40)):'body'})()`;
const outline = `(()=>{const a=document.activeElement;const s=getComputedStyle(a);return s.outlineStyle!=='none'&&parseFloat(s.outlineWidth)>=2})()`;

// 1. клавиатурный обход /runs
await b.goto(B + "/runs");
const seen = [];
for (let i = 0; i < 16; i++) { await b.tab(); seen.push(await b.ev(describe)); if (i === 3) ok("фокус виден (outline ≥2px, :focus-visible)", await b.ev(outline)); }
ok("Tab достигает навигации и «Новое исследование»", seen.some(s => s.includes("История")) && seen.filter(s => s.includes("Новое исследование")).length >= 1, seen.join(" | "));
ok("Tab достигает фильтров", seen.some(s => s.includes("Выполняются")), seen.join(" | "));
ok("Tab достигает ссылок запусков", seen.some(s => s.includes("Кросс-продажи")), seen.join(" | "));

// 2. фильтры
await b.clickText("button.chip", "Ошибки");
ok("фильтр «Ошибки» оставляет 1 строку", (await b.ev(`document.querySelectorAll('tbody tr').length`)) === 1);
await b.clickText("button.chip", "Все");
ok("фильтр «Все» — 6 строк", (await b.ev(`document.querySelectorAll('tbody tr').length`)) === 6);

// 3. боковая панель с клавиатуры
await b.goto(B + "/runs/demo-improved");
await b.ev(`document.querySelector('.feed-item').focus()`);
await b.enter();
ok("Enter открывает панель эксперимента", await b.ev(`!!document.querySelector('[role=dialog].drawer')`));
ok("фокус внутри панели", await b.ev(`document.querySelector('.drawer').contains(document.activeElement)`));
ok("панель показывает «Предложено» и «Запущено» раздельно", await b.ev(`document.querySelector('.drawer').textContent.includes('Предложено')&&document.querySelector('.drawer').textContent.includes('Запущено')`));
await b.shot(`${WORK}/shots/` + "drawer.png");
for (let i = 0; i < 40; i++) await b.tab();
ok("Tab не выходит за пределы панели", await b.ev(`document.querySelector('.drawer').contains(document.activeElement)`));
await b.esc();
ok("Esc закрывает панель", await b.ev(`!document.querySelector('.drawer')`));
ok("фокус вернулся на элемент ленты", await b.ev(`document.activeElement.classList.contains('feed-item')`));
// панель кодa
await b.clickText(".feed-item", "Эксперимент e2");
ok("блок кода сворачиваемый", await b.ev(`!!document.querySelector('.drawer details.code')&&!document.querySelector('.drawer details.code').open`));
await b.esc();

// 4. воспроизведение и прерывание
await b.goto(B + "/runs/demo-running");
const step0 = await b.ev(`document.querySelector('.playback').textContent`);
ok("демо стоит на паузе до старта", /шаг 5 из 10/.test(step0), step0);
await b.clickText(".playback button", "Старт");
await b.sleep(3600);
const step1 = await b.ev(`document.querySelector('.playback').textContent`);
ok("после Старт шаг растёт", /шаг ([6-9]|10) из 10/.test(step1), step1);
await b.clickText(".playback button", "Пауза");
const s2 = await b.ev(`document.querySelector('.playback').textContent`); await b.sleep(2000);
ok("Пауза останавливает проигрывание", s2 === await b.ev(`document.querySelector('.playback').textContent`));
await b.clickText(".playback button", "Старт");
await b.clickText("button", "Прервать запуск");
ok("прерывание требует подтверждения", await b.ev(`!!document.querySelector('[role=dialog].dialog')`));
await b.esc();
ok("Esc отменяет прерывание, запуск идёт", await b.ev(`!document.querySelector('.dialog')&&document.body.textContent.includes('Выполняется')`));
await b.clickText("button", "Прервать запуск");
await b.clickText(".dialog button", "Прервать");
await b.sleep(400);
ok("после подтверждения статус «Остановлено»", await b.ev(`document.querySelector('.run-head .badge').textContent.includes('Остановлено')`));
ok("текущий CSV доступен, итоговый — нет", await b.ev(`(()=>{const bs=[...document.querySelectorAll('.actions button')];const g=t=>bs.find(x=>x.textContent.includes(t));return !g('текущий').disabled&&g('итоговый').disabled})()`));
ok("кнопка «Прервать» исчезла", await b.ev(`![...document.querySelectorAll('button')].some(x=>x.textContent.includes('Прервать запуск'))`));
await b.clickText(".playback button", "Повтор");
await b.sleep(300);
ok("Повтор перезапускает сценарий", await b.ev(`document.querySelector('.run-head .badge').textContent.includes('Выполняется')`));
await b.shot(`${WORK}/shots/` + "running.png");

// 5. все состояния
for (const [id, needle, label] of [
  ["demo-improved", "Завершено", "улучшение"], ["demo-noimp", "Ни одна гипотеза не превзошла baseline", "нет улучшения"],
  ["demo-qwenfail", "Qwen недоступна", "отказ Qwen"], ["demo-timeout", "Worker timeout", "timeout"], ["demo-baseline", "Режим baseline", "baseline"]]) {
  await b.goto(B + "/runs/" + id, 600);
  ok(`состояние: ${label}`, await b.ev(`document.body.textContent.includes(${JSON.stringify(needle)})`));
}
ok("отклонение по ресурсам видно внутри попытки", await b.ev(`document.querySelector('.badge-exp-rejected')?.closest('.feed-item')?.textContent.match(/1\\D140/)!==null`));
await b.goto(B + "/runs/demo-qwenfail", 600);
ok("при отказе Qwen текущий CSV доступен, итоговый — нет", await b.ev(`(()=>{const bs=[...document.querySelectorAll('.actions button')];const g=t=>bs.find(x=>x.textContent.includes(t));return !g('текущий').disabled&&g('итоговый').disabled})()`));
await b.goto(B + "/runs/demo-timeout", 600);
await b.clickText(".feed-item", "Эксперимент e3");
ok("timeout показан в панели", await b.ev(`document.querySelector('.drawer').textContent.includes('Worker timeout')`));
await b.esc();
ok("баннер демо виден на каждой странице", await b.ev(`document.body.textContent.includes('Демонстрационные данные')`));

// 6. пустая история
await b.goto(B + "/runs");
await b.clickText("button", "Показать пустую историю");
ok("пустая история: две кнопки", await b.ev(`(()=>{const e=document.querySelector('.empty');return !!e&&e.textContent.includes('Создать исследование')&&e.textContent.includes('Посмотреть демо')})()`));
await b.shot(`${WORK}/shots/` + "empty.png");
await b.clickText(".empty button", "Посмотреть демо");
ok("«Посмотреть демо» открывает запуск", /\/runs\/demo-/.test(await b.ev(`location.pathname`)));

// 7. новое исследование
await b.goto(B + "/runs/new");
await b.clickText("button", "Далее");
ok("валидация шага 1 без данных", await b.ev(`document.body.textContent.includes('Укажите название')&&document.body.textContent.includes('Не загружено')`));
await b.ev(`(()=>{const i=document.querySelector('input:not([type])');i.focus()})()`);
await b.send("Input.insertText", { text: "Проверка из браузера" });
await b.ev(`document.querySelector('textarea').focus()`);
await b.send("Input.insertText", { text: "Бинарная классификация, метрика ROC AUC" });
const { root } = await b.send("DOM.getDocument");
const inputs = (await b.send("DOM.querySelectorAll", { nodeId: root.nodeId, selector: "input[type=file]" })).nodeIds.slice(1);
const FILES = ["train", "test", "data_dictionary", "sample_submission"].map(n => `${WORK}/syn/${n}.csv`);
for (let i = 0; i < 4; i++) await b.send("DOM.setFileInputFiles", { nodeId: inputs[i], files: [FILES[i]] });
for (let i = 0; i < 20 && (await b.ev(`document.querySelectorAll(".file-loading").length`)); i++) await b.sleep(300);
const fileStates = await b.ev(`[...document.querySelectorAll('.file')].map(f=>f.className)`);
ok("все файлы показаны как загруженные", fileStates?.length === 4 && fileStates.every(c => c.includes('file-ready')), JSON.stringify(fileStates));
await b.clickText("button", "Далее");
ok("переход к шагу «Ресурсы», значения по умолчанию 30/32000/4", await b.ev(`(()=>{const v=[...document.querySelectorAll('input[type=number]')].map(i=>i.value);return v[0]==='30'&&v[1]==='32000'&&v[2]==='4'})()`));
ok("доп. настройки раскрываются отдельно", await b.ev(`(()=>{const d=document.querySelector('details.adv');return !d.open})()`));
ok("сводка и доступность Qwen", await b.ev(`document.querySelector('.summary-box').textContent.includes('Qwen')&&document.querySelector('.summary-box').textContent.includes('Доступна')`));
await b.shot(`${WORK}/shots/` + "step2.png");
await b.clickText("button", "Запустить");
await b.sleep(600);
ok("запуск открывает страницу нового исследования", /\/runs\/demo-new-1/.test(await b.ev(`location.pathname`)));
ok("новое демо выполняется", await b.ev(`document.querySelector('.run-head .badge').textContent.includes('Выполняется')`));

// 8. раскладка без переполнения
for (const [w, h] of [[1440, 900], [1280, 720], [800, 900]]) {
  await b.size(w, h);
  for (const p of ["/runs", "/runs/new", "/runs/demo-improved", "/runs/demo-timeout"]) {
    await b.goto(B + p, 500);
    const over = await b.ev(`document.documentElement.scrollWidth-window.innerWidth`);
    ok(`без горизонтального переполнения ${p} @${w}`, over <= 0 || (p === "/runs" && w < 1000), `overflow ${over}`);
  }
}

// Наложения: соседние элементы в строках/сетках не пересекаются, текст не вылезает из блоков
const OVERLAP = `(()=>{const bad=[];const sel='.row,.summary,.stagebar,.actions,.filters,.feed-top,.page-head,.run-head,.meter-top,.kv,.file,.grid3,.two,.cols,.feed-steps,.playback,.chart-head';
for(const p of document.querySelectorAll(sel)){const k=[...p.children].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.height>0&&getComputedStyle(e).position!=='absolute'});
for(let i=0;i<k.length;i++)for(let j=i+1;j<k.length;j++){const a=k[i].getBoundingClientRect(),b=k[j].getBoundingClientRect();
const ix=Math.min(a.right,b.right)-Math.max(a.left,b.left),iy=Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top);
if(ix>1&&iy>1)bad.push(p.className+': '+(k[i].textContent||'').trim().slice(0,20)+' × '+(k[j].textContent||'').trim().slice(0,20))}}
for(const e of document.querySelectorAll('.stat-value,.badge,.btn,.stage,.chip,.tag,h1'))if(e.scrollWidth>e.clientWidth+1)bad.push('overflow '+e.className+': '+e.textContent.trim().slice(0,30));
return bad})()`;
for (const [w, h] of [[1440, 900], [1280, 720], [800, 900]]) {
  await b.size(w, h);
  for (const p of ["/runs", "/runs/new", "/runs/demo-improved", "/runs/demo-timeout", "/runs/demo-qwenfail", "/runs/demo-running"]) {
    await b.goto(B + p, 600);
    const bad = await b.ev(OVERLAP);
    ok(`без наложений ${p} @${w}`, Array.isArray(bad) && bad.length === 0, JSON.stringify(bad));
  }
}
ok("нет исключений JS", errors.length === 0, errors.join("\n"));
console.log(failed ? `FAILED: ${failed}` : "ALL PASSED");
await b.close(); process.exit(failed ? 1 : 0);
