// Небольшая синтетическая бинарная задача для e2e: <work>/syn/*.csv и task.txt
import { mkdirSync, writeFileSync } from "node:fs";
const dir = `${process.argv[2]}/syn`;
mkdirSync(dir, { recursive: true });
let seed = 1;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(2 * Math.PI * rnd());
const row = (i, withTarget) => {
  const a = gauss(), b = "xyz"[Math.floor(rnd() * 3)], c = rnd();
  const y = a + (b === "x" ? 0.8 : 0) + gauss() * 0.7 > 0.4 ? 1 : 0;
  return [i, a.toFixed(4), b, c.toFixed(4), ...(withTarget ? [y] : [])].join(",");
};
writeFileSync(`${dir}/train.csv`, ["id,f1,f2,f3,target", ...Array.from({ length: 600 }, (_, i) => row(i, true))].join("\n") + "\n");
writeFileSync(`${dir}/test.csv`, ["id,f1,f2,f3", ...Array.from({ length: 200 }, (_, i) => row(1000 + i, false))].join("\n") + "\n");
writeFileSync(`${dir}/data_dictionary.csv`, "feature,role,dtype,description\nid,id,int,идентификатор\nf1,feature,float,числовой признак\nf2,feature,category,категория\nf3,feature,float,числовой признак\ntarget,target,int,бинарная цель\n");
writeFileSync(`${dir}/sample_submission.csv`, "id,target\n" + Array.from({ length: 200 }, (_, i) => `${1000 + i},0.5\n`).join(""));
writeFileSync(`${dir}/task.txt`, "Бинарная классификация. Целевая переменная `target` принимает значения 0 и 1.\nИдентификатор — `id`.\n\nОсновная метрика — **ROC AUC**.\n\nДля каждого `id` из `test.csv` предскажите вероятность класса 1 в колонке `target`.\nФормат ответа повторяет `sample_submission.csv`.\n");
console.log(dir);
