"""Human check of correctness labels.

1) Make a blind sample and an offline annotation page:
    python -m hallu.human_eval sample runs/colab_qwen7b_triviaqa_n400 runs/colab_qwen7b_nq_n400 \
        runs/colab_qwen7b_squad_n400 --out annotation
   -> annotation/annotate.html (open in a browser, no internet needed)
      annotation/items.json

   Sample = per run, N random questions (unbiased estimate of label accuracy)
            + up to M questions where the label methods disagree (hard cases).
   The page never shows any model's label, only question / gold / answer.

2) Annotate: keys C = correct, W = wrong (hallucination), U = unsure, arrows to move.
   Progress autosaves in the browser. Click "Export CSV" when done.

3) Score every label method against your annotations:
    python -m hallu.human_eval score annotation/annotations.csv runs/colab_qwen7b_*_n400
   Prints accuracy, Cohen's kappa, and hallucination precision / recall for each
   method, separately for the random and the disagreement subsets.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

LABEL_COLS_SKIP = {"id", "pred", "gold"}


def _read(p: Path) -> List[Dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def _by_id(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for r in rows:
        out.setdefault(r["id"], r)
    return out


def _dataset(run: Path) -> str:
    for d in ("triviaqa", "nq", "squad", "bioasq"):
        if d in run.name.lower():
            return d
    return run.name


def make_sample(runs: List[Path], out: Path, n_random: int, n_disagree: int, seed: int) -> None:
    rng = random.Random(seed)
    items = []
    for run in runs:
        gens = _by_id(_read(run / "generations.jsonl"))
        labs = _by_id(_read(run / "labels.jsonl"))
        ids = sorted(i for i in gens if i in labs)
        rnd = rng.sample(ids, min(n_random, len(ids)))
        cols = [c for c in next(iter(labs.values())) if c not in LABEL_COLS_SKIP]
        a = "judges_majority" if "judges_majority" in cols else ("judge" if "judge" in cols else None)
        dis_pool = [i for i in ids if i not in set(rnd) and a
                    and labs[i].get(a) is not None and labs[i].get("ensemble") is not None
                    and labs[i][a] != labs[i]["ensemble"]]
        dis = rng.sample(dis_pool, min(n_disagree, len(dis_pool)))
        for subset, chosen in (("random", rnd), ("disagree", dis)):
            for i in chosen:
                g = gens[i]
                items.append({"key": f"{_dataset(run)}::{i}", "dataset": _dataset(run), "run": run.name,
                              "id": i, "subset": subset, "question": g["question"],
                              "gold": g["answers"][:6], "answer": g["most_likely"]["text"]})
    rng.shuffle(items)  # mix datasets and subsets so the annotator can't tell them apart
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    page = HTML.replace("__ITEMS__", json.dumps(
        [{k: it[k] for k in ("key", "question", "gold", "answer")} for it in items], ensure_ascii=False))
    (out / "annotate.html").write_text(page)
    n_r = sum(it["subset"] == "random" for it in items)
    print(f"{len(items)} items ({n_r} random, {len(items) - n_r} disagreement) -> {out}/annotate.html")


def _kappa(a: List[int], b: List[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def score(ann_csv: Path, runs: List[Path], items_json: Path) -> None:
    items = {it["key"]: it for it in json.loads(items_json.read_text())}
    human: Dict[str, int] = {}
    with open(ann_csv) as f:
        for row in csv.DictReader(f):
            v = row["verdict"].strip().lower()
            if v in ("correct", "wrong"):
                human[row["key"]] = int(v == "wrong")  # 1 = hallucination
    labs = {}
    for run in runs:
        for i, r in _by_id(_read(run / "labels.jsonl")).items():
            labs[f"{_dataset(run)}::{i}"] = r
    keys = [k for k in human if k in items and k in labs]
    methods = sorted({c for k in keys for c in labs[k] if c not in LABEL_COLS_SKIP})
    print(f"{len(keys)} annotated items used (unsure / missing skipped)")
    for subset in ("random", "disagree", "all"):
        ks = [k for k in keys if subset == "all" or items[k]["subset"] == subset]
        if not ks:
            continue
        rate = sum(human[k] for k in ks) / len(ks)
        print(f"\n== {subset}: n={len(ks)}, human hallucination rate {rate:.3f}")
        print(f"  {'method':18s} {'acc':>6s} {'kappa':>6s} {'prec':>6s} {'recall':>6s}")
        for m in methods:
            pairs = [(human[k], labs[k][m]) for k in ks if labs[k].get(m) is not None]
            if not pairs:
                continue
            h, p = [x for x, _ in pairs], [y for _, y in pairs]
            acc = sum(x == y for x, y in pairs) / len(pairs)
            tp = sum(x == 1 and y == 1 for x, y in pairs)
            prec = tp / max(1, sum(p))
            rec = tp / max(1, sum(h))
            print(f"  {m:18s} {acc:6.3f} {_kappa(h, p):6.3f} {prec:6.3f} {rec:6.3f}")


def _read_verdicts(p: Path) -> Dict[str, str]:
    with open(p) as f:
        return {r["key"]: r["verdict"].strip().lower() for r in csv.DictReader(f) if r["verdict"].strip()}


def agree(a_csv: Path, b_csv: Path) -> None:
    """Inter-annotator agreement (correct/wrong only; unsure counted separately)."""
    a, b = _read_verdicts(a_csv), _read_verdicts(b_csv)
    both = [k for k in a if k in b]
    cw = [k for k in both if a[k] in ("correct", "wrong") and b[k] in ("correct", "wrong")]
    x = [int(a[k] == "wrong") for k in cw]
    y = [int(b[k] == "wrong") for k in cw]
    acc = sum(i == j for i, j in zip(x, y)) / max(1, len(cw))
    print(f"items in both: {len(both)}; decided by both: {len(cw)}")
    print(f"raw agreement {acc:.3f}, Cohen's kappa {_kappa(x, y):.3f}")
    for k in cw:
        if a[k] != b[k]:
            print(f"  disagree {k}: {a_csv.name}={a[k]}  {b_csv.name}={b[k]}")


def make_subset(items_json: Path, ann_csv: Path, out: Path, n: int, seed: int) -> None:
    """Random subset of already-annotated items (correct/wrong) for a second annotator."""
    items = json.loads(items_json.read_text())
    ann = _read_verdicts(ann_csv)
    pool = [it for it in items if ann.get(it["key"]) in ("correct", "wrong")]
    sub = random.Random(seed).sample(pool, min(n, len(pool)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(sub, ensure_ascii=False, indent=1))
    (out / "annotate.html").write_text(HTML.replace("__ITEMS__", json.dumps(
        [{k: it[k] for k in ("key", "question", "gold", "answer")} for it in sub], ensure_ascii=False)))
    print(f"{len(sub)} items -> {out}/annotate.html")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("agree")
    g.add_argument("a")
    g.add_argument("b")
    t = sub.add_parser("subset")
    t.add_argument("items")
    t.add_argument("annotations")
    t.add_argument("--out", default="annotation_check")
    t.add_argument("--n", type=int, default=40)
    t.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("sample")
    s.add_argument("runs", nargs="+")
    s.add_argument("--out", default="annotation")
    s.add_argument("--n-random", type=int, default=50)
    s.add_argument("--n-disagree", type=int, default=17)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("annotations")
    c.add_argument("runs", nargs="+")
    c.add_argument("--items", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "agree":
        agree(Path(a.a), Path(a.b))
        return
    if a.cmd == "subset":
        make_subset(Path(a.items), Path(a.annotations), Path(a.out), a.n, a.seed)
        return
    if a.cmd == "sample":
        make_sample([Path(r) for r in a.runs], Path(a.out), a.n_random, a.n_disagree, a.seed)
    else:
        items = Path(a.items) if a.items else Path(a.annotations).parent / "items.json"
        score(Path(a.annotations), [Path(r) for r in a.runs], items)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Check</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--ok:#1f7a4d;--bad:#b3261e;--unsure:#8a6d00;--accent:#2b59c3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecece8;--muted:#a3a39c;--line:#33332f;--ok:#4cc38a;--bad:#f2766b;--unsure:#e0b84a;--accent:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 48px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:18px;margin:0}.muted{color:var(--muted);font-size:14px}
.bar{height:6px;background:var(--line);border-radius:3px;margin:12px 0 20px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent);width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.lab{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-top:14px}
.q{font-size:18px;margin-top:4px}.gold{margin-top:4px}.gold span{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 10px;margin:2px 6px 2px 0;font-size:14px}
.ans{font-size:18px;margin-top:4px;padding:10px 12px;border-left:3px solid var(--accent);background:color-mix(in srgb,var(--accent) 7%,transparent);border-radius:4px}
.btns{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:20px}
button{font:inherit;padding:12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.sel{outline:3px solid currentColor}.c{color:var(--ok)}.w{color:var(--bad)}.u{color:var(--unsure)}
.nav{display:flex;justify-content:space-between;margin-top:16px;gap:10px}.nav button{color:var(--ink)}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px}
.help{margin-top:18px;font-size:14px;color:var(--muted)}
</style></head><body><main>
<header><h1>Label Check</h1><div class="muted" id="count"></div></header>
<div class="bar"><i id="prog"></i></div>
<div class="card">
 <div class="lab">Question</div><div class="q" id="q"></div>
 <div class="lab">Correct answer(s)</div><div class="gold" id="gold"></div>
 <div class="lab">Model's answer</div><div class="ans" id="ans"></div>
 <div class="btns">
  <button class="c" id="bc">Correct <kbd>C</kbd></button>
  <button class="w" id="bw">Wrong <kbd>W</kbd></button>
  <button class="u" id="bu">Unsure <kbd>U</kbd></button>
 </div>
</div>
<div class="nav"><button id="prev">&larr; Back</button><button id="next">Next unanswered &rarr;</button><button id="exp">Export CSV</button></div>
<p class="help">Mark <b>Correct</b> if the model's answer states a correct answer or an equivalent (alias, abbreviation, or inside a longer sentence). Mark <b>Wrong</b> if it names a different entity/number/date, contradicts the correct answer, or doesn't answer. Use <b>Unsure</b> when the correct-answer list itself looks wrong or ambiguous. Progress is saved in this browser.</p>
</main>
<script>
const ITEMS=__ITEMS__;const KEY="labelcheck:"+ITEMS.length+":"+(ITEMS[0]||{}).key;
let ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=Math.max(0,ITEMS.findIndex(x=>!ans[x.key]));if(i<0)i=0;
const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function show(){const it=ITEMS[i];if(!it)return;$("q").textContent=it.question;$("ans").textContent=it.answer;
 $("gold").innerHTML="";it.gold.forEach(g=>{const s=document.createElement("span");s.textContent=g;$("gold").appendChild(s)});
 const v=ans[it.key];[["bc","correct"],["bw","wrong"],["bu","unsure"]].forEach(([b,x])=>$(b).classList.toggle("sel",v===x));
 const done=ITEMS.filter(x=>ans[x.key]).length;$("count").textContent=`${i+1} / ${ITEMS.length} · ${done} done`;$("prog").style.width=(100*done/ITEMS.length)+"%"}
function mark(v){ans[ITEMS[i].key]=v;save();const n=ITEMS.findIndex((x,j)=>j>i&&!ans[x.key]);i=n>=0?n:Math.min(i+1,ITEMS.length-1);show()}
function exp(){const rows=[["key","verdict"]].concat(ITEMS.map(x=>[x.key,ans[x.key]||""]));
 const csv=rows.map(r=>r.map(c=>'"'+String(c).replace(/"/g,'""')+'"').join(",")).join("\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="annotations.csv";a.click()}
$("bc").onclick=()=>mark("correct");$("bw").onclick=()=>mark("wrong");$("bu").onclick=()=>mark("unsure");
$("prev").onclick=()=>{i=Math.max(0,i-1);show()};$("next").onclick=()=>{const n=ITEMS.findIndex(x=>!ans[x.key]);if(n>=0)i=n;show()};$("exp").onclick=exp;
document.addEventListener("keydown",e=>{const k=e.key.toLowerCase();if(k==="c")mark("correct");else if(k==="w")mark("wrong");else if(k==="u")mark("unsure");
 else if(e.key==="ArrowLeft"){i=Math.max(0,i-1);show()}else if(e.key==="ArrowRight"){i=Math.min(ITEMS.length-1,i+1);show()}});
show();
</script></body></html>"""


if __name__ == "__main__":
    main()
