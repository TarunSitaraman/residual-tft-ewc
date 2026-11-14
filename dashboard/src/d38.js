// D38 (replay ensemble on S10 and S7): the registered verdict grid and the method RMSE comparison.
// Data: results/dashboard/fact_d38_verdicts.csv and fact_d38_method.csv (in Git once exported). Either file may be missing
// while the export is pending; each view then says so instead of drawing. S10 and S7 are never pooled.
import { PATHS } from "../config.js";
import { el, fmt } from "./views.js";
import { dotRows, legend, tableView } from "./charts.js";
import { load, card, takeaway, empty, esc, b, num, NEUTRAL } from "./tabs.js";

const DATASETS = ["S10", "S7"];   // run order (S7 started only after S10 was collected)
const EXPERIMENTS = ["E1", "E2"];
const ACCENT = "var(--c1)";
// What each question compares, for when the export carries no comparison text.
const QUESTION = { Q1: "V2 − GBM-ALL (all conditions stored)", Q2: "V2 − GBM-ER32 (the same 32 stored rows)", Q3: "V2 − linear extrapolation, H+ origins",
  Q4: "V2 − earlier proposed model (S7 shrunk, S10 SEL)", Q5: "90 % coverage of V2 at least 0.85" };
// Readable names for method ids that dim_method does not label yet. Unknown ids are shown as they are.
// Keys are the method ids written by scripts/export_dashboard_data.py (D38_IDS).
const LABEL = { v2: "V2: replay ensemble, calibrated", r32_ens: "Replay ensemble (R32), uncalibrated", r32_single: "Replay, single model (R32)",
  r8_ens_cal: "Replay ensemble (R8), calibrated", r8_ens: "Replay ensemble (R8), uncalibrated", r8_single: "Replay, single model (R8)",
  a0_single: "EWC model, single (A0)", a0_ens: "EWC model, ensemble (A0)", a0_ens_cal: "EWC model, ensemble (A0), calibrated",
  sel: "Earlier proposed model (SEL)", shrunk: "Earlier proposed model (shrunk)", joint: "Joint residual TFT (all stored)",
  gbm_all: "GBM-ALL (all stored)", gbm_er32: "GBM-ER32 (32 stored rows)", gbm_er8: "GBM-ER8 (8 stored rows)", gbm_last: "GBM-LAST (latest only)",
  linear: "Linear extrapolation", ridge: "Ridge", persistence: "Persistence" };
const isV2 = (m) => /^v2$|(^|_)v2(_|$)/i.test(m);
const signed = (x, d = 2) => (x === null || Number.isNaN(x) ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${fmt(Math.abs(x), d)}`);
const CAPTION = "Development test, not confirmatory: the S10 and S7 conditions had already been tested with the earlier model (D36, D34) and read " +
  "per condition before D38 was designed; only the protocol was fixed in advance. On S7, gradient boosting is more accurate on most conditions " +
  "(Q1 and Q2 negative). On S10, the Q1 and Q2 mean gains come mainly from the conditions where gradient boosting fails; the median gains are much smaller.";
const MISSING = (file) => `D38 data not yet exported: ${file} is not in results/dashboard/. Run python scripts/export_dashboard_data.py once the D38 export is added; the results themselves are in docs/DECISIONS.md (D38 run log) and on the Studies page.`;

// The export lists every table it wrote in manifest.json; a table it does not list is not fetched (no 404 in the console).
let manifest;
async function listed(name) {
  if (manifest === undefined) {
    try { const r = await fetch(`${PATHS.data}manifest.json`); manifest = r.ok ? await r.json() : null; } catch { manifest = null; }
  }
  return !manifest?.row_counts || name in manifest.row_counts;
}
async function tryLoad(name) {
  if (!(await listed(name))) return null;
  try { return await load(name); } catch { return null; }
}

/** Show the tooltip on hover and keyboard focus. */
function withTip(node, tip, html) {
  node.setAttribute("tabindex", "0");
  const show = () => tip.show(node, html);
  node.addEventListener("mouseenter", show);
  node.addEventListener("focus", show);
  node.addEventListener("mouseleave", tip.hide);
  node.addEventListener("blur", tip.hide);
}

// ---------- verdict grid ----------
function verdictGrid(rows, tip) {
  const qs = [...new Set(rows.map((r) => r.question))].sort();
  const get = (q, ds, ex, set) => rows.find((r) => r.question === q && r.dataset.toUpperCase() === ds && r.experiment.toUpperCase() === ex && r.seed_set === set);
  // Registered answer for a dataset: positive only if it holds in E1 and E2 and in both seed sets.
  const answerFor = (q, ds) => {
    const rs = rows.filter((r) => r.question === q && r.dataset.toUpperCase() === ds);
    if (!rs.length) return null;
    return rs.every((r) => r.answer === "positive") ? "positive" : "negative";
  };
  const tally = Object.fromEntries(DATASETS.map((ds) => [ds, qs.filter((q) => answerFor(q, ds) === "positive").length]));
  const c = card(DATASETS.map((ds) => `${ds}: ${tally[ds]} of ${qs.length} questions answered positively`).join("; "),
    "Each cell: the answer in that horizon set (main seed set) and the mean difference V2 − comparator over the held-out conditions, main / replicate (pp of SOH; negative favours V2). Q5 shows pooled 90 % coverage instead (target 0.90, threshold 0.85). The answer column needs E1, E2 and both seed sets. Hover or focus a cell for the median, the conditions favouring V2 and p_Holm.",
    "S10 and S7 · P2 · D38");
  const ans = (a) => el("span", { class: `ans ${a || "na"}`, text: a || "–" });
  const head1 = el("tr", {}, el("th", { rowspan: 2, scope: "col", text: "Question" }),
    DATASETS.map((ds) => el("th", { class: "colh", colspan: 3, scope: "colgroup", text: ds === "S10" ? "S10 (64 conditions)" : "S7 (60 conditions)" })));
  const head2 = el("tr", {}, DATASETS.flatMap((ds) => [...EXPERIMENTS.map((ex) => el("th", { class: "colh", scope: "col", text: ex === "E1" ? "E1, h 1-3" : "E2, h 1-6" })),
    el("th", { class: "colh", scope: "col", text: "Answer" })]));
  const body = qs.map((q) => {
    const any = rows.find((r) => r.question === q);
    return el("tr", {}, el("th", { scope: "row" }, el("b", { text: q }), " ", any.comparison || QUESTION[q] || ""),
      DATASETS.flatMap((ds) => [...EXPERIMENTS.map((ex) => {
        const m = get(q, ds, ex, "main"), r = get(q, ds, ex, "replicate");
        if (!m) return el("td", { class: "v", text: "–" });
        const isCov = q === "Q5" || (m.mean_diff_pp === "" && m.coverage_90 !== "");
        const val = isCov ? `coverage ${fmt(num(m.coverage_90), 3)} / ${r ? fmt(num(r.coverage_90), 3) : "–"}`
          : `${signed(num(m.mean_diff_pp), 3)} / ${r ? signed(num(r.mean_diff_pp), 3) : "–"} pp`;
        const td = el("td", { class: "v", "aria-label": `${q}, ${ds} ${ex}: ${m.answer}, ${val}` }, ans(m.answer), el("span", { class: "vnum", text: val }));
        const line = (x, set) => !x ? "" : isCov ? `<br>${set}: coverage ${fmt(num(x.coverage_90), 3)}`
          : `<br>${set}: mean ${signed(num(x.mean_diff_pp), 3)} pp, median ${signed(num(x.median_diff_pp), 3)} pp, ${esc(x.n_favour_v2)} of ${esc(x.n_conditions)} favour V2, p_Holm ${num(x.p_holm)?.toExponential(1) ?? "–"}${x.verdict ? `, ${esc(x.verdict)}` : ""}`;
        withTip(td, tip, `${b(`${q}, ${ds}, ${ex}`)}: ${esc(any.comparison || QUESTION[q] || "")}${line(m, "Main")}${line(r, "Replicate")}`);
        return td;
      }), el("td", { class: "v" }, ans(answerFor(q, ds)))]));
  });
  c.body.append(el("div", { class: "scroll" }, el("table", { class: "vgrid" }, el("thead", {}, head1, head2), el("tbody", {}, body))));
  const tv = tableView([{ key: "dataset", label: "Data" }, { key: "experiment", label: "Exp." }, { key: "seed_set", label: "Seeds" }, { key: "question", label: "Q" },
    { key: "comparison", label: "Comparison" }, { key: "mean_diff_pp", label: "Mean diff (pp)", right: true, fmt: (x) => signed(num(x), 3) },
    { key: "median_diff_pp", label: "Median diff (pp)", right: true, fmt: (x) => signed(num(x), 3) },
    { key: "n_favour_v2", label: "Favour V2", right: true, fmt: (x, r) => (x === "" ? "–" : `${x} of ${r.n_conditions}`) },
    { key: "p_holm", label: "p_Holm", right: true, fmt: (x) => (x === "" ? "–" : Number(x).toExponential(1)) },
    { key: "verdict", label: "Verdict" }, { key: "answer", label: "Answer" }, { key: "coverage_90", label: "Coverage 90 %", right: true, fmt: (x) => (x === "" ? "–" : fmt(num(x), 3)) }],
  rows, "Show every row as a table");
  c.setFoot(el("div", { class: "clegend" }, el("span", { class: "ans positive", text: "positive: holds for V2" }), el("span", { class: "ans negative", text: "negative: does not hold" })), tv.btn);
  c.body.append(tv.table);
  return { node: c.node, answerFor, qs };
}

// ---------- method RMSE comparison ----------
let stat = "mean";   // mean or median RMSE over conditions; kept across redraws

function methodPanels(rows, labels, state, tip) {
  const ex = state.setting.toUpperCase();
  const nm = (m) => LABEL[m] || labels[m] || m;   // D38 ids first: dim_method's "joint" is the MLP, not the joint residual TFT
  const seg = el("div", { class: "seg", role: "group", "aria-label": "Statistic over conditions" },
    [["mean", "Mean over conditions"], ["median", "Median over conditions"]].map(([k, t]) => el("button", { type: "button", "aria-pressed": String(k === stat), "data-k": k, text: t })));
  const cards = DATASETS.map((ds) => {
    const sel = rows.filter((r) => r.dataset.toUpperCase() === ds && r.experiment.toUpperCase() === ex);
    const c = card(ds, "", `${ds} · P2 · D38 · ${ex}`, "half");
    return { ds, sel, c };
  });
  function draw() {
    seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.k === stat)));
    const key = `${stat}_rmse_pp`;
    for (const { ds, sel, c } of cards) {
      const title = c.node.querySelector("h2"), cap = c.node.querySelector("figcaption");
      const methods = [...new Set(sel.map((r) => r.method))];
      const val = (m, set) => { const r = sel.find((q) => q.method === m && q.seed_set === set); return r ? num(r[key]) : null; };
      const items = methods.map((m) => ({ m, main: val(m, "main") ?? val(m, "replicate"), rep: val(m, "replicate") }))
        .filter((x) => x.main !== null).sort((a, z) => a.main - z.main);
      if (!items.length) { c.body.replaceChildren(empty(`No ${ds} ${ex} rows in fact_d38_method.`)); title.textContent = ds; continue; }
      const rank = items.findIndex((x) => isV2(x.m)) + 1;
      title.textContent = rank === 1 ? `${ds}, ${ex}: V2 has the lowest ${stat} RMSE of ${items.length} methods`
        : `${ds}, ${ex}: ${rank ? `V2 ranks ${rank} of ${items.length} by ${stat} RMSE` : `${items.length} methods`}; lowest: ${nm(items[0].m)}`;
      cap.textContent = `${stat === "mean" ? "Mean" : "Median"} test RMSE over the ${ds === "S10" ? "64" : "60"} held-out conditions, pp of SOH; lower is better. Filled: main seed set; ring: replicate (deterministic comparators have one value). V2 is the replay ensemble, calibrated. ${ds}, P2, D38, ${ex}.`;
      const labelW = Math.min(250, innerWidth * 0.5), fit = Math.floor((labelW - 14) / 7);   // about 7 px per character
      const short = (t) => (t.length > fit ? `${t.slice(0, fit - 1)}…` : t);   // full name stays in the tooltip and the table
      dotRows(c.body, { tip, labelW, rowH: 30, x: { label: `${stat === "mean" ? "Mean" : "Median"} RMSE over conditions (pp of SOH)`, min: 0, fmt: (v) => v.toFixed(1) },
        rows: items.map((x) => {
          const v2 = isV2(x.m), color = v2 ? ACCENT : NEUTRAL;
          const t = `${b(nm(x.m))}, ${ds} ${ex}<br>${stat} RMSE: main ${fmt(x.main, 3)} pp${x.rep !== null ? `, replicate ${fmt(x.rep, 3)} pp` : ""}`;
          return { label: short(nm(x.m)), emph: v2, range: x.rep !== null ? [Math.min(x.main, x.rep), Math.max(x.main, x.rep)] : null,
            marks: [...(x.rep !== null ? [{ v: x.rep, color, open: true, r: 6.5, tip: t }] : []), { v: x.main, color, r: 4.5, tip: t }] };
        }) });
      const tv = tableView([{ key: "method", label: "Method", fmt: nm }, { key: "seed_set", label: "Seeds" },
        { key: "mean_rmse_pp", label: "Mean RMSE (pp)", right: true, fmt: (x) => fmt(num(x), 3) }, { key: "median_rmse_pp", label: "Median RMSE (pp)", right: true, fmt: (x) => fmt(num(x), 3) },
        { key: "coverage_90", label: "Coverage 90 %", right: true, fmt: (x) => (x === "" ? "–" : fmt(num(x), 3)) }, { key: "coverage_80", label: "Coverage 80 %", right: true, fmt: (x) => (x === "" ? "–" : fmt(num(x), 3)) }],
      sel);
      c.setFoot(legend([{ label: "V2, main seed set", color: ACCENT, kind: "dot" }, { label: "V2, replicate", color: ACCENT, kind: "ring" }, { label: "Comparator", color: NEUTRAL, kind: "dot" }]), tv.btn);
      c.body.append(tv.table);
    }
  }
  seg.addEventListener("click", (e) => { const k = e.target.closest("button")?.dataset.k; if (k) { stat = k; draw(); } });
  draw();
  return [el("div", { class: "seg-row" }, el("span", { class: "sub", text: "Statistic:" }), seg), ...cards.map((x) => x.c.node)];
}

export async function renderD38(panel, ctx) {
  const { state, tip } = ctx;
  const [verdicts, methods, dim] = await Promise.all([tryLoad("fact_d38_verdicts"), tryLoad("fact_d38_method"), tryLoad("dim_method")]);
  const labels = Object.fromEntries((dim || []).map((m) => [m.method, m.label]));
  const kids = [];
  let grid = null;
  if (verdicts?.length) grid = verdictGrid(verdicts, tip);
  if (grid) {
    const say = (ds) => {
      const pos = grid.qs.filter((q) => grid.answerFor(q, ds) === "positive"), neg = grid.qs.filter((q) => grid.answerFor(q, ds) === "negative");
      return `on ${ds} ${pos.length ? `positive for ${b(pos.join(", "))}` : "positive for none"}${neg.length ? ` and negative for ${b(neg.join(", "))}` : ""}`;
    };
    kids.push(takeaway(`The replay ensemble (V2)'s registered answers are ${say("S10")}; ${say("S7")}. A question is answered positively for a dataset only if it holds in E1 and E2 and in both seed sets.`, CAPTION));
  } else kids.push(takeaway("The replay-ensemble study D38 tests a residual TFT with a small replay memory, a three-seed ensemble and predict-before-training calibration on the S10 and S7 conditions.", CAPTION));
  kids.push(grid ? grid.node : (() => { const c = card("Registered answers by dataset and horizon set", "", "S10 and S7 · P2 · D38"); c.body.append(el("p", { class: "notice", text: MISSING("fact_d38_verdicts.csv") })); return c.node; })());
  if (methods?.length) kids.push(...methodPanels(methods, labels, state, tip));
  else { const c = card("Test error of every method", "", `S10 and S7 · P2 · D38 · ${state.setting.toUpperCase()}`); c.body.append(el("p", { class: "notice", text: MISSING("fact_d38_method.csv") })); kids.push(c.node); }
  kids.push(el("p", { class: "sub" }, "Scope: first-life cells (S10 one cell type in three laboratories, S7 one cell type in one laboratory), not a second-life result. V2 keeps 40 rows of each earlier condition, so it is compared as a bounded-memory method; the no-storage statements about the EWC model do not apply to it. Full provenance: ",
    el("a", { href: "#page=studies&sec=study-D38", text: "D38 on the Studies page" }), "."));
  panel.replaceChildren(...kids);
}
