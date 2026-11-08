// Many conditions (D34 on S7, D36 on S10) and Registered results (every replication-rule verdict).
// Data: results/dashboard/fact_many_condition.csv and fact_verdicts.csv (in Git).
import { el, fmt } from "./views.js";
import { dotRows, scatter, matrix, legend, tableView, seqFill, divFill, rampLegend, DIV_GRADIENT } from "./charts.js";
import { load, card, takeaway, empty, esc, b, num, mean, NEUTRAL } from "./tabs.js";

// ---------- Many conditions ----------
const SETS = [["S7", "tft_residual_ewc_shrunk", "var(--c7)", "S7: 60 KIT conditions"], ["S10", "tft_residual_ewc_selective", "var(--c6)", "S10: 64 Stroebl conditions"]];
const COMPARE = [["gbm_last", "Gradient boosting, latest condition only", false], ["linear_extrapolation", "Linear extrapolation", false],
  ["ridge_shared", "Ridge", false], ["persistence", "Persistence", false],
  ["gbm_all", "Gradient boosting, all conditions stored", true], ["tft_residual_joint", "Joint residual TFT, all conditions stored", true]];
const AGE = { calendar: ["Calendar", "var(--c4)"], cyclic: ["Cyclic", "var(--c2)"], wltp: ["WLTP drive cycle", "var(--c5)"] };
let compareTo = "gbm_last";   // the comparator of the paired views; kept across redraws

/** Per dataset: condition -> {proposed, comparator rows}, main seed set (baselines are deterministic). */
function byCondition(rows, dataset, proposed) {
  const out = new Map();
  for (const r of rows.filter((q) => q.dataset === dataset && q.seed_set === "main")) {
    const c = out.get(r.condition) || out.set(r.condition, { condition: r.condition, age: r.age_type, temp: num(r.age_temp), lab: r.lab }).get(r.condition);
    c[r.method === proposed ? "proposed" : r.method] = num(r.rmse);
  }
  return [...out.values()].filter((c) => c.proposed !== null && c.proposed !== undefined);
}

export async function renderMany(panel, ctx) {
  const { state, tip } = ctx;
  const all = (await load("fact_many_condition")).filter((r) => r.setting === state.setting);
  if (!all.length) { panel.replaceChildren(empty("No many-condition results: run python scripts/export_dashboard_data.py.")); return; }
  const data = Object.fromEntries(SETS.map(([ds, p]) => [ds, byCondition(all, ds, p)]));
  const tag = (ds) => `${ds} · P2 · ${ds === "S7" ? "D34" : "D36"} · ${state.setting.toUpperCase()} · main seeds`;
  const share = (ds, m) => { const c = data[ds].filter((x) => x[m] !== undefined); return c.length ? [c.filter((x) => x.proposed < x[m]).length, c.length] : null; };
  const meanDiff = (ds, m) => mean(data[ds].filter((x) => x[m] !== undefined).map((x) => x.proposed - x[m]));

  // 1. Win rate against every comparator, both datasets.
  const wr = card("How often the continual model is more accurate", "Share of held-out conditions on which the proposed model's RMSE is lower than the comparator's. Comparators below the line may store every earlier condition's data; the proposed model may not.", `S7 and S10 · P2 · D34 and D36 · ${state.setting.toUpperCase()} · main seeds`);
  dotRows(wr.body, { tip, labelW: Math.min(300, innerWidth * 0.42), x: { label: "Share of conditions where the proposed model is better", min: 0, max: 1, fmt: (v) => `${Math.round(v * 100)} %` },
    refs: [{ x: 0.5, label: "half" }],
    rows: COMPARE.map(([m, lab, stores]) => ({ label: stores ? `${lab} ↓` : lab, emph: m === compareTo,
      marks: SETS.map(([ds, , color]) => { const s = share(ds, m); return s && { v: s[0] / s[1], color, r: 6,
        tip: `${b(ds)} against ${esc(lab)}<br>Better on ${s[0]} of ${s[1]} conditions<br>Mean difference ${fmt(meanDiff(ds, m), 3)} pp (negative favours the model)` }; }).filter(Boolean) })) });
  wr.setFoot(legend(SETS.map(([, , color, lab]) => ({ label: lab, color, kind: "dot" }))), el("span", { class: "sub", text: "↓ stores all earlier data." }));

  // 2. Paired scatter per dataset against the chosen comparator (log axes: S7 errors span two decades).
  const seg = el("div", { class: "seg", role: "group", "aria-label": "Comparator" }, COMPARE.map(([m, lab]) =>
    el("button", { type: "button", "aria-pressed": String(m === compareTo), text: lab.replace("Gradient boosting", "GBM").replace(", all conditions stored", ", all stored") })));
  const pairs = SETS.map(([ds]) => card(`${ds}: every held-out condition`, "Each point is one held-out condition. Below the diagonal the proposed model is more accurate. Both axes on a log scale.", tag(ds), "half"));
  const heat = card("Where the comparator wins", "Mean of proposed minus comparator RMSE over the conditions in each cell, by ageing type and temperature. Blue: the proposed model is more accurate; red: the comparator is.", `S7 and S10 · P2 · ${state.setting.toUpperCase()} · main seeds`);
  const lg = (v) => Math.log10(Math.max(v, 0.05)), tick = (v) => { const x = 10 ** v; return x < 1 ? x.toFixed(1) : x.toFixed(0); };

  function draw() {
    const lab = COMPARE.find(([m]) => m === compareTo)[1];
    seg.querySelectorAll("button").forEach((btn, i) => btn.setAttribute("aria-pressed", String(COMPARE[i][0] === compareTo)));
    SETS.forEach(([ds], i) => {
      const pts = data[ds].filter((c) => c[compareTo] !== undefined);
      const s = share(ds, compareTo);
      scatter(pairs[i].body, { tip, diag: "equal error", square: true,
        x: { label: `${lab} RMSE (pp)`, fmt: tick }, y: { label: "Proposed model RMSE (pp)", fmt: tick },
        points: pts.map((c) => ({ x: lg(c[compareTo]), y: lg(c.proposed), color: (AGE[c.age] || ["", NEUTRAL])[1], r: 4.5,
          tip: `${b(c.condition)}: ${esc((AGE[c.age] || [c.age])[0])}, ${fmt(c.temp, 0)} °C${c.lab ? `, ${esc(c.lab)}` : ""}<br>Proposed ${fmt(c.proposed, 3)} pp, comparator ${fmt(c[compareTo], 3)} pp` })) });
      pairs[i].setFoot(legend(Object.values(AGE).filter(([n]) => pts.some((c) => (AGE[c.age] || [])[0] === n)).map(([n, color]) => ({ label: n, color, kind: "dot" }))),
        el("span", { class: "sub", text: s ? `Better on ${s[0]} of ${s[1]}; mean difference ${fmt(meanDiff(ds, compareTo), 2)} pp.` : "" }));
    });
    // Ageing type x temperature, both datasets.
    const cells = new Map(), temps = new Set();
    for (const [ds] of SETS) for (const c of data[ds]) {
      if (c[compareTo] === undefined) continue;
      const k = `${ds}|${c.age}|${c.temp}`;
      (cells.get(k) || cells.set(k, []).get(k)).push(c.proposed - c[compareTo]);
      temps.add(c.temp);
    }
    const cols = [...temps].sort((a, z) => a - z).map((t) => ({ key: t, label: `${fmt(t, 0)} °C` }));
    const rows = SETS.flatMap(([ds]) => Object.keys(AGE).filter((a) => data[ds].some((c) => c.age === a)).map((a) => ({ key: `${ds}|${a}`, label: `${ds}, ${a === "wltp" ? "WLTP" : AGE[a][0].toLowerCase()}` })));
    const vals = [...cells.values()].map(mean), lim = Math.max(...vals.map(Math.abs), 0.1);
    matrix(heat.body, { tip, corner: "Dataset, ageing", rows, cols,
      cell: (r, c) => { const v = cells.get(`${r.key}|${c.key}`); return v ? { v: mean(v), tip: `${b(r.label)} at ${c.label}<br>${v.length} condition${v.length > 1 ? "s" : ""}, mean difference ${fmt(mean(v), 3)} pp` } : null; },
      fill: (v) => divFill(v / lim), fmtV: (v) => (v > 0 ? "+" : "") + v.toFixed(2) });
    heat.setFoot(rampLegend(`−${fmt(lim)} pp`, `+${fmt(lim)} pp`, DIV_GRADIENT, `against ${lab.toLowerCase()}`));
  }
  seg.addEventListener("click", (e) => { const i = [...seg.children].indexOf(e.target); if (i >= 0) { compareTo = COMPARE[i][0]; draw(); } });
  draw();

  const s10 = share("S10", "gbm_last"), s7 = share("S7", "gbm_last"), a10 = share("S10", "gbm_all"), a7 = share("S7", "gbm_all");
  panel.replaceChildren(takeaway(
    `Held to the same no-storage rule, gradient boosting loses to the continual model on ${b(`${s10[0]} of ${s10[1]}`)} S10 conditions and ${b(`${s7[0]} of ${s7[1]}`)} S7 conditions. Allowed to store every earlier condition, it is better on ${a10[1] - a10[0]} of ${a10[1]} (S10) and ${a7[1] - a7[0]} of ${a7[1]} (S7).`,
    "Two first-life datasets added after the S6 results to test many conditions at once (post hoc in motivation; protocols registered before running). Unit of analysis: the held-out condition."),
  wr.node, el("div", { class: "seg-row" }, el("span", { class: "sub", text: "Compare the proposed model with:" }), seg), ...pairs.map((p) => p.node), heat.node);
}

// ---------- Registered results ----------
const TONE = { "replicated, a better": ["Better, replicated", "var(--good)"], "replicated, a worse": ["Worse, replicated", "var(--bad)"],
  "practically equivalent": ["Practically equivalent", "var(--ink-2)"], "not replicated": ["Not replicated", "var(--neutral)"] };
const METRIC = { final_avg_error: "stream error", p2_error: "unseen-condition error", p2_error_hplus: "unseen, H+ origins", p2_error_hminus: "unseen, H− origins",
  rmse: "error over conditions", rmse_hplus: "over conditions, H+", rmse_hminus: "over conditions, H−", is90: "90 % interval score", is80: "80 % interval score", forgetting: "forgetting" };
const PROPOSED_IDS = new Set(["tft_residual_ewc_shrunk", "tft_residual_ewc_selective"]);
const DATASET = { D34: "S7", D36: "S10" };
const EXTRA_LABEL = { gbm_last_direct: "GBM-LAST, direct", gbm_last_residual: "GBM-LAST, residual" };   // D35 arms, not in dim_method

export async function renderEvidence(panel, ctx) {
  const { state, tip } = ctx;
  const [verdicts, methods] = await Promise.all([load("fact_verdicts"), load("dim_method")]);
  const label = Object.fromEntries(methods.map((m) => [m.method, m.label]));
  const short = (m) => (PROPOSED_IDS.has(m) ? "Proposed" : (label[m] || EXTRA_LABEL[m] || m).replace(" (proposed)", ""));
  const rows = verdicts.filter((v) => v.setting === state.setting && v.family === "primary" && TONE[v.verdict])
    .sort((a, z) => z.experiment.localeCompare(a.experiment));
  const ds = (v) => v.dataset || DATASET[v.experiment] || "S6";

  const fp = card("Every registered primary comparison", "Each row is one registered question: model a minus model b, in pp (interval score for the 90 % rows). Filled: main seed set; ring: replicate. Left of zero favours a. Colour gives the verdict under the replication rule.", `All datasets · ${state.setting.toUpperCase()}`);
  const lo = Math.min(0, ...rows.flatMap((v) => [num(v.diff_main), num(v.diff_replicate)]));
  dotRows(fp.body, { tip, labelW: Math.min(430, innerWidth * 0.5), rowH: 30, x: { label: "Difference a − b (pp)", min: Math.floor(lo * 2) / 2, fmt: (v) => v.toFixed(1) }, refs: [{ x: 0, label: "no difference" }],
    rows: rows.map((v) => {
      const [word, color] = TONE[v.verdict];
      const t = `${b(`${v.experiment}, ${ds(v)}`)}: ${esc(short(v.a))} − ${esc(short(v.b))}, ${esc(METRIC[v.metric] || v.metric)}<br>Main ${fmt(num(v.diff_main), 3)} pp (p_Holm ${num(v.p_holm_main)?.toExponential(1)}), replicate ${fmt(num(v.diff_replicate), 3)} pp<br>${esc(word)}${v.experiment === "D35" ? " (post hoc)" : ""}`;
      return { label: `${v.experiment} ${ds(v)} · ${short(v.a)} − ${short(v.b)} · ${METRIC[v.metric] || v.metric}`,
        range: [Math.min(num(v.diff_main), num(v.diff_replicate)), Math.max(num(v.diff_main), num(v.diff_replicate))],
        marks: [{ v: num(v.diff_replicate), color, open: true, r: 6.5, tip: t }, { v: num(v.diff_main), color, r: 4, tip: t }] };   // ring under the dot
    }) });
  fp.setFoot(legend(Object.values(TONE).map(([w, color]) => ({ label: w, color, kind: "dot" }))));

  // Tally of every verdict (primary and secondary) per study.
  const all = verdicts.filter((v) => v.setting === state.setting && TONE[v.verdict]);
  const exps = [...new Set(all.map((v) => v.experiment))].sort().reverse();
  const tally = card("Verdicts per study", "Number of registered comparisons per study and outcome, primary and secondary questions together. Studies D23, D24 and D27-D31 report their results without this verdict table and are on the Studies page.", `All datasets · ${state.setting.toUpperCase()}`);
  const count = (e, k) => all.filter((v) => v.experiment === e && v.verdict === k).length;
  const maxN = Math.max(...exps.flatMap((e) => Object.keys(TONE).map((k) => count(e, k))), 1);
  matrix(tally.body, { tip, corner: "Study", rows: exps.map((e) => ({ key: e, label: `${e}${DATASET[e] ? ` (${DATASET[e]})` : ""}` })),
    cols: Object.entries(TONE).map(([k, [w]]) => ({ key: k, label: w.replace(", replicated", "").replace("Practically e", "E") })),
    cell: (r, c) => ({ v: count(r.key, c.key), tip: `${b(r.label)}: ${count(r.key, c.key)} ${esc(TONE[c.key][0].toLowerCase())}` }),
    fill: (v) => seqFill(v / maxN), fmtV: (v) => String(v) });

  const n = (k) => rows.filter((v) => v.verdict === k).length;
  const table = tableView([{ key: "experiment", label: "Study" }, { key: "dataset", label: "Data", fmt: (_, v) => ds(v) }, { key: "family", label: "Question" },
    { key: "a", label: "a", fmt: short }, { key: "b", label: "b", fmt: short }, { key: "metric", label: "Metric", fmt: (m) => METRIC[m] || m },
    { key: "diff_main", label: "a − b, main", right: true, fmt: (x) => fmt(num(x), 3) }, { key: "diff_replicate", label: "replicate", right: true, fmt: (x) => fmt(num(x), 3) },
    { key: "verdict", label: "Verdict" }], all, "Show every verdict as a table");
  const listing = card("All verdicts", "Primary and secondary questions, as written by each study's runner.", `All datasets · ${state.setting.toUpperCase()}`);
  table.table.hidden = false;
  listing.body.append(table.table);

  panel.replaceChildren(takeaway(
    `Of ${rows.length} registered primary comparisons in this setting, ${b(`${n("replicated, a better")} favour`)} model a, ${n("practically equivalent")} are practically equivalent, ${n("replicated, a worse")} go against it and ${n("not replicated")} did not replicate.`,
    `Replicated: p_Holm < 0.05 in both seed sets, the same sign, and a difference of at least 0.05 pp; otherwise equivalent or not replicated.${state.setting === "e1" ? " D36 against gradient boosting with all data stored is “better” by the registered rule (it reads the mean), but 52 of 64 conditions favour gradient boosting; the study does not count it as a win." : ""}`),
  fp.node, tally.node, listing.node);
}
