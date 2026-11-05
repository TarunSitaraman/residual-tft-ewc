// Native analytical views for the Forecast accuracy, Continual learning and Reuse & uncertainty tabs, and the headline
// dot plot on the Overview. Data: results/dashboard/*.csv (in Git) and data/dashboard/fact_predictions.csv (local).
// Every view follows the filter bar; S1 and S6 are never pooled.
import { PATHS } from "../config.js";
import { el, fmt, groupInfo } from "./views.js";
import { parseCsv } from "./soh.js";
import { lineChart, dotRows, scatter, matrix, legend, tableView, seqFill, divFill, rampLegend, SEQ_GRADIENT, DIV_GRADIENT } from "./charts.js";

export const PROPOSED = "tft_residual_ewc_shrunk";
const SETTING_LABEL = { e1: "E1 (h = 1-3)", e2: "E2 (h = 1-6)" };
const PROTOCOL_LABEL = { P1: "Continual stream (P1)", P2: "Unseen usage condition (P2)" };
const ACCENT = "var(--c1)", NEUTRAL = "var(--neutral)";

// ---------- data ----------
const cache = {};
const num = (v) => (v === "" || v === undefined ? null : Number(v));
export function load(name, local = false) {
  const path = `${local ? PATHS.local : PATHS.data}${name}.csv`;
  cache[path] ||= fetch(path).then((r) => {
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.text();
  }).then(parseCsv);
  return cache[path];
}
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
function groupMean(rows, keyFn, valFn) {
  const acc = new Map();
  for (const r of rows) {
    const v = valFn(r);
    if (v === null || Number.isNaN(v)) continue;
    const k = keyFn(r);
    (acc.get(k) || acc.set(k, []).get(k)).push(v);
  }
  return new Map([...acc].map(([k, v]) => [k, mean(v)]));
}

// ---------- shared pieces ----------
/** The section's result in one sentence (the lede), with an optional note in the interface face. */
export function takeaway(html, note) {
  const p = el("p", { class: "finding" });
  p.innerHTML = html;   // trusted: composed from numbers and fixed text in this file, labels escaped by b()
  if (note) p.append(el("small", { text: note }));
  return p;
}
// Width of the panels column, set by main.js before each render. Cards record the width their body will have in the
// 12-column grid, so charts drawn before the card is attached still lay out at their final size.
let layoutW = 960;
export const setLayoutWidth = (w) => { if (w > 0) layoutW = w; };
const SPAN = { half: 6, wide: 7, narrow: 5 };
function bodyWidth(cls) {
  const k = innerWidth <= 1100 ? 12 : SPAN[cls.trim()] || 12;
  const col = (layoutW - 11 * 22) / 12;
  return Math.round(col * k + (k - 1) * 22 - (innerWidth <= 600 ? 30 : 46));
}
/**
 * A numbered figure: title, body, a tools row (legend, table toggle) and a caption that also states the scope.
 * `scope` is a short "S6 · P2 · E1" string; it is written into the caption as plain words.
 */
export function card(title, caption, scope, cls = "") {
  const body = el("div", { class: "fig-body", "data-w": bodyWidth(cls) });
  const foot = el("div", { class: "fig-tools", hidden: true });
  const cap = [caption, scope ? `${scope.replace(/ · /g, ", ")}.` : null].filter(Boolean).join(" ");
  const node = el("figure", { class: `fig ${cls}` }, el("div", { class: "fig-head" }, el("h2", { text: title })), body, foot,
    cap ? el("figcaption", { text: cap }) : null);
  return { node, body, foot, setFoot(...kids) { foot.hidden = !kids.length; foot.replaceChildren(...kids); } };
}
const notice = (text) => el("p", { class: "notice", text });
const empty = (text) => el("div", { class: "empty", text });
const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const b = (x) => `<b>${esc(x)}</b>`;

// ---------- Overview: headline dot plot ----------
const HEAD_ORDER = [PROPOSED, "linear_extrapolation", "gbm_shared", "gbm_last", "persistence"];
const HEAD_FAMILY = { [PROPOSED]: "TFT", linear_extrapolation: "Baseline", persistence: "Baseline", gbm_shared: "GBM", gbm_last: "GBM" };

export function renderHeadlineChart(root, summary, state, tip) {
  const rows = summary.headline.filter((r) => r.setting === SETTING_LABEL[state.setting] && !r.method.startsWith("vs "));
  const max = Math.max(...rows.map((r) => Math.max(r.rmse_main, r.rmse_replicate)));
  const panels = ["P1", "P2"].map((p) => {
    const host = el("div", {});
    const items = HEAD_ORDER.map((m) => rows.find((r) => r.method === m && r.protocol === PROTOCOL_LABEL[p])).filter(Boolean);
    const best = Math.min(...items.map((r) => (r.rmse_main + r.rmse_replicate) / 2));
    const panel = el("div", { class: "panel" }, el("h3", { text: PROTOCOL_LABEL[p] }), host);
    panel._draw = () => dotRows(host, {
      tip, x: { label: "Test RMSE (pp)", min: 0, max, fmt: (v) => v.toFixed(1) }, labelW: 200, rowH: 36,
      rows: items.map((r) => {
        const isP = r.method === PROPOSED;
        const dim = state.family !== "all" && HEAD_FAMILY[r.method] !== state.family;
        const col = dim ? "var(--neutral-soft)" : isP ? ACCENT : NEUTRAL;
        const isBest = Math.abs((r.rmse_main + r.rmse_replicate) / 2 - best) < 0.005;
        const t = `${b(r.label)}<br>${PROTOCOL_LABEL[p]}, ${SETTING_LABEL[state.setting]}`;
        return {
          label: `${r.method === "gbm_shared" && p === "P2" ? "Gradient boosting, all conditions stored" : r.label.replace(" (proposed)", "")}${isBest ? "  ●" : ""}`, emph: isP,
          range: isP ? [Math.min(r.rmse_main, r.rmse_replicate), Math.max(r.rmse_main, r.rmse_replicate)] : null,
          marks: isP
            ? [{ v: r.rmse_replicate, color: col, open: true, r: 7.5, tip: `${t}<br>Replicate seed set: ${fmt(r.rmse_replicate, 3)} pp (50 runs)` },
              { v: r.rmse_main, color: col, r: 5, tip: `${t}<br>Main seed set: ${fmt(r.rmse_main, 3)} pp (50 runs)` }]
            : [{ v: r.rmse_main, color: col, tip: `${t}<br>${fmt(r.rmse_main, 3)} pp (single deterministic fit)` }],
        };
      }),
    });
    return panel;
  });
  root.replaceChildren(el("div", { class: "panels2" }, panels),
    legend([{ label: "Proposed, main seed set", color: ACCENT, kind: "dot" }, { label: "Proposed, replicate seed set", color: ACCENT, kind: "ring" },
      { label: "Baseline, single fit", color: NEUTRAL, kind: "dot" }]),
    el("p", { class: "sub", text: "● after a name marks the lowest error in that panel." }));
  panels.forEach((p) => p._draw());
}

const VERDICT_PHRASE = { "practically equivalent": "level with", "replicated, a better": "ahead of", "replicated, a worse": "behind",
  "not replicated": "not distinguishable from" };

const STATUS = { "practically equivalent": ["level", "Practically equivalent"], "replicated, a better": ["better", "Better, replicated"],
  "replicated, a worse": ["worse", "Worse, replicated"], "not replicated": ["level", "Not distinguishable"] };

const signed = (d) => `${d > 0 ? "+" : "−"}${fmt(Math.abs(d))}`;
function kpi(k, v, unit, note, verdict) {
  const [cls, word] = STATUS[verdict] || ["level", verdict];
  return el("div", { class: "kpi" }, el("span", { class: "k", text: k }),
    el("span", { class: "v" }, v, el("small", { text: unit })), el("span", { class: "n", text: note }),
    verdict ? el("span", { class: `st ${cls}`, text: word }) : null);
}

/** Headline numbers: where the proposed model stands against the baselines that work without stored data, with the
 *  registered verdicts. The comparison with gradient boosting that stores every condition is stated in kpiNote. */
export function renderKpis(root, summary, state) {
  const rows = summary.headline.filter((r) => r.setting === SETTING_LABEL[state.setting]);
  const get = (m, p) => rows.find((r) => r.method === m && r.protocol === PROTOCOL_LABEL[p]);
  const p1 = get(PROPOSED, "P1"), p2 = get(PROPOSED, "P2"), lin1 = get("linear_extrapolation", "P1"), gbm1 = get("gbm_shared", "P1"),
    lin2 = get("linear_extrapolation", "P2");
  const v = (m, p) => get(`vs ${m}`, p) || {};
  const tiles = [
    p1 && lin1 && kpi("Continual stream, test error", fmt(p1.rmse_main), "pp",
      `Level with linear extrapolation (${fmt(lin1.rmse_main)} pp), the strongest baseline on the stream.`, v("linear_extrapolation", "P1").verdict),
    p1 && gbm1 && kpi("Against gradient boosting on the stream", signed(v("gbm_shared", "P1").diff_main), "pp",
      `${fmt(p1.rmse_main)} against ${fmt(gbm1.rmse_main)} pp test RMSE.`, v("gbm_shared", "P1").verdict),
    p2 && lin2 && kpi("Unseen usage condition, against extrapolation", signed(v("linear_extrapolation", "P2").diff_main), "pp",
      `${fmt(p2.rmse_main)} against ${fmt(lin2.rmse_main)} pp on conditions never seen in training.`, v("linear_extrapolation", "P2").verdict),
    kpi("Forgetting with EWC, horizons 1-6", "0.09", "pp", "Down from 0.28 / 0.22 pp without EWC (study D23).", "replicated, a better")];
  root.replaceChildren(...tiles.filter(Boolean));
}

/** One sentence that puts the strongest unseen-condition comparator in context. */
export function kpiNote(summary, state) {
  const rows = summary.headline.filter((r) => r.setting === SETTING_LABEL[state.setting] && r.protocol === PROTOCOL_LABEL.P2);
  const g = (m) => rows.find((r) => r.method === m);
  const all = g("gbm_shared"), last = g("gbm_last");
  if (!all || !last) return "";
  return `Gradient boosting that stores every earlier condition's data reaches ${fmt(all.rmse_main)} pp on unseen conditions; held to the same no-storage rule as the proposed model it reaches ${fmt(last.rmse_main)} pp (studies D26 and D32).`;
}

export function headlineFinding(summary, state) {
  const rows = summary.headline.filter((r) => r.setting === SETTING_LABEL[state.setting]);
  const get = (m, p) => rows.find((r) => r.method === m && r.protocol === PROTOCOL_LABEL[p]);
  const pp = (r) => (r ? `${fmt(r.rmse_main)} pp` : "–");
  const vs = (m, p) => VERDICT_PHRASE[get(`vs ${m}`, p)?.verdict] || "compared with";
  const h = state.setting === "e1" ? "1 to 3 checks ahead" : "1 to 6 checks ahead";
  return takeaway(`The residual TFT with EWC is the only model compared that has no weak side across the two settings (test RMSE of SOH, ${h}). ` +
    `On the continual stream it is ${vs("linear_extrapolation", "P1")} linear extrapolation (${b(pp(get(PROPOSED, "P1")))} against ${pp(get("linear_extrapolation", "P1"))}) and ${vs("gbm_shared", "P1")} gradient boosting (${pp(get("gbm_shared", "P1"))}). ` +
    `On unseen usage conditions it is ${vs("linear_extrapolation", "P2")} linear extrapolation (${b(pp(get(PROPOSED, "P2")))} against ${pp(get("linear_extrapolation", "P2"))}). ${kpiNote(summary, state)}`,
    "S6, main seed set. A verdict needs significance after Holm correction in both seed sets and a difference of at least 0.05 pp.");
}

// ---------- Forecast accuracy ----------
const HORIZON_SERIES = [   // fixed entity -> colour (never re-coloured by filters)
  ["tft_ewc", "var(--c1)", "TFT"], ["tft_joint", "var(--c3)", "TFT"], ["gbm_F1", "var(--c7)", "GBM"],
  ["ewc", "var(--c2)", "MLP"], ["linear_extrapolation", "var(--c4)", "Baseline"], ["persistence", NEUTRAL, "Baseline"]];

export async function renderAccuracy(panel, ctx) {
  const { state, summary, tip, onSelect } = ctx;
  const [horizon, d30, methods] = await Promise.all([load("fact_horizon"), load("fact_d30_group"), load("dim_method")]);
  const family = Object.fromEntries(methods.map((m) => [m.method, m.family]));
  const label = Object.fromEntries(methods.map((m) => [m.method, m.label]));
  const groups = summary.groups.filter((g) => g.dataset === state.dataset).map((g) => g.group);
  const keep = (m, fam = family[m]) => state.family === "all" || fam === state.family;
  const gl = (g) => { const i = groupInfo(summary, g); return state.dataset === "S6" ? `${i.name} (${i.desc})` : i.name; };

  // Per-group matrix: S6 from the headline studies (per protocol), S1 from D30 (leave-one-cell-out).
  let rows, cell, tag, sub;
  if (state.dataset === "S6") {
    const pg = summary.per_group.filter((r) => r.protocol === state.protocol && r.setting === state.setting);
    const ms = HEAD_ORDER.filter((m) => pg.some((r) => r.method === m) && keep(m, HEAD_FAMILY[m]));
    rows = ms.map((m) => ({ key: m, label: pg.find((r) => r.method === m).label, emph: m === PROPOSED }));
    cell = (r, c) => {
      const x = pg.find((q) => q.method === r.key && q.group === c.key);
      return x && { v: x.rmse_mean, tip: `${b(x.label)}<br>${esc(gl(c.key))}: ${fmt(x.rmse_mean, 3)} pp` + (x.n_runs > 1 ? ` ± ${fmt(x.rmse_sd, 3)} (s.d., ${x.n_runs} runs)` : " (single fit)") };
    };
    tag = `S6 · ${state.protocol} · ${state.setting.toUpperCase()}`;
    sub = state.protocol === "P1" ? "Final error on each task after the whole stream." : "Error on each usage condition when it is held out entirely.";
  } else {
    const rm = d30.filter((r) => r.dataset === "S1" && r.metric === "reg_RMSE");
    const ms = [...new Set(rm.map((r) => r.method))].filter((m) => keep(m));
    rows = ms.map((m) => ({ key: m, label: label[m] || m, emph: m === "tft" }));
    cell = (r, c) => {
      const x = rm.find((q) => q.method === r.key && q.group === c.key);
      return x && { v: num(x.value), tip: `${b(label[r.key] || r.key)}<br>${esc(gl(c.key))}: ${fmt(num(x.value), 3)} pp (D30)` };
    };
    tag = "S1 · D30 · leave one cell out";
    sub = "S1 (six NMC second-life cells): each cell held out in turn (study D30, h = 1-3).";
  }
  const vals = rows.flatMap((r) => groups.map((g) => cell(r, { key: g })?.v)).filter((v) => v > 0);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const t = (v) => (hi > lo ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : 0);

  const heat = card(state.dataset === "S6" && state.protocol === "P1" ? "Final error per task" : "Error per held-out group", sub, tag);
  if (rows.length) {
    matrix(heat.body, { rows, cols: groups.map((g) => { const i = groupInfo(summary, g); return { key: g, label: i.name, sub: state.dataset === "S6" ? i.desc : null }; }), cell, fill: (v) => seqFill(t(v)), best: "min", tip,
      selCol: state.group, onCell: (r, c) => onSelect({ group: c.key, method: r?.key }) });
    heat.setFoot(rampLegend(`${fmt(lo)} pp`, `${fmt(hi)} pp`, SEQ_GRADIENT, "log scale; ● lowest error per column; click a cell for details"));
  } else heat.body.append(empty(`No ${state.family} method in this view.`));

  // Skill against persistence, averaged over groups.
  const skill = card("Skill against persistence", "1 − RMSE / persistence RMSE, mean over groups. Positive means better than carrying the last SOH forward.", tag, "half");
  const base = (g) => cell({ key: "persistence" }, { key: g })?.v;
  const skills = rows.filter((r) => r.key !== "persistence").map((r) => {
    const per = groups.map((g) => { const v = cell(r, { key: g })?.v, p = base(g); return v && p ? 1 - v / p : null; }).filter((v) => v !== null);
    return { ...r, v: mean(per), lo: Math.min(...per), hi: Math.max(...per), n: per.length };
  }).filter((r) => r.v !== null).sort((a, b2) => b2.v - a.v);
  if (skills.length) {
    dotRows(skill.body, { tip, x: { label: "Skill vs persistence (mean over groups)", min: Math.min(0, ...skills.map((r) => r.lo)), max: 1, fmt: (v) => v.toFixed(1) },
      refs: [{ x: 0, label: "persistence" }], labelW: 190,
      rows: skills.map((r) => ({ label: r.label.replace(" (proposed)", ""), emph: r.emph, range: [r.lo, r.hi],
        marks: [{ v: r.v, color: r.emph ? ACCENT : NEUTRAL, tip: `${b(r.label)}<br>Skill ${fmt(r.v)} (range ${fmt(r.lo)} to ${fmt(r.hi)} over ${r.n} groups)` }] })) });
    skill.setFoot(el("span", { class: "sub", text: "Dot: mean skill. Line: range over groups." }));
  } else skill.body.append(empty("Persistence is not in this view."));

  // Error by horizon (D30).
  const hz = card("Error by forecast horizon", "Pooled test RMSE at each horizon, leave-one-group-out study D30 (h = 1-3).", state.dataset, "half");
  const hrows = horizon.filter((r) => r.dataset === state.dataset);
  const series = HORIZON_SERIES.filter(([m, , fam]) => keep(m, fam) && hrows.some((r) => r.method === m)).map(([m, color]) => ({
    key: m, color, emph: m === "tft_ewc", label: hrows.find((r) => r.method === m).label,
    points: hrows.filter((r) => r.method === m).map((r) => ({ x: num(r.horizon), y: num(r.rmse_pp) })) }));
  if (series.length) {
    lineChart(hz.body, { series, tip, x: { label: "Horizon (capacity checks ahead)", ticks: [1, 2, 3], min: 1, max: 3 },
      y: { label: "RMSE (pp)", min: 0, fmt: (v) => v.toFixed(1), tipFmt: (v) => `${v.toFixed(3)} pp` } });
    if (series.length > 4) hz.body.prepend(legend(series.map((sr) => ({ label: sr.label, color: sr.color }))));
    const tv = tableView([{ key: "label", label: "Method" }, { key: "horizon", label: "h", right: true }, { key: "rmse_pp", label: "RMSE (pp)", right: true, fmt: (v) => fmt(num(v), 3) }],
      hrows.filter((r) => series.some((sr) => sr.key === r.method)));
    hz.setFoot(el("span", { class: "sub", text: "Six of 14 methods shown; the table has the rest of this selection." }), tv.btn);
    hz.body.append(tv.table);
  } else hz.body.append(empty(`No ${state.family} method in the horizon study.`));

  const best = skills[0];
  const colMean = groups.map((g) => ({ g, v: mean(rows.map((r) => cell(r, { key: g })?.v).filter((v) => v > 0)) })).filter((c) => c.v);
  const hard = colMean.reduce((a, c) => (c.v > a.v ? c : a), colMean[0] || {}), easy = colMean.reduce((a, c) => (c.v < a.v ? c : a), colMean[0] || {});
  const spread = hard.g ? `Averaged over the methods shown, the hardest group (${esc(groupInfo(summary, hard.g).name)}, ${fmt(hard.v)} pp) has ${fmt(hard.v / easy.v, 1)} times the error of the easiest (${esc(groupInfo(summary, easy.g).name)}, ${fmt(easy.v)} pp).` : "";
  const find = state.dataset === "S6"
    ? takeaway(`Across the seven LFP usage conditions, ${b(best ? best.label.replace(" (proposed)", "") : "–")} has the highest skill against persistence (${best ? fmt(best.v) : "–"}). ${spread}`,
      `Colour runs on a log scale because errors span ${fmt(hi / lo, 0)} times from best to worst.`)
    : takeaway(`On the six NMC second-life cells, per-cell errors run from ${fmt(lo)} to ${fmt(hi)} pp across all methods; ${b(best ? best.label : "–")} leads on skill against persistence (${best ? fmt(best.v) : "–"}). ${spread}`,
      "S1 results come from the leave-one-cell-out study D30, not from the continual stream.");
  panel.replaceChildren(find, heat.node, skill.node, hz.node);
}

// ---------- Continual learning ----------
const STRATEGY = { residual_ewc: ["Residual + EWC", "var(--c1)"], residual_finetune: ["Residual, fine-tuning", "var(--c2)"],
  direct_ewc: ["Direct + EWC", "var(--c3)"], residual_saewc: ["Residual + SA-EWC", "var(--c4)"] };
const FAMILY_PREFIX = { TFT: "tft", MLP: "mlp", BiLSTM: "lstm", TCN: "tcn", Transformer: "transformer" };

export async function renderContinual(panel, ctx) {
  const { state, summary, tip, onSelect } = ctx;
  if (state.dataset !== "S6") {
    panel.replaceChildren(takeaway("The continual stream (P1) is an S6 experiment: seven LFP usage conditions arrive one after another.", "Switch the dataset to S6 to see adaptation, retention and forgetting."));
    return;
  }
  const [task, run, base, methods] = await Promise.all([load("fact_stream_task"), load("fact_stream_run"), load("fact_stream_baseline"), load("dim_method")]);
  const label = Object.fromEntries(methods.map((m) => [m.method, m.label]));
  const famOf = Object.fromEntries(methods.map((m) => [m.method, m.family]));
  const tag = `S6 · P1 · ${state.setting.toUpperCase()} · main seeds`;
  const d24 = task.filter((r) => r.experiment === "D24" && r.setting === state.setting && r.seed_set === "main");

  // Adaptation vs retention.
  const ar = card("Adaptation against retention", "Each point is one model: mean error on a task right after learning it (adaptation) against the error on that task after the whole stream (retention). Points above the line forgot.", tag, "wide");
  const diag = groupMean(d24, (r) => r.method, (r) => num(r.diagonal_error));
  const fin = groupMean(d24, (r) => r.method, (r) => num(r.final_error));
  const d25 = task.filter((r) => r.experiment === "D25" && r.setting === state.setting && r.seed_set === "main" && r.method === PROPOSED);
  const pts = [...diag.keys()].map((m) => ({ m, x: diag.get(m), y: fin.get(m) }));
  if (d25.length) pts.push({ m: PROPOSED, x: mean(d25.map((r) => num(r.diagonal_error))), y: mean(d25.map((r) => num(r.final_error))) });
  scatter(ar.body, { tip, diag: "no forgetting", x: { label: "Adaptation: error on the newest task (pp)", fmt: (v) => v.toFixed(2) },
    y: { label: "Retention: final error (pp)", fmt: (v) => v.toFixed(2) },
    points: pts.map(({ m, x, y }) => {
      const isP = m === PROPOSED, inFam = state.family === "all" || famOf[m] === state.family;
      const color = isP || m === "tft_residual_ewc" ? ACCENT : inFam ? NEUTRAL : "var(--neutral-soft)";
      return { x, y, color, open: m.includes("finetune"), r: isP ? 7 : 5.5, label: isP ? "proposed" : m === "tft_residual_ewc" ? "residual TFT + EWC" : null,
        tip: `${b(label[m] || m)}<br>Adaptation ${fmt(x, 3)} pp · retention ${fmt(y, 3)} pp<br>Forgetting ${fmt(y - x, 3)} pp` };
    }) });
  ar.setFoot(legend([{ label: "Residual TFT", color: ACCENT, kind: "dot" }, { label: "Other models, EWC or SA-EWC", color: NEUTRAL, kind: "dot" },
    { label: "Fine-tuning (no protection)", color: NEUTRAL, kind: "ring" }]));

  // Forgetting by arrival position for one family.
  const fam = FAMILY_PREFIX[state.family] ? state.family : "TFT";
  const pre = FAMILY_PREFIX[fam];
  const fg = card("Forgetting by arrival order", `${fam}: how much the error on a task grows by the end of the stream, by when the task arrived. Later tasks have less time to be forgotten.`, `${fam} · D24 · ${state.setting.toUpperCase()}`, "narrow");
  const series = Object.entries(STRATEGY).map(([k, [lab, color]]) => {
    const rows = d24.filter((r) => r.method === `${pre}_${k}`);
    const by = groupMean(rows, (r) => num(r.arrival_pos), (r) => num(r.forgetting));
    return { key: k, label: lab, color, emph: k === "residual_ewc", points: [...by].filter(([x]) => x < 7).map(([x, y]) => ({ x, y })) };
  }).filter((sr) => sr.points.length);
  if (series.length) lineChart(fg.body, { series, tip, x: { label: "Arrival position of the task", min: 1, max: 6, ticks: [1, 2, 3, 4, 5, 6] },
    y: { label: "Forgetting (pp)", min: 0, fmt: (v) => v.toFixed(2), tipFmt: (v) => `${v.toFixed(3)} pp` } });
  else fg.body.append(empty(`${fam} was not run on the stream in this setting.`));
  if (!FAMILY_PREFIX[state.family] && state.family !== "all") fg.setFoot(el("span", { class: "sub", text: `${state.family} has no continual variants; showing TFT.` }));

  // What EWC changes: forgetting with and without EWC (D23), for the residual and direct targets.
  const d23 = run.filter((r) => r.experiment === "D23" && r.setting === state.setting);
  const ew = card("What EWC changes", "Mean forgetting per run with EWC (filled) and with plain fine-tuning (ring), for the residual target and for forecasting SOH directly. Main seed set; the replicate is in the tooltip.", `S6 · P1 · D23 · ${state.setting.toUpperCase()}`);
  const fgMean = (m, set) => mean(d23.filter((r) => r.method === m && r.seed_set === set).map((r) => num(r.forgetting)).filter((v) => v !== null));
  const ewRows = [["tft", "residual", "TFT, residual"], ["tft", "direct", "TFT, direct"], ["mlp", "residual", "MLP, residual"], ["mlp", "direct", "MLP, direct"]]
    .filter(([f]) => state.family === "all" || FAMILY_PREFIX[state.family] === f);
  const ewcGain = {};
  if (!d23.length || !ewRows.length) ew.body.append(empty(d23.length ? `${state.family} was not part of the EWC study (D23); it compared TFT and MLP.` : "No D23 runs in this setting."));
  else {
    dotRows(ew.body, { tip, labelW: 170, x: { label: "Forgetting (pp)", min: 0, fmt: (v) => v.toFixed(2) },
      rows: ewRows.map(([f, t, lab]) => {
        const isP = f === "tft" && t === "residual";
        const mk = (strat) => {
          const m = `${f}_${t}_${strat}`, v = fgMean(m, "main"), vr = fgMean(m, "replicate");
          return v === null ? null : { v, color: isP ? ACCENT : NEUTRAL, open: strat === "finetune", r: 6,
            tip: `${b(label[m] || m)}<br>Forgetting ${fmt(v, 3)} pp (main), ${fmt(vr, 3)} pp (replicate)` };
        };
        const e = mk("ewc"), ft = mk("finetune");
        if (e && ft) ewcGain[`${f}_${t}`] = [e.v, ft.v];
        return { label: lab, emph: isP, range: e && ft ? [Math.min(e.v, ft.v), Math.max(e.v, ft.v)] : null, marks: [ft, e].filter(Boolean) };
      }) });
    ew.setFoot(legend([{ label: "With EWC", color: NEUTRAL, kind: "dot" }, { label: "Fine-tuning, no protection", color: NEUTRAL, kind: "ring" }]));
  }

  // Seed spread of the D25 finalists against the baselines.
  const d25r = run.filter((r) => r.experiment === "D25" && r.setting === state.setting);
  const nRuns = d25r.filter((r) => r.method === PROPOSED && r.seed_set === "main").length;
  const sp = card("Run-to-run spread", `Final average error of every run (${nRuns} runs per seed set: seeds × task orders). Vertical lines: deterministic baselines.`, `S6 · P1 · D25 · ${state.setting.toUpperCase()}`);
  const order = [PROPOSED, "tft_residual_ewc", "tft_pre_residual_ewc_shrunk", "tft_pre_residual_ewc"].filter((m) => d25r.some((r) => r.method === m));
  const bl = base.filter((r) => r.experiment === "D24" && r.setting === state.setting && r.group === "all");
  const ref = (m, lab) => { const r = bl.find((q) => q.method === m); return r ? [{ x: num(r.final_error), label: lab }] : []; };
  let k = 0;
  dotRows(sp.body, { tip, rowH: 46, labelW: 230, x: { label: "Final average error (pp)", min: 0, fmt: (v) => v.toFixed(1) },
    refs: [...ref("linear_extrapolation", "linear extrap."), ...ref("gbm_shared", "gradient boosting")],
    rows: order.map((m) => ({ label: label[m] || m, emph: m === PROPOSED,
      marks: d25r.filter((r) => r.method === m).map((r) => ({ v: num(r.final_avg_error), color: m === PROPOSED ? ACCENT : NEUTRAL, open: r.seed_set === "replicate", r: 4,
        jitter: ((k++ * 0.618) % 1) - 0.5, tip: `${b(label[m] || m)}<br>${r.seed_set} seed ${r.seed}, order ${r.order_id}: ${fmt(num(r.final_avg_error), 3)} pp` })) })) });
  sp.setFoot(legend([{ label: "Main seed set", color: NEUTRAL, kind: "dot" }, { label: "Replicate seed set", color: NEUTRAL, kind: "ring" }]));

  // Backward transfer: group x arrival position for the residual TFT + EWC.
  const bt = card("Backward transfer by task and arrival position", "Residual TFT + EWC: change in a task's error between learning it and the end of the stream. Red means the error grew (forgetting), blue means later learning helped.", `S6 · P1 · D24 · ${state.setting.toUpperCase()}`);
  const btr = d24.filter((r) => r.method === "tft_residual_ewc");
  const groups = summary.groups.filter((g) => g.dataset === "S6").map((g) => g.group);
  const cellMap = groupMean(btr, (r) => `${r.group}|${r.arrival_pos}`, (r) => num(r.backward_transfer));
  const lim = Math.max(...[...cellMap.values()].map(Math.abs), 0.01);
  const nm = (g) => { const i = groupInfo(summary, g); return `${i.short}: ${i.desc}`; };
  matrix(bt.body, { tip, corner: "Task (usage condition)", rows: groups.map((g) => ({ key: g, label: nm(g), emph: g === state.group })),
    cols: [1, 2, 3, 4, 5, 6].map((p) => ({ key: p, label: `Position ${p}` })),
    cell: (r, c) => { const v = cellMap.get(`${r.key}|${c.key}`); return v === undefined ? null : { v, tip: `${b(r.label)} arriving at position ${c.key}<br>Error change ${v > 0 ? "+" : ""}${fmt(v, 3)} pp` }; },
    fill: (v) => divFill(v / lim), fmtV: (v) => (v > 0 ? "+" : "") + v.toFixed(2),
    onCell: (r) => r && onSelect({ group: r.key }) });
  bt.setFoot(rampLegend(`−${fmt(lim)} pp`, `+${fmt(lim)} pp`, DIV_GRADIENT, "grey is no change; a task arriving last has nothing after it"));

  const pP = pts.find((p) => p.m === PROPOSED);
  const head = summary.headline.filter((r) => r.setting === SETTING_LABEL[state.setting] && r.protocol === PROTOCOL_LABEL.P1);
  const hP = head.find((r) => r.method === PROPOSED), hL = head.find((r) => r.method === "linear_extrapolation");
  const verdict = VERDICT_PHRASE[head.find((r) => r.method === "vs linear_extrapolation")?.verdict] || "compared with";
  const g = ewcGain.tft_residual;
  const ewcLine = !g ? "" : g[1] - g[0] > 0.05
    ? ` Without EWC the same model forgets ${b(`${fmt(g[1], 2)} pp`)} instead of ${fmt(g[0], 2)} pp (D23).`
    : ` Over these horizons EWC makes little difference to its forgetting (${fmt(g[0], 2)} against ${fmt(g[1], 2)} pp without it, D23); the gain appears over horizons 1-6.`;
  panel.replaceChildren(takeaway(
    `With the residual target and EWC, the proposed TFT's error on a task grows by ${b(`${fmt(pP ? pP.y - pP.x : null, 3)} pp`)} on average between learning it and the end of the stream, and its final error is ${verdict} linear extrapolation (${fmt(hP?.rmse_main)} against ${fmt(hL?.rmse_main)} pp).${ewcLine}`,
    "Adaptation and retention are reported together because a model can look accurate on new data while forgetting the old."),
  ar.node, fg.node, ew.node, sp.node, bt.node);
}

// ---------- Reuse & uncertainty ----------
const CLASSES = ["C0 stable", "C1 moderate fade", "C2 substantial fade", "C3 severe fade"];
const PRED_METHOD = { all: "tft", TFT: "tft", MLP: "mlp", GBM: "gbm_F1", Baseline: "linear_extrapolation" };
const RULES = [["gbm_loco", "var(--c1)"], ["lin_all", "var(--c2)"], ["lin_val", "var(--c3)"]];
// D28 arms; the static arm is the D27 GBM-LOCO rule and keeps its colour.
const ARMS = [["d28_static", "var(--c1)"], ["d28_track", "var(--c4)"], ["d28_refit_static", "var(--c7)"], ["d28_refit_track", "var(--c5)"]];

export async function renderReuse(panel, ctx) {
  const { state, tip, onSelect, summary } = ctx;
  const [cov, methods] = await Promise.all([load("fact_coverage"), load("dim_method")]);
  const label = Object.fromEntries(methods.map((m) => [m.method, m.label]));
  let preds = null;
  try { preds = await load("fact_predictions", true); } catch { /* local, git-ignored input */ }
  const pm = PRED_METHOD[state.family];
  const kids = [];

  // Confusion and parity from the D30 forecasts (local data folder).
  const conf = card("Reuse-class confusion", "True reuse class of each forecast target against the class of the forecast. Shading: share of the true class (row).", pm ? `${state.dataset} · D30 · ${label[pm] || pm}` : null, "narrow");
  const par = card("Forecast against measured SOH", "Every saved forecast of the first seed. Points on the line are exact.", pm ? `${state.dataset} · D30 · ${label[pm] || pm}` : null, "wide");
  let acc = null, within = null;
  if (!preds) {
    const msg = "Built from the local data folder: run python scripts/export_dashboard_data.py.";
    conf.body.append(empty(msg)); par.body.append(empty(msg));
  } else if (!pm) {
    conf.body.append(empty(`No ${state.family} model in the D30 forecasts.`)); par.body.append(empty(`No ${state.family} model in the D30 forecasts.`));
  } else {
    const rows = preds.filter((r) => r.dataset === state.dataset && r.method === pm);
    const seed0 = rows.filter((r) => r.seed === rows[0]?.seed);
    const count = new Map();
    for (const r of seed0) count.set(`${r.class_true}|${r.class_pred}`, (count.get(`${r.class_true}|${r.class_pred}`) || 0) + 1);
    const rowTot = Object.fromEntries(CLASSES.map((c) => [c, CLASSES.reduce((a, p) => a + (count.get(`${c}|${p}`) || 0), 0)]));
    acc = seed0.length ? seed0.filter((r) => r.class_true === r.class_pred).length / seed0.length : null;
    within = seed0.length ? seed0.filter((r) => Math.abs(num(r.class_distance)) <= 1).length / seed0.length : null;
    matrix(conf.body, { tip, corner: "True \\ forecast", rows: CLASSES.map((c) => ({ key: c, label: c.slice(0, 2) })), cols: CLASSES.map((c) => ({ key: c, label: c.slice(0, 2) })),
      cell: (r, c) => { const n = count.get(`${r.key}|${c.key}`) || 0; return rowTot[r.key] ? { v: n, tip: `True ${b(r.key)}, forecast ${c.key}<br>${n} of ${rowTot[r.key]} (${fmt((100 * n) / rowTot[r.key], 1)} %)`, share: n / rowTot[r.key] } : null; },
      fill: (v) => v, fmtV: (v) => String(v) });
    // Colour by row share (the table shows counts).
    conf.body.querySelectorAll("tbody tr").forEach((tr, i) => tr.querySelectorAll("td.cell:not(.na)").forEach((td, j) => {
      const n = count.get(`${CLASSES[i]}|${CLASSES[j]}`) || 0, f = seqFill(n / (rowTot[CLASSES[i]] || 1));
      td.style.setProperty("--bg", f.bg); td.style.setProperty("--fg", f.fg);
    }));
    conf.setFoot(el("span", { class: "sub", text: `Exact class ${fmt(100 * acc, 1)} %, within one class ${fmt(100 * within, 1)} % (${seed0.length} forecasts). C0 > 95 %, C1 90-95 %, C2 80-90 %, C3 ≤ 80 % SOH.` }));
    scatter(par.body, { tip, diag: true, square: true,
      x: { label: "Measured SOH (%)", min: Math.floor(Math.min(...seed0.map((r) => num(r.soh_target))) / 10) * 10, max: 100, fmt: (v) => v.toFixed(0) }, y: { label: "Forecast SOH (%)", fmt: (v) => v.toFixed(0) },
      points: seed0.map((r) => ({ x: num(r.soh_target), y: num(r.soh_pred), color: state.group === "all" || r.group === state.group ? ACCENT : "var(--neutral-soft)" }))
        .sort((a) => (a.color === ACCENT ? 1 : -1)) });
    const rmse = Math.sqrt(mean(seed0.map((r) => num(r.error_pp) ** 2)));
    par.setFoot(el("span", { class: "sub", text: `${seed0.length} forecasts, RMSE ${fmt(rmse, 3)} pp.${state.group !== "all" ? ` ${groupInfo(summary, state.group).name} highlighted.` : " Pick a group to highlight it."}` }));
  }

  // Interval coverage per group (S6, D27).
  const cv = card("Coverage of 90 % prediction intervals", "Share of measured SOH values inside the interval, per held-out usage condition. Target 0.90.", `S6 · P2 · D27 · ${state.setting.toUpperCase()}`);
  if (state.dataset !== "S6") cv.body.append(empty("The interval study (D27) is on S6."));
  else {
    const c9 = cov.filter((r) => r.setting === state.setting && num(r.level) === 0.9);
    const groups = summary.groups.filter((g) => g.dataset === "S6").map((g) => g.group);
    const rules = RULES.filter(([m]) => state.family === "all" || (state.family === "GBM") === m.startsWith("gbm"));
    const nm = (g) => { const i = groupInfo(summary, g); return `${i.short}: ${i.desc}`; };
    dotRows(cv.body, { tip, labelW: 290, x: { label: "Coverage", min: 0.4, max: 1, fmt: (v) => v.toFixed(1) }, refs: [{ x: 0.9, label: "target 0.90" }],
      onRow: (r) => r.key !== "all" && onSelect({ group: r.key }),
      rows: ["all", ...groups].map((g) => ({ key: g, label: g === "all" ? "All conditions" : nm(g), emph: g === state.group || (g === "all" && state.group === "all"),
        marks: rules.map(([m, color]) => { const r = c9.find((q) => q.method === m && q.group === g); return r && { v: num(r.coverage), color,
          tip: `${b(label[m] || m)}<br>${esc(g === "all" ? "All conditions" : nm(g))}: coverage ${fmt(num(r.coverage), 3)}${r.width ? `, mean width ${fmt(num(r.width))} pp` : ""}` }; }).filter(Boolean) })) });
    cv.setFoot(legend(rules.map(([m, color]) => ({ label: label[m] || m, color, kind: "dot" }))));
  }

  // Updating the intervals as the new condition's checks arrive (D28, gradient boosting only).
  const tag28 = `S6 · P2 · D28 · ${state.setting.toUpperCase()}`;
  const up = card("Updating the intervals as checks arrive", "Coverage of 90 % intervals per held-out condition when the half-width tracks the errors seen so far, the forecaster is refitted with the new condition's measured checks, or both.", tag28, "wide");
  const pool = card("Pooled 90 % intervals", "All seven held-out conditions together. Interval score: width plus a penalty for each miss, lower is better.", tag28, "narrow");
  const d28 = cov.filter((r) => r.experiment === "D28" && r.setting === state.setting && num(r.level) === 0.9);
  let isGain = null;
  const noD28 = state.dataset !== "S6" ? "The interval study (D28) is on S6."
    : !["all", "GBM"].includes(state.family) ? "D28 updates gradient-boosting intervals only; set the family to All or GBM."
    : !d28.length ? "No D28 results in this setting." : null;
  if (noD28) { up.body.append(empty(noD28)); pool.body.append(empty(noD28)); }
  else {
    const groups = summary.groups.filter((g) => g.dataset === "S6").map((g) => g.group);
    const nm = (g) => { if (g === "all") return "All conditions"; const i = groupInfo(summary, g); return `${i.short}: ${i.desc}`; };
    const get = (m, g) => d28.find((q) => q.method === m && q.group === g);
    dotRows(up.body, { tip, labelW: 290, x: { label: "Coverage", min: 0.4, max: 1, fmt: (v) => v.toFixed(1) }, refs: [{ x: 0.9, label: "target 0.90" }],
      onRow: (r) => r.key !== "all" && onSelect({ group: r.key }),
      rows: ["all", ...groups].map((g) => ({ key: g, label: nm(g), emph: g === state.group || (g === "all" && state.group === "all"),
        marks: ARMS.map(([m, color]) => {
          const r = get(m, g);
          return r && { v: num(r.coverage), color, r: m === "d28_refit_track" ? 6.5 : 5,
            tip: `${b(label[m] || m)}<br>${esc(nm(g))}: coverage ${fmt(num(r.coverage), 3)}, interval score ${fmt(num(r.interval_score), 2)} pp` };
        }).filter(Boolean) })) });
    up.setFoot(legend(ARMS.map(([m, color]) => ({ label: label[m] || m, color, kind: "dot" }))));
    const st = get("d28_static", "all"), rt = get("d28_refit_track", "all");
    if (st && rt) isGain = 1 - num(rt.interval_score) / num(st.interval_score);
    const head = ["Arm", "Coverage", "Score (pp)", "Width (pp)"].map((h, i) => el("th", { class: i ? "num" : "", scope: "col", text: h }));
    const body = ARMS.map(([m]) => get(m, "all") && el("tr", {},
      el("th", { scope: "row", text: label[m] || m }),
      ...["coverage", "interval_score", "width"].map((k, i) => el("td", { class: "num", text: fmt(num(get(m, "all")[k]), i ? 2 : 3) })))).filter(Boolean);
    pool.body.append(el("div", { class: "scroll" }, el("table", { class: "stbl" }, el("thead", {}, el("tr", {}, ...head)), el("tbody", {}, ...body))));
    if (isGain !== null) pool.setFoot(el("span", { class: "sub", text: `Refit + tracking lowers the interval score by ${fmt(100 * isGain, 0)} % against the static intervals, with pooled coverage kept near 0.90. Condition 7 (100 % depth of discharge) stays below 0.80.` }));
  }
  const d28Line = isGain !== null ? ` Updating them as the new condition's checks arrive lowers the interval score by ${b(`${fmt(100 * isGain, 0)} %`)} (D28).` : "";

  kids.push(takeaway(
    acc !== null
      ? `${esc(label[pm] || pm)} forecasts (D30) land in the right reuse class ${b(`${fmt(100 * acc, 0)} %`)} of the time and within one class ${b(`${fmt(100 * within, 0)} %`)}. Leave-one-condition-out conformal intervals around gradient boosting hold their 90 % coverage overall; the D26 rule (LIN-val) under-covers.${d28Line}`
      : `Leave-one-condition-out conformal intervals around gradient boosting hold their 90 % coverage overall; the D26 rule (LIN-val) under-covers.${d28Line}`,
    "Reuse classes describe capacity fade since the first check, not suitability for a particular application. No interval rule reaches 0.80 coverage on condition 7, the one condition outside the training range."),
  conf.node, par.node, cv.node, up.node, pool.node);
  panel.replaceChildren(...kids);
}
