// Native views rendered from results/dashboard/summary.json: KPI strip, headline table, per-group heatmap, details.
import { PATHS } from "../config.js";

const PROPOSED = "tft_residual_ewc_shrunk";
const METHOD_ORDER = [PROPOSED, "linear_extrapolation", "gbm_shared", "gbm_last", "persistence"];
const FAMILY = { [PROPOSED]: "TFT", linear_extrapolation: "Baseline", persistence: "Baseline", gbm_shared: "GBM", gbm_last: "GBM" };
const SETTING_LABEL = { e1: "E1 (h = 1-3)", e2: "E2 (h = 1-6)" };
const PROTOCOL_LABEL = { P1: "Continual stream (P1)", P2: "Unseen usage condition (P2)" };

const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v === true ? "" : v);
  }
  n.append(...kids.flat().filter((k) => k !== null && k !== undefined));
  return n;
};
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const signed = (x) => (x === null || x === undefined ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(2));

export { el, fmt };

/**
 * Readable name of a battery group. S6 conditions are numbered in dataset order and described by what differs
 * between them (charge mode, voltage, rate, and discharge or depth of discharge where they differ from the rest).
 */
export function groupInfo(summary, group) {
  const g = summary?.groups.find((x) => x.group === group);
  if (!g) return { name: String(group), short: String(group), desc: "" };
  if (g.dataset !== "S6") return { name: `Cell ${g.group}`, short: g.group, desc: "NMC second-life cell" };
  const n = Number(g.group_order) + 1;
  let desc = `${g.charge_mode} ${g.v_max} V, ${Number(Number(g.charge_c_rate).toFixed(2))}C`;
  if (g.discharge !== "WLTP") desc += `, ${g.discharge} discharge`;
  if (Number(g.dod) !== 0.7) desc += `, ${Math.round(Number(g.dod) * 100)} % DoD`;
  return { name: `Condition ${n}`, short: `Cond. ${n}`, desc };
}

export function renderHeadline(root, sub, summary, state) {
  const setting = SETTING_LABEL[state.setting];
  const rows = summary.headline.filter((r) => r.setting === setting);
  sub.textContent = `S6 · ${setting}`;
  if (state.dataset !== "S6") {
    root.replaceChildren(el("p", { class: "notice", text: "The headline comparison is on S6. S1 per-cell results are in the Forecast accuracy view (D30)." }));
    return;
  }
  const value = (method, proto) => rows.find((r) => r.method === method && r.protocol === PROTOCOL_LABEL[proto]);
  const best = {};
  for (const p of ["P1", "P2"]) {
    const vals = METHOD_ORDER.map((m) => value(m, p)).filter(Boolean).map((r) => (r.rmse_main + r.rmse_replicate) / 2);
    best[p] = Math.min(...vals);
  }
  const body = [];
  for (const m of METHOD_ORDER) {
    const r1 = value(m, "P1"), r2 = value(m, "P2");
    if (!r1 && !r2) continue;
    if (state.family !== "all" && FAMILY[m] !== state.family) continue;
    const cell = (r, p) => {
      if (!r) return el("td", { class: "r", text: "–" });
      const mean = (r.rmse_main + r.rmse_replicate) / 2;
      const txt = m === PROPOSED ? `${fmt(r.rmse_main)} / ${fmt(r.rmse_replicate)}` : fmt(r.rmse_main);
      return el("td", { class: "r" + (Math.abs(mean - best[p]) < 0.005 ? " best" : "") }, Math.abs(mean - best[p]) < 0.005 ? "● " : "", txt);
    };
    body.push(el("tr", { class: m === PROPOSED ? "proposed" : null },
      el("td", { text: (r1 || r2).label }), cell(r1, "P1"), cell(r2, "P2")));
  }
  const tests = rows.filter((r) => r.method.startsWith("vs "));
  if (tests.length) body.push(el("tr", { class: "sep" }, el("td", { colspan: 3, text: "Proposed minus baseline (main / replicate), verdict" })));
  for (const t of ["vs linear_extrapolation", "vs gbm_shared", "vs gbm_last"]) {
    const a = tests.find((r) => r.method === t && r.protocol === PROTOCOL_LABEL.P1);
    const b = tests.find((r) => r.method === t && r.protocol === PROTOCOL_LABEL.P2);
    if (!a && !b) continue;
    const cell = (r) => r ? el("td", { class: "r" }, `${signed(r.diff_main)} / ${signed(r.diff_replicate)}`,
      el("span", { class: `verdict ${shortVerdict(r.verdict).replace(".", "")}`, title: !r.post_hoc ? "Registered test" : r.method === "vs gbm_last" ? "Post hoc comparison (D35, not pre-registered)" : "Post hoc for this model (registered for the unshrunk model, same verdict)" },
        shortVerdict(r.verdict) + (r.post_hoc ? "*" : ""))) : el("td", { class: "r", text: "–" });
    body.push(el("tr", {}, el("td", { text: (a || b).label.replace("Proposed vs ", "vs ") }), cell(a), cell(b)));
  }
  root.replaceChildren(el("table", {},
    el("thead", {}, el("tr", {}, el("th", { text: "Method" }), el("th", { class: "r", text: "Continual stream (P1)" }),
      el("th", { class: "r", text: "Unseen condition (P2)" }))),
    el("tbody", {}, body)),
  el("p", { class: "sub", text: "● lowest error in the column. Verdicts: better/worse = significant after Holm in both seed sets with the same sign and at least 0.05 pp; equiv. = below 0.05 pp in both; n.r. = not replicated. * post hoc." }));
}

function shortVerdict(v) {
  return { "replicated, a better": "better", "replicated, a worse": "worse", "practically equivalent": "equiv.", "not replicated": "n.r." }[v] || v;
}

export function renderDetails(root, titleEl, summary, state, selection) {
  const g = summary.groups.find((x) => x.group === selection.group);
  titleEl.textContent = groupInfo(summary, selection.group).name;
  const kids = [];
  if (g) {
    kids.push(el("dl", {},
      el("dt", { text: "Dataset" }), el("dd", { text: `${g.dataset} · ${g.chemistry}` }),
      el("dt", { text: "Unit" }), el("dd", { text: g.unit }),
      el("dt", { text: "Condition" }), el("dd", { text: g.description })));
  }
  if (selection.fields) {
    kids.push(el("h3", { text: "Selection" }), el("dl", {}, Object.entries(selection.fields).slice(0, 12)
      .flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: String(v) })])));
  }
  for (const proto of ["P1", "P2"]) {
    const rows = summary.per_group.filter((r) => r.group === selection.group && r.setting === state.setting && r.protocol === proto)
      .sort((a, b) => a.rmse_mean - b.rmse_mean);
    if (!rows.length) continue;
    kids.push(el("h3", { text: `${PROTOCOL_LABEL[proto]}, ${SETTING_LABEL[state.setting]}` }),
      el("table", {}, el("tbody", {}, rows.map((r) => el("tr", { class: r.method === PROPOSED ? "proposed" : null },
        el("td", { class: "prose", text: r.label }),
        el("td", { class: "r", text: fmt(r.rmse_mean) + (r.n_runs > 1 ? ` ± ${fmt(r.rmse_sd)}` : "") }))))),
      el("p", { class: "sub", text: "RMSE in pp, sorted best first; ± is the s.d. over runs of the main seed set." }));
  }
  const anchors = { P1: "d25-horizon-wise-shrinkage-conformal-intervals-and-residual-pretraining-1516-oct-2026",
    P2: "d26-unseen-usage-conditions-p2-for-the-residual-tft--ewc-by-origin-history-1718-oct-2026" };
  kids.push(el("p", {}, el("a", { href: `${PATHS.repo}docs/RESULTS.md#${anchors[state.protocol]}`, target: "_blank", rel: "noopener", text: "Read the result in docs/RESULTS.md ↗" })));
  root.replaceChildren(...kids);
}

export function renderCaveats(root, summary) {
  root.replaceChildren(...summary.caveats.map((c) => el("li", { text: c })));
}
