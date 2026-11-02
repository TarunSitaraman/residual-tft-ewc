// Entry point: state (kept in the URL hash), section rail, filter bar, native views, and optional Tableau embeds.
import { TABLEAU, PATHS } from "../config.js";
import { mountViz, applyState } from "./embed.js";
import { el, fmt, renderHeadline, renderDetails, renderCaveats } from "./views.js";
import { parseCsv, sohRows, renderSoh, unitFacts } from "./soh.js";
import { renderHeadlineChart, headlineFinding, renderAccuracy, renderContinual, renderReuse, takeaway, card, setLayoutWidth } from "./tabs.js";

const DEFAULTS = { tab: "overview", dataset: "S6", setting: "e1", protocol: "P2", group: "all", family: "all" };
const KEYS = ["dataset", "setting", "protocol", "group", "family"];

const TABS = {
  overview: { title: "How well a continual model forecasts battery health",
    deck: "Forecasts of state of health (SOH) at the next capacity checks, from a model that keeps learning as new battery groups arrive. Every number is read from a committed results file and shown beside simple baselines." },
  soh: { title: "Capacity fade of every cell", deck: "SOH of each cell relative to its own first valid capacity check, with the reuse-class boundaries at 95, 90 and 80 %. S1 and S6 are never drawn on one axis." },
  accuracy: { title: "Forecast error by group and horizon", deck: "Test error for each battery group, skill against carrying the last SOH forward, and how error grows with the forecast horizon." },
  continual: { title: "Learning new groups without forgetting old ones", deck: "Groups arrive one after another. A useful model adapts to each new group and keeps its accuracy on the earlier ones." },
  reuse: { title: "Reuse classes and prediction intervals", deck: "How often a forecast puts a cell in the right reuse class, and whether the prediction intervals hold their stated coverage on unseen usage conditions." },
};
const NAMES = { overview: "Overview", soh: "Capacity fade", accuracy: "Forecast error", continual: "Continual learning", reuse: "Reuse and intervals" };

const state = { ...DEFAULTS };
let summary = null;
let selection = null;
let soh = [];
const vizzes = {};
let renderToken = 0;

// ---------- state <-> URL hash ----------
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(DEFAULTS)) state[k] = p.get(k) || DEFAULTS[k];
  if (!TABS[state.tab]) state.tab = DEFAULTS.tab;
}
function writeHash() {
  const p = new URLSearchParams();
  for (const k of Object.keys(DEFAULTS)) if (state[k] !== DEFAULTS[k]) p.set(k, state[k]);
  const h = p.toString();
  if (h !== location.hash.slice(1)) history.pushState(null, "", h ? `#${h}` : location.pathname + location.search);
}

// ---------- tooltip ----------
const tipEl = document.getElementById("tip");
const tip = {
  show(target, html) {
    const r = target.getBoundingClientRect();
    tipEl.innerHTML = html;   // composed by the views from committed results and fixed text
    tipEl.hidden = false;
    const w = tipEl.offsetWidth / 2;
    tipEl.style.left = `${Math.min(Math.max(r.left + r.width / 2, w + 8), innerWidth - w - 8)}px`;
    tipEl.style.top = `${Math.max(r.top, tipEl.offsetHeight + 16)}px`;
  },
  hide() { tipEl.hidden = true; },
};

// ---------- theme ----------
function initTheme() {
  const btn = document.getElementById("theme");
  const order = ["auto", "light", "dark"];
  let t = "auto";
  try { t = localStorage.getItem("dash-theme") || "auto"; } catch { /* storage unavailable */ }
  const apply = () => {
    if (t === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.dataset.theme = t;
    btn.textContent = `${t[0].toUpperCase()}${t.slice(1)} theme`;
  };
  btn.addEventListener("click", () => {
    t = order[(order.indexOf(t) + 1) % order.length];
    try { localStorage.setItem("dash-theme", t); } catch { /* ignore */ }
    apply();
    update();   // charts read colours through CSS variables; redraw so the canvas-free SVG picks up any computed values
  });
  apply();
}

// ---------- filters ----------
const form = document.getElementById("filters");
function syncForm() { for (const k of KEYS) form.elements[k].value = state[k]; }
function fillGroups() {
  const sel = form.elements.group;
  const groups = summary.groups.filter((g) => g.dataset === state.dataset);
  sel.replaceChildren(el("option", { value: "all", text: "All groups" }),
    ...groups.map((g) => el("option", { value: g.group, text: `${g.group} · ${g.dataset === "S6" ? g.description.split(",")[0] : g.chemistry}` })));
  if (state.group !== "all" && !groups.some((g) => g.group === state.group)) state.group = "all";
  sel.value = state.group;
}
form.addEventListener("change", (e) => {
  const k = e.target.name;
  state[k] = e.target.value;
  if (k === "dataset") { fillGroups(); selection = null; }
  if (k === "group") selection = state.group === "all" ? null : { group: state.group };
  update();
});
document.getElementById("reset").addEventListener("click", () => {
  Object.assign(state, { ...DEFAULTS, tab: state.tab });
  selection = null;
  fillGroups();
  update();
});

// ---------- sections ----------
const tabButtons = [...document.querySelectorAll('[role="tab"]')];
function showTab(name, focus = false) {
  state.tab = name;
  for (const b of tabButtons) {
    const on = b.id === `t-${name}`;
    b.setAttribute("aria-selected", on);
    b.tabIndex = on ? 0 : -1;
    if (on && focus) b.focus();
    document.getElementById(b.getAttribute("aria-controls")).hidden = !on;
  }
  const t = TABS[name];
  document.getElementById("page-title").textContent = t.title;
  document.getElementById("page-deck").textContent = t.deck;
  document.title = name === "overview" ? "Residual TFT-EWC results" : `${NAMES[name]}, Residual TFT-EWC results`;
  ensurePanel(name);
}
tabButtons.forEach((b, i) => {
  b.addEventListener("click", () => { showTab(b.id.slice(2)); update(); scrollTo({ top: 0 }); });
  b.addEventListener("keydown", (e) => {
    const d = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
    if (!d) return;
    e.preventDefault();
    showTab(tabButtons[(i + d + tabButtons.length) % tabButtons.length].id.slice(2), true);
    update();
  });
});

// ---------- analytical panels: native views, plus the Tableau view when one is published ----------
function ensurePanel(name) {
  const panel = document.getElementById(`tab-${name}`);
  if (name === "overview" || panel.dataset.ready) return;
  panel.dataset.ready = "1";
  panel.replaceChildren(el("div", { class: "native", style: "display:contents" }));
  if (!TABLEAU.views[name]) return;
  const c = card(`${NAMES[name]} in Tableau`, "The published Tableau Public workbook for this section. It follows the same filters.");
  const holder = el("div", { class: "viz", "aria-label": `${NAMES[name]} (Tableau)` }, el("p", { class: "sub", style: "padding:16px", text: "Loading the Tableau view…" }));
  c.body.append(holder);
  panel.append(c.node);
  mountViz(holder, TABLEAU.views[name], {
    onReady: (viz) => { vizzes[name] = viz; applyState(viz, state); },
    onFail: (reason) => { vizzes[name] = "failed"; holder.replaceWith(el("p", { class: "notice", text: `The Tableau view could not be shown (${reason}). The charts above are complete without it.` })); },
    onMarks: (rows) => {
      if (!rows.length) return;
      selection = { group: rows[0].Group || rows[0][Object.keys(rows[0])[0]], fields: rows[0] };
      renderSide();
    },
  });
}

const onSelect = (sel) => { selection = sel; if (sel.group) state.group = sel.group; update(); };

async function renderNative(name) {
  const host = document.querySelector(`#tab-${name} .native`);
  if (!host) return;
  const token = ++renderToken;
  const ctx = { state: { ...state }, summary, tip, onSelect };
  const sink = el("div", { style: "display:contents" });
  setLayoutWidth(document.querySelector(".panels").clientWidth);
  if (!host.childNodes.length) host.replaceChildren(el("div", { class: "empty", text: "Loading results…" }));
  try {
    if (name === "soh") renderSohTab(sink);
    if (name === "accuracy") await renderAccuracy(sink, ctx);
    if (name === "continual") await renderContinual(sink, ctx);
    if (name === "reuse") await renderReuse(sink, ctx);
  } catch (e) {
    sink.replaceChildren(el("p", { class: "notice", text: `This view could not be drawn: ${e.message}. Run python scripts/export_dashboard_data.py and serve the page from the repository root.` }));
  }
  if (token !== renderToken) return;   // a newer render started while this one loaded data
  host.replaceChildren(...sink.childNodes);
}

function sohFinding(rows, dataset) {
  const units = [...new Set(rows.map((r) => r.unit))];
  const lastOf = (u) => { const r = rows.filter((x) => x.unit === u); return r.reduce((a, x) => (x.x > a.x ? x : a)); };
  const lasts = units.map(lastOf);
  const lo = Math.min(...lasts.map((r) => r.soh)), hi = Math.max(...lasts.map((r) => r.soh));
  const below80 = lasts.filter((r) => r.soh <= 80).length;
  return takeaway(`The ${units.length} ${dataset === "S6" ? "LFP cells" : "NMC second-life cells"} end between <b>${fmt(lo, 1)} %</b> and <b>${fmt(hi, 1)} %</b> of their first capacity check; ${below80} of ${units.length} have crossed the 80 % boundary into class C3.`,
    dataset === "S6" ? `${units.length} cells in ${new Set(rows.map((r) => r.group)).size} usage conditions; colour follows the usage condition.` : "One cell per duty cycle; capacity from the C/20 discharge of each reference performance test.");
}

function renderSohTab(sink) {
  const c = card("SOH trajectories", state.dataset === "S6" ? "SOH from the C/25 capacity of each check, against the cycle at which the check was made." : "SOH from the C/20 discharge capacity of each reference performance test (RPT), pulses excluded.", state.dataset);
  const rows = soh.filter((r) => r.dataset === state.dataset);
  if (!rows.length) {
    c.body.append(el("div", { class: "empty", text: soh.length
      ? `${state.dataset} trajectories are not built in this copy: build the data folder (data/README.md) and run python scripts/export_dashboard_data.py.`
      : "No SOH trajectories yet: build the data folder (data/README.md) and run python scripts/export_dashboard_data.py." }));
    sink.append(c.node);
    return;
  }
  sink.append(sohFinding(rows, state.dataset), c.node);
  const groupOrder = summary.groups.filter((g) => g.dataset === state.dataset).map((g) => g.group);
  const selected = state.group !== "all" && state.dataset === "S1" ? state.group : (selection?.unit || "all");
  const draw = () => renderSoh(c.body, soh, { dataset: state.dataset, selected, groupOrder, tip, onSelect: (unit) => {
    if (unit === "all") { selection = null; state.group = "all"; update(); return; }
    const row = soh.find((r) => r.unit === unit && r.dataset === state.dataset);
    selection = { group: row.group, unit, fields: unitFacts(soh, unit) };
    state.group = row.group;
    update();
  } });
  draw();
}

// ---------- render ----------
function renderSide() {
  const aside = document.getElementById("details");
  const layout = document.querySelector(".layout");
  const show = Boolean(selection);
  aside.hidden = !show;
  layout.classList.toggle("with-details", show);
  if (!show) return;
  renderDetails(document.getElementById("details-body"), document.getElementById("details-title"), summary, state, selection);
}

function renderOverview() {
  const chart = document.getElementById("headline-chart");
  const finding = document.getElementById("overview-finding");
  finding.replaceChildren(headlineFinding(summary, state));
  document.getElementById("headline-caption").textContent = `Test RMSE of SOH in percentage points on S6, ${state.setting === "e1" ? "1 to 3" : "1 to 6"} checks ahead. Proposed model: mean of 50 runs in each seed set (filled: main, ring: replicate). Baselines are single deterministic fits. Lower is better.`;
  renderHeadlineChart(chart, summary, state, tip);
  renderHeadline(document.getElementById("headline"), document.createElement("span"), summary, { ...state, dataset: "S6" });
}

function update() {
  if (!summary) return;
  writeHash();
  syncForm();
  renderSide();
  if (state.tab === "overview") renderOverview(); else renderNative(state.tab);
  for (const v of Object.values(vizzes)) if (v && v !== "failed") applyState(v, state);
}

document.getElementById("details-close").addEventListener("click", () => { selection = null; state.group = "all"; update(); });
addEventListener("popstate", () => { readHash(); fillGroups(); showTab(state.tab); update(); });
addEventListener("scroll", tip.hide, { passive: true });
let resizeTimer = 0, lastW = innerWidth;
addEventListener("resize", () => {
  if (innerWidth === lastW) return;
  lastW = innerWidth;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(update, 180);
});

async function start() {
  initTheme();
  document.getElementById("platform-link").href = PATHS.platform;
  readHash();
  try {
    const res = await fetch(PATHS.summary);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    summary = await res.json();
  } catch (e) {
    document.getElementById("overview-finding").replaceChildren(el("p", { class: "notice",
      text: `Could not read ${PATHS.summary} (${e.message}). Run python scripts/export_dashboard_data.py and serve the page from the repository root.` }));
    document.getElementById("provenance").textContent = "No summary loaded.";
    return;
  }
  renderCaveats(document.getElementById("caveats"), summary);
  document.getElementById("provenance").textContent =
    `Built on ${summary.generated_utc.slice(0, 10)} from commit ${String(summary.commit || "unknown").slice(0, 7)} by scripts/export_dashboard_data.py.`;
  fillGroups();
  try {
    const res = await fetch(PATHS.soh);
    if (res.ok) soh = sohRows(parseCsv(await res.text()));
  } catch { /* optional input: the SOH section says how to build it */ }
  if (state.group !== "all") selection = { group: state.group };
  showTab(state.tab);
  update();
}

start();
