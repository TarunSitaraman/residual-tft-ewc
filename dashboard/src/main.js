// Entry point: state (kept in the URL hash), section rail, filter bar, native views, and optional Tableau embeds.
import { TABLEAU, PATHS } from "../config.js";
import { mountViz, applyState } from "./embed.js";
import { el, fmt, groupInfo, renderHeadline, renderDetails, renderCaveats } from "./views.js";
import { parseCsv, sohRows, renderSoh, unitFacts } from "./soh.js";
import { renderPage, PAGE_TITLES } from "./pages.js";
import { renderExplorer } from "./explorer.js";
import { renderMany, renderEvidence } from "./evidence.js";
import { renderD38 } from "./d38.js";
import { GLOSSARY } from "./content.js";
import { renderHeadlineChart, headlineFinding, renderKpis, renderAccuracy, renderContinual, renderReuse, takeaway, card, setLayoutWidth, headlineTitle } from "./tabs.js";

const DEFAULTS = { page: "home", tab: "overview", dataset: "S6", setting: "e1", protocol: "P2", group: "all", family: "all" };
const KEYS = ["dataset", "setting", "protocol", "group", "family"];

const TABS = {
  overview: { title: "How well a continual model forecasts battery health",
    deck: "Forecasts of state of health (SOH) at the next capacity checks, from a model that keeps learning as new battery groups arrive. Every number is read from a committed results file and shown beside simple baselines." },
  soh: { title: "Capacity fade of every cell", deck: "SOH of each cell relative to its own first valid capacity check, with the reuse-class boundaries at 95, 90 and 80 %. S1 and S6 are never drawn on one axis." },
  accuracy: { title: "Forecast error by group and horizon", deck: "Test error for each battery group, skill against carrying the last SOH forward, and how error grows with the forecast horizon." },
  continual: { title: "Learning new groups without forgetting old ones", deck: "Groups arrive one after another. A useful model adapts to each new group and keeps its accuracy on the earlier ones." },
  reuse: { title: "Reuse classes and prediction intervals", deck: "How often a forecast puts a cell in the right reuse class, and whether the prediction intervals hold their stated coverage on unseen usage conditions." },
  many: { title: "131 unseen usage conditions, one at a time", deck: "The seven S6 conditions, then two first-life datasets with many ageing conditions (S7: 60, S10: 64). Every condition is held out in turn, so each point is a separate test of the model on conditions it has never seen." },
  d38: { title: "Replay ensemble on 124 unseen conditions (D38)", deck: "The residual TFT with a small replay memory, a three-seed ensemble and predict-before-training calibration, tested on the S10 and S7 conditions already used in D36 and D34. A development test with a protocol fixed in advance, not a confirmatory one." },
  evidence: { title: "Every registered result", deck: "Each comparison fixed before its study ran, with its outcome in both seed sets. Results against the proposed model are listed with the ones for it." },
};
const NAMES = { overview: "Overview", soh: "Capacity fade", accuracy: "Forecast error", continual: "Continual learning", reuse: "Reuse and intervals", many: "Many conditions", d38: "Replay ensemble (D38)", evidence: "Registered results" };

const state = { ...DEFAULTS };
let summary = null;
let selection = null;
let soh = [];
const vizzes = {};
let renderToken = 0;

// ---------- state <-> URL hash ----------
let section = "";   // a section of a content page to scroll to (not part of the saved state)
/** Read the hash into the state; false for a plain in-page anchor such as the skip link. */
function readHash() {
  const raw = location.hash.slice(1);
  if (raw && !raw.includes("=")) return false;
  const p = new URLSearchParams(raw);
  for (const k of Object.keys(DEFAULTS)) state[k] = p.get(k) || DEFAULTS[k];
  if (!p.get("page") && p.get("tab")) state.page = "dashboard";   // links from before the site had pages
  if (!PAGE_TITLES[state.page]) state.page = DEFAULTS.page;
  if (!TABS[state.tab]) state.tab = DEFAULTS.tab;
  section = p.get("sec") || "";
  return true;
}
function writeHash() {
  const p = new URLSearchParams();
  const keys = state.page === "dashboard" ? Object.keys(DEFAULTS) : ["page"];
  for (const k of keys) if (state[k] !== DEFAULTS[k] || k === "page") p.set(k, state[k]);
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

// ---------- glossary ----------
function initGlossary() {
  const dlg = document.getElementById("glossary");
  document.getElementById("glossary-body").replaceChildren(...GLOSSARY.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
  document.getElementById("glossary-open").addEventListener("click", () => dlg.showModal());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });   // a click on the backdrop closes it
  // Any link or button with data-glossary opens it too (the reading guide on Home).
  document.addEventListener("click", (e) => { if (e.target.closest?.("[data-glossary]")) { e.preventDefault(); dlg.showModal(); } });
}

// ---------- filters ----------
const form = document.getElementById("filters");
// Filters each section reads; the others are greyed out so nobody expects them to change the figures.
const APPLIES = { overview: ["setting", "family"], soh: ["dataset", "group"], accuracy: KEYS,
  continual: ["setting", "group", "family"], reuse: ["dataset", "setting", "group", "family"],
  many: ["setting"], d38: ["setting"], evidence: ["setting"] };
function syncForm() {
  for (const k of KEYS) {
    const sel = form.elements[k], on = APPLIES[state.tab].includes(k);
    sel.value = state[k];
    sel.disabled = !on;
    sel.closest("label").classList.toggle("off", !on);
    sel.closest("label").title = on ? "" : "This section does not use this filter.";
  }
}
function fillGroups() {
  const sel = form.elements.group;
  const groups = summary.groups.filter((g) => g.dataset === state.dataset);
  sel.replaceChildren(el("option", { value: "all", text: "All groups" }),
    ...groups.map((g) => { const i = groupInfo(summary, g.group); return el("option", { value: g.group, text: g.dataset === "S6" ? `${i.name}: ${i.desc}` : i.name }); }));
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
    if (on) {   // keep the current pane visible in the tab row, which scrolls sideways on narrow screens
      const list = b.parentElement;
      if (b.offsetLeft < list.scrollLeft || b.offsetLeft + b.offsetWidth > list.scrollLeft + list.clientWidth) list.scrollLeft = b.offsetLeft - 16;
    }
    document.getElementById(b.getAttribute("aria-controls")).hidden = !on;
  }
  const t = TABS[name];
  document.getElementById("page-title").textContent = t.title;
  document.getElementById("page-deck").textContent = t.deck;
  document.title = `Dashboard: ${NAMES[name]}, Residual TFT-EWC`;
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
  if (!panel || panel.dataset.ready) return;
  panel.dataset.ready = "1";
  // Overview's native content is static markup in index.html; the other panes are drawn into a .native host.
  if (name !== "overview") panel.replaceChildren(el("div", { class: "native", style: "display:contents" }));
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
    if (name === "many") await renderMany(sink, ctx);
    if (name === "d38") await renderD38(sink, ctx);
    if (name === "evidence") await renderEvidence(sink, ctx);
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
  const draw = () => renderSoh(c.body, soh, { dataset: state.dataset, selected, groupOrder, groupName: (g) => groupInfo(summary, g), tip, onSelect: (unit) => {
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
  document.getElementById("headline-title").textContent = headlineTitle(summary, state);
  renderKpis(document.getElementById("kpis"), summary, state);
  document.getElementById("headline-caption").textContent = `Test RMSE of SOH in percentage points on S6, ${state.setting === "e1" ? "1 to 3" : "1 to 6"} checks ahead. Proposed model: mean of 50 runs in each seed set (filled: main, ring: replicate). Baselines are single deterministic fits. Lower is better; ● marks the lowest error in each panel.`;
  renderHeadlineChart(chart, summary, state, tip);
  renderHeadline(document.getElementById("headline"), document.createElement("span"), summary, { ...state, dataset: "S6" });
}

function update() {
  if (!summary) return;
  writeHash();
  if (state.page !== "dashboard") return;
  syncForm();
  renderSide();
  if (state.tab === "overview") renderOverview(); else renderNative(state.tab);
  for (const v of Object.values(vizzes)) if (v && v !== "failed") applyState(v, state);
}

document.getElementById("details-close").addEventListener("click", () => { selection = null; state.group = "all"; update(); });
// ---------- site pages ----------
const sitePage = document.getElementById("site-page");
async function route() {
  if (!readHash()) return;
  for (const a of document.querySelectorAll(".site-nav a")) {
    if (a.dataset.page === state.page) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
  const dash = state.page === "dashboard";
  for (const id of ["dashbar", "filters", "dash-page"]) document.getElementById(id).hidden = !dash;
  sitePage.hidden = dash;
  if (dash) {
    fillGroups();
    if (state.group !== "all") selection = { group: state.group };
    showTab(state.tab);
    update();
    return;
  }
  document.title = state.page === "home" ? "Residual TFT-EWC" : `${PAGE_TITLES[state.page]}, Residual TFT-EWC`;
  try {
    if (state.page === "explorer") { await renderExplorer(sitePage); scrollTo({ top: 0 }); }
    else await renderPage(state.page, sitePage, { summary, state: { ...state } }, section);
  } catch (e) {
    sitePage.replaceChildren(el("p", { class: "notice", text: `This page could not be drawn: ${e.message}.` }));
  }
}
addEventListener("hashchange", route);
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
  initGlossary();
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
  try {
    const res = await fetch(PATHS.soh);
    if (res.ok) soh = sohRows(parseCsv(await res.text()));
  } catch { /* optional input: the SOH section says how to build it */ }
  route();
}

start();
