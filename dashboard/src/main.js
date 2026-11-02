// Entry point: state (kept in the URL hash), tabs, filter bar, native views and Tableau embeds with a static fallback.
import { TABLEAU, PATHS } from "../config.js";
import { mountViz, applyState } from "./embed.js";
import { el, renderKpis, renderHeadline, renderPerGroup, renderDetails, renderCaveats } from "./views.js";

const DEFAULTS = { tab: "overview", dataset: "S6", setting: "e1", protocol: "P2", group: "all", family: "all" };
const KEYS = ["dataset", "setting", "protocol", "group", "family"];

// What each analytical tab shows. Figures are the paper's own (paper/figures), used when no Tableau view is set.
const TABS = {
  overview: { title: "Overview" },
  soh: {
    title: "SOH explorer",
    lede: "SOH trajectories of every cell, relative to its first valid capacity check, with the reuse-class bands (C0 above 95 %, C1 90–95 %, C2 80–90 %, C3 at or below 80 %). S1 and S6 are shown separately and never pooled.",
    figures: [["13_soh_trajectories.png", "S6 SOH trajectories per cell and usage condition."],
      ["preprocessing_s6.png", "S6 preprocessing: from raw checks to labelled samples."]],
  },
  accuracy: {
    title: "Forecast accuracy",
    lede: "Test RMSE per battery group, by horizon, and skill against persistence. Simple baselines are always shown alongside.",
    figures: [["d24_per_group_e1.png", "P1 final error per task, all methods (E1)."],
      ["d24_horizon_e2.png", "Error by horizon (E2, h = 1-6)."],
      ["d30_error_by_group.png", "D30: error per held-out group, S1 and S6."]],
  },
  continual: {
    title: "Continual learning",
    lede: "Adaptation (error on the newest task) against retention (error on earlier tasks after later learning), forgetting and backward transfer by arrival order, and EWC against fine-tuning and joint training.",
    figures: [["d24_forgetting_e1.png", "Forgetting by method (E1)."], ["13_tft_ewc_matrix.png", "TFT + EWC: error on each task after each training step."],
      ["d24_comparison_e1.png", "Final error against forgetting, all continual methods (E1)."]],
  },
  reuse: {
    title: "Reuse & uncertainty",
    lede: "Reuse-class confusion of the forecasts, and the coverage of split-conformal intervals on unseen usage conditions. Classes describe capacity fade since the first check, not suitability for an application.",
    figures: [["d26_confusion.png", "Reuse classes of the proposed model on unseen conditions (D26)."],
      ["d27_coverage.png", "Coverage of 90 % intervals per held-out condition (D27)."], ["d30_confusion.png", "D30 confusion matrices."]],
  },
};

const state = { ...DEFAULTS };
let summary = null;
let selection = null;
const vizzes = {};

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
    tipEl.innerHTML = html;
    tipEl.hidden = false;
    const w = tipEl.offsetWidth / 2;
    tipEl.style.left = `${Math.min(Math.max(r.left + r.width / 2, w + 8), innerWidth - w - 8)}px`;
    tipEl.style.top = `${r.top}px`;
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
    btn.textContent = `Theme: ${t}`;
  };
  btn.addEventListener("click", () => {
    t = order[(order.indexOf(t) + 1) % order.length];
    try { localStorage.setItem("dash-theme", t); } catch { /* ignore */ }
    apply();
  });
  apply();
}

// ---------- filters ----------
const form = document.getElementById("filters");
function syncForm() {
  for (const k of KEYS) form.elements[k].value = state[k];
  const tableauOnly = state.tab !== "overview" && !hasViz(state.tab);
  // Filters drive the native views and the embedded views; the static figures do not respond to them.
  for (const k of KEYS) form.elements[k].disabled = tableauOnly;
}
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

// ---------- tabs ----------
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
  if (name !== "overview") ensureTab(name);
}
tabButtons.forEach((b, i) => {
  b.addEventListener("click", () => {
    showTab(b.id.slice(2));
    update();
    const nav = document.querySelector(".tabs");
    if (nav.getBoundingClientRect().top <= 1) scrollTo({ top: document.querySelector("header").getBoundingClientRect().bottom + scrollY });
  });
  b.addEventListener("keydown", (e) => {
    const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!d) return;
    e.preventDefault();
    const next = tabButtons[(i + d + tabButtons.length) % tabButtons.length];
    showTab(next.id.slice(2), true);
    update();
  });
});

// ---------- analytical tabs: Tableau view or the paper figures ----------
const hasViz = (name) => Boolean(TABLEAU.views[name]) && vizzes[name] !== "failed";

function fallback(name, reason) {
  const t = TABS[name];
  const why = reason
    ? `The interactive Tableau view could not be shown (${reason}). These are the paper's static figures and do not follow the filters.`
    : "The interactive Tableau view for this tab is not published yet (dashboard/config.js). These are the paper's static figures and do not follow the filters.";
  return [el("p", { class: "notice", text: why }),
    el("div", { class: "figs" }, t.figures.map(([f, cap]) => el("figure", {},
      el("img", { src: PATHS.figures + f, alt: cap, loading: "lazy" }), el("figcaption", { text: cap }))))];
}

function ensureTab(name) {
  const panel = document.getElementById(`tab-${name}`);
  if (panel.dataset.ready) return;
  panel.dataset.ready = "1";
  const t = TABS[name];
  const body = el("div", { class: "card" }, el("h2", { text: t.title }), el("p", { class: "sub", text: t.lede }));
  panel.replaceChildren(body);
  if (!TABLEAU.views[name]) {
    body.append(...fallback(name));
    return;
  }
  const holder = el("div", { class: "viz", "aria-label": `${t.title} (Tableau)` }, el("p", { class: "sub", style: "padding:16px", text: "Loading the Tableau view…" }));
  body.append(holder);
  mountViz(holder, TABLEAU.views[name], {
    onReady: (viz) => { vizzes[name] = viz; applyState(viz, state); },
    onFail: (reason) => { vizzes[name] = "failed"; holder.replaceWith(...fallback(name, reason)); syncForm(); },
    onMarks: (rows) => {
      if (!rows.length) return;
      const g = rows[0].Group;
      selection = { group: g || rows[0][Object.keys(rows[0])[0]], fields: rows[0] };
      renderSide();
    },
  });
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
  if (selection.fields?.Group && state.group !== selection.fields.Group) {
    const b = el("button", { type: "button", class: "ghost", text: `Filter every view to ${selection.fields.Group}` });
    b.addEventListener("click", () => { state.group = selection.fields.Group; update(); });
    document.getElementById("details-body").append(b);
  }
}

function update() {
  writeHash();
  syncForm();
  renderHeadline(document.getElementById("headline"), document.getElementById("headline-sub"), summary, state);
  renderPerGroup(document.getElementById("pergroup"), document.getElementById("pergroup-title"),
    document.getElementById("pergroup-sub"), summary, state, {
      tip,
      onSelect: (s) => { selection = s; if (s.group) state.group = s.group; update(); },
    });
  renderSide();
  for (const v of Object.values(vizzes)) if (v && v !== "failed") applyState(v, state);
}

document.getElementById("details-close").addEventListener("click", () => { selection = null; state.group = "all"; update(); });
addEventListener("popstate", () => { readHash(); fillGroups(); showTab(state.tab); update(); });
addEventListener("scroll", tip.hide, { passive: true });

async function start() {
  initTheme();
  document.getElementById("platform-link").href = PATHS.platform;
  readHash();
  try {
    const res = await fetch(PATHS.summary);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    summary = await res.json();
  } catch (e) {
    document.getElementById("kpis").replaceChildren(el("div", { class: "fact" }, el("span", { class: "k", text: "No data" }),
      el("span", { class: "n", text: `Could not read ${PATHS.summary} (${e.message}). Run python scripts/export_dashboard_data.py and serve the page from the repository root.` })));
    document.getElementById("provenance").textContent = "No summary loaded.";
    return;
  }
  renderKpis(document.getElementById("kpis"), summary);
  renderCaveats(document.getElementById("caveats"), summary);
  document.getElementById("provenance").textContent =
    `Built ${summary.generated_utc.slice(0, 10)} from commit ${String(summary.commit || "unknown").slice(0, 7)} by scripts/export_dashboard_data.py.`;
  fillGroups();
  if (state.group !== "all") selection = { group: state.group };
  showTab(state.tab);
  update();
}

start();
