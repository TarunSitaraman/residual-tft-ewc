// SOH explorer: SOH trajectories per cell from data/dashboard/fact_soh.csv (built by scripts/export_dashboard_data.py).
// SOH is relative to each cell's first valid capacity check; S1 and S6 are drawn separately, never on one axis.
import { el, fmt } from "./views.js";
import { ticks } from "./charts.js";

const SVG = "http://www.w3.org/2000/svg";
const CUTS = [[95, "C0 / C1"], [90, "C1 / C2"], [80, "C2 / C3"]];

/** Rows of a CSV with a header line; handles quoted fields with commas and doubled quotes. */
export function parseCsv(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (c === '"') q = false; else field += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return body.map((r) => Object.fromEntries(head.map((h, k) => [h, r[k] ?? ""])));
}

/** Typed, labelled checks only (invalid checks have no SOH and are not drawn). */
export function sohRows(raw) {
  return raw.filter((r) => String(r.valid).toLowerCase() === "true" && r.soh_pct !== "").map((r) => ({
    dataset: r.dataset, group: r.group, unit: r.unit, x: Number(r.check_index), label: r.check_label,
    date: r.check_time, soh: Number(r.soh_pct), cls: r.reuse_class, tempMissing: r.temperature_c === "",
    tempSource: r.temperature_source,
  }));
}

const s = (tag, attrs = {}, ...kids) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  n.append(...kids);
  return n;
};
const niceStep = (span) => [0.5, 1, 2, 5, 10].find((st) => span / st <= 6) || 10;

/** Summary facts of one unit, shown in the details panel when a cell is selected. */
export function unitFacts(rows, unit) {
  const r = rows.filter((x) => x.unit === unit);
  if (!r.length) return null;
  const last = r[r.length - 1];
  return { Checks: String(r.length), "Latest SOH": `${fmt(last.soh)} % (${last.label})`,
    "Lowest SOH": `${fmt(Math.min(...r.map((x) => x.soh)))} %`, "Reuse class (latest)": last.cls,
    "Temperature missing": `${r.filter((x) => x.tempMissing).length} of ${r.length} checks (not filled)` };
}

/**
 * Draw the trajectories of `dataset` into `root`. `groupOrder` fixes the colour of each group (never re-coloured by
 * filters); `selected` is a unit or "all"; `onSelect(unit)` is called from the legend and the lines.
 */
export function renderSoh(root, all, { dataset, selected, groupOrder, groupName, tip, onSelect }) {
  const rows = all.filter((r) => r.dataset === dataset);
  const units = [...new Set(rows.map((r) => r.unit))].sort((a, b) =>
    groupOrder.indexOf(rows.find((r) => r.unit === a).group) - groupOrder.indexOf(rows.find((r) => r.unit === b).group) || a.localeCompare(b));
  const colour = (u) => `var(--c${(groupOrder.indexOf(rows.find((r) => r.unit === u).group) % 7) + 1})`;
  const sel = units.includes(selected) ? selected : "all";
  const xName = dataset === "S1" ? "RPT number" : "Check cycle";
  const someTemp = rows.some((r) => !r.tempMissing);   // S6 records no check temperature: no hollow markers then

  const W = Math.max(300, Math.round(root.clientWidth || Number(root.closest?.("[data-w]")?.dataset.w) || 640)), H = W < 520 ? 260 : Math.min(460, Math.round(W * 0.4));
  const m = { l: 46, r: sel === "all" ? 14 : 44, t: 12, b: 38 };
  const xs = rows.map((r) => r.x), ys = rows.map((r) => r.soh);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.floor(Math.min(...ys, 94.5) * 2) / 2, y1 = Math.ceil(Math.max(...ys, 100) * 2) / 2 + 0.25;
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": `${dataset} SOH per cell against ${xName}, percent of first valid check` });
  // Reuse-class bands: shade C0, draw each boundary the data reach.
  svg.append(s("rect", { class: "band", x: m.l, y: Y(y1), width: W - m.l - m.r, height: Math.max(0, Y(95) - Y(y1)) }));
  const step = niceStep(y1 - y0);
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
    svg.append(s("line", { class: "gl", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }),
      s("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, `${v}`));
  }
  const cutLabels = [];
  for (const [c, name] of CUTS) {
    if (c < y0 || c > y1) continue;
    svg.append(s("line", { class: "ref", x1: m.l, x2: W - m.r, y1: Y(c), y2: Y(c) }));
    cutLabels.push(s("text", { class: "reflab", x: W - m.r - 4, y: Y(c) - 5, "text-anchor": "end" }, `${c} % · ${name}`));
  }
  for (const v of ticks(x0, x1, W < 520 ? 5 : 9).filter((v) => Number.isInteger(v))) svg.append(s("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, `${v}`));
  svg.append(s("line", { class: "ax", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }),
    s("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, xName),
    s("text", { x: 12, y: (m.t + H - m.b) / 2, transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle" },
      "SOH, % of first valid check"));

  const order = sel === "all" ? units : [...units.filter((u) => u !== sel), sel];   // selected line on top
  for (const u of order) {
    const pts = rows.filter((r) => r.unit === u).sort((a, b) => a.x - b.x);
    const g = s("g", { class: "series" + (sel !== "all" && u !== sel ? " dim" : ""), style: `--col:${colour(u)}`,
      "data-unit": u });
    g.append(s("path", { class: "ln", d: pts.map((p, k) => `${k ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.soh).toFixed(1)}`).join("") }),
      s("path", { class: "hitline", d: pts.map((p, k) => `${k ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.soh).toFixed(1)}`).join("") }));
    if (pts.length <= 25 || u === sel) {
      for (const p of pts) g.append(s("circle", { class: p.tempMissing && someTemp ? "pt open" : "pt", cx: X(p.x), cy: Y(p.soh), r: 4 }));
    }
    if (u === sel) {
      const last = pts[pts.length - 1];
      g.append(s("text", { class: "lab", x: X(last.x) + 7, y: Y(last.soh) + 4 }, u));
    }
    g.addEventListener("click", () => onSelect(u));
    svg.append(g);
  }

  svg.append(...cutLabels);   // above the lines, so the boundaries stay readable

  // Hover layer: crosshair at the nearest check, tooltip with every cell's SOH there.
  const cross = s("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
  const hover = s("rect", { class: "hover", x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b });
  svg.append(cross, hover);
  const xsUnique = [...new Set(xs)].sort((a, b) => a - b);
  hover.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = (e.clientX - box.left) * (W / box.width);
    const xv = xsUnique.reduce((a, b) => (Math.abs(X(b) - px) < Math.abs(X(a) - px) ? b : a));
    cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv)); cross.setAttribute("visibility", "visible");
    const at = units.map((u) => rows.find((r) => r.unit === u && r.x === xv)).filter(Boolean).sort((a, b) => b.soh - a.soh);
    const html = `<b>${xName} ${xv}</b>` + at.map((r) => `<br>${r.unit}: ${fmt(r.soh)} %${r.tempMissing ? " (temp. missing)" : ""}`).join("");
    const left = box.left + X(xv) * (box.width / W);
    tip.show({ getBoundingClientRect: () => ({ left, top: box.top + m.t * (box.height / H), width: 0 }) }, html);
  });
  hover.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
  hover.addEventListener("click", (e) => {   // pass clicks through to the nearest line
    hover.style.pointerEvents = "none";
    const under = document.elementFromPoint(e.clientX, e.clientY)?.closest?.("g.series");
    hover.style.pointerEvents = "";
    if (under) onSelect(under.dataset.unit);
  });

  const chip = (u, text) => {
    const b = el("button", { type: "button", class: "chip", "aria-pressed": String(u === sel), style: `--col:${colour(u)}` },
      dataset === "S6" ? null : el("span", { class: "sw", "aria-hidden": "true" }), text);
    b.addEventListener("click", () => onSelect(u === sel ? "all" : u));
    return b;
  };
  // S6: one row per usage condition (the colour), with its cells as buttons. S1: one cell per group, flat.
  const keys = dataset === "S6"
    ? el("div", { class: "lgroups" }, groupOrder.filter((g) => rows.some((r) => r.group === g)).map((g) => {
      const info = groupName ? groupName(g) : { name: g, desc: "" };
      const cells = units.filter((u) => rows.find((r) => r.unit === u).group === g);
      return el("div", { class: "lgroup", style: `--col:${colour(cells[0])}` },
        el("span", { class: "gname", title: info.desc }, el("span", { class: "sw", "aria-hidden": "true" }), info.name),
        el("span", { class: "gdesc", text: info.desc }), cells.map((u) => chip(u, u)));
    }))
    : units.map((u) => chip(u, u));
  const legend = el("div", { class: "clegend", role: "group", "aria-label": "Cells (select to highlight)" }, keys,
    someTemp
      ? el("span", { class: "chip", style: "--col:var(--muted)" }, el("span", { class: "sw ring", "aria-hidden": "true" }), "temperature missing, not filled")
      : el("span", { class: "sub", text: "No check temperature is recorded for this dataset (not filled)." }));

  const table = el("div", { class: "scroll stable", hidden: true }, el("table", {},
    el("thead", {}, el("tr", {}, ["Cell", xName, "Date", "SOH (%)", "Class", "Temperature"].map((h, k) =>
      el("th", { class: k === 3 ? "r" : null, text: h })))),
    el("tbody", {}, rows.filter((r) => sel === "all" || r.unit === sel).map((r) => el("tr", {},
      el("td", { text: r.unit }), el("td", { text: String(r.x) }), el("td", { text: r.date || "–" }),
      el("td", { class: "r", text: fmt(r.soh, 3) }), el("td", { text: r.cls }), el("td", { text: r.tempSource }))))));
  const toggle = el("button", { type: "button", class: "textbtn", "aria-expanded": "false", text: "Show table" });
  toggle.addEventListener("click", () => {
    table.hidden = !table.hidden;
    toggle.textContent = table.hidden ? "Show table" : "Hide table";
    toggle.setAttribute("aria-expanded", String(!table.hidden));
  });

  root.replaceChildren(legend, el("div", { class: "chart" }, svg),
    el("div", { class: "fig-tools" }, el("p", { class: "sub", text:
      `${units.length} cells, ${rows.length} checks. ${dataset === "S1" ? "C/20 discharge capacity of each RPT, pulses excluded (D17)." : "C/25 capacity of each check."}` }), toggle),
    table);
}
