// Chart primitives for the native views: SVG line chart, dot rows (dot plot / strip plot), scatter, and an HTML matrix.
// Rules: one y-axis, colours by entity in fixed order, 2px lines, >= 8px markers with a surface ring,
// recessive grid, hover tooltips with hit targets larger than the marks, and a table view for every chart.
import { el, fmt } from "./views.js";

const SVG = "http://www.w3.org/2000/svg";
export const s = (tag, attrs = {}, ...kids) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v);
  n.append(...kids.flat().filter((k) => k !== null && k !== undefined));
  return n;
};

/** Round tick values covering [lo, hi] (about `n` of them). */
export function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((st) => span / st <= n + 0.5) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}
/** Ticks whose first and last values enclose [lo, hi]. */
export function niceTicks(lo, hi, n = 5) {
  const t = ticks(lo, hi, n), st = t.length > 1 ? t[1] - t[0] : 1;
  const a = Math.floor(lo / st + 1e-9) * st, z = Math.ceil(hi / st - 1e-9) * st;
  const out = [];
  for (let v = a; v <= z + st / 2; v += st) out.push(Number(v.toFixed(10)));
  return out;
}
const niceMax = (v) => { const t = ticks(0, v, 5); return t[t.length - 1] < v ? t[t.length - 1] + (t[1] - t[0]) : t[t.length - 1]; };

/** Sequential fill for t in [0, 1] (one hue, light to dark) and the ink that reads on it. */
export function seqFill(t) {
  // Start one step in from the surface so the smallest value still reads as a filled cell.
  const x = (0.15 + 0.85 * Math.max(0, Math.min(1, t))) * 5, i = Math.min(4, Math.floor(x)), f = Math.round((x - i) * 100);
  return { bg: `color-mix(in oklab, var(--seq-${i + 1}) ${f}%, var(--seq-${i}))`, fg: t > 0.62 ? "var(--on-seq)" : "var(--ink)" };
}
/** Diverging fill for t in [-1, 1]: blue below zero, red above, grey midpoint. */
export function divFill(t) {
  const a = Math.round(Math.min(1, Math.abs(t)) * 100);
  return { bg: `color-mix(in oklab, ${t < 0 ? "var(--div-neg)" : "var(--div-pos)"} ${a}%, var(--div-mid))`, fg: a > 70 ? "#fff" : "var(--ink)" };
}

function attachTip(node, tip, html) {
  node.setAttribute("tabindex", "0");
  const show = () => tip.show(node, typeof html === "function" ? html() : html);
  node.addEventListener("mouseenter", show);
  node.addEventListener("focus", show);
  node.addEventListener("mouseleave", tip.hide);
  node.addEventListener("blur", tip.hide);
}

// Attached: the measured width. Detached: the width recorded on the enclosing card body (see card() in tabs.js).
const width = (root, min = 280) => Math.max(min, Math.round(root.clientWidth || Number(root.closest?.("[data-w]")?.dataset.w) || 640));

/** Legend of chips. items: [{label, color, kind: "line"|"dot"|"ring"}]. */
export function legend(items) {
  return el("div", { class: "clegend" }, items.map((it) => el("span", { class: "chip", style: `--col:${it.color}` },
    el("span", { class: `sw${it.kind && it.kind !== "line" ? ` ${it.kind}` : ""}`, "aria-hidden": "true" }), it.label)));
}

/** A "Show table" toggle and the hidden table it controls. cols: [{key, label, right?, fmt?}]. */
export function tableView(cols, rows, label = "Show table") {
  const table = el("div", { class: "scroll stable", hidden: true }, el("table", {},
    el("thead", {}, el("tr", {}, cols.map((c) => el("th", { class: c.right ? "r" : null, text: c.label })))),
    el("tbody", {}, rows.map((r) => el("tr", {}, cols.map((c) => el("td", { class: c.right ? "r" : null,
      text: c.fmt ? c.fmt(r[c.key], r) : String(r[c.key] ?? "–") })))))));
  const btn = el("button", { type: "button", class: "textbtn", "aria-expanded": "false", text: label });
  btn.addEventListener("click", () => {
    table.hidden = !table.hidden;
    btn.textContent = table.hidden ? label : "Hide table";
    btn.setAttribute("aria-expanded", String(!table.hidden));
  });
  return { btn, table };
}

/**
 * Line chart. series: [{key, label, color, emph, dash, points: [{x, y}]}]; x/y: {label, fmt, min, max, ticks};
 * refs: [{y, label}]. Lines are direct-labelled at their end when there are at most 4 series.
 */
export function lineChart(root, { series, x, y, refs = [], tip, height, ariaLabel }) {
  const W = width(root), H = height || (W < 520 ? 250 : 300);
  const direct = series.length <= 4;
  const longest = Math.max(0, ...series.map((sr) => sr.label.length));
  const m = { l: 52, r: direct ? Math.min(W * 0.4, 18 + longest * 6.4) : 16, t: 14, b: 44 };
  const xs = series.flatMap((sr) => sr.points.map((p) => p.x));
  const ys = series.flatMap((sr) => sr.points.map((p) => p.y)).concat(refs.map((r) => r.y));
  const x0 = x.min ?? Math.min(...xs), x1 = x.max ?? Math.max(...xs);
  const yt = ticks(y.min ?? Math.min(0, ...ys), y.max ?? niceMax(Math.max(...ys)), 5);
  const y0 = yt[0], y1 = yt[yt.length - 1];
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": ariaLabel || `${y.label} against ${x.label}` });
  for (const v of yt) svg.append(s("line", { class: "gl", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }),
    s("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, (y.fmt || String)(v)));
  for (const v of x.ticks || ticks(x0, x1, W < 520 ? 4 : 8)) svg.append(s("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle" }, (x.fmt || String)(v)));
  svg.append(s("line", { class: "ax", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }),
    s("text", { class: "title-x", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, x.label),
    s("text", { class: "title-y", x: 14, y: (m.t + H - m.b) / 2, transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle" }, y.label));
  for (const r of refs) svg.append(s("line", { class: "ref", x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y) }),
    s("text", { class: "reflab", x: W - m.r - 4, y: Y(r.y) - 6, "text-anchor": "end" }, r.label));
  const ordered = [...series].sort((a, b) => Number(Boolean(a.emph)) - Number(Boolean(b.emph)));
  const ends = [];
  for (const sr of ordered) {
    const pts = [...sr.points].sort((a, b) => a.x - b.x);
    if (!pts.length) continue;
    const d = pts.map((p, k) => `${k ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    const g = s("g", { style: `--col:${sr.color}` }, s("path", { class: `ln${sr.emph ? " emph" : ""}`, d, "stroke-dasharray": sr.dash || null }));
    if (pts.length <= 12) for (const p of pts) g.append(s("circle", { class: "pt", cx: X(p.x), cy: Y(p.y), r: sr.emph ? 5 : 4 }));
    svg.append(g);
    const last = pts[pts.length - 1];
    ends.push({ y: Y(last.y), x: X(last.x), label: sr.label, emph: sr.emph });
  }
  if (direct) {   // end labels, nudged apart so they never collide
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
    for (const e of ends) svg.append(s("text", { class: `lab${e.emph ? "" : " muted"}`, x: e.x + 9, y: e.y + 4 }, e.label));
  }
  // Crosshair and tooltip at the nearest x.
  const xsU = [...new Set(xs)].sort((a, b) => a - b);
  const cross = s("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
  const hover = s("rect", { class: "hover", x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b });
  svg.append(cross, hover);
  hover.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = (e.clientX - box.left) * (W / box.width);
    const xv = xsU.reduce((a, b) => (Math.abs(X(b) - px) < Math.abs(X(a) - px) ? b : a));
    cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv)); cross.setAttribute("visibility", "visible");
    const rows = series.map((sr) => ({ sr, p: sr.points.find((p) => p.x === xv) })).filter((r) => r.p).sort((a, b) => a.p.y - b.p.y);
    const html = `<b>${x.label} ${(x.fmt || String)(xv)}</b>` + rows.map(({ sr, p }) =>
      `<br>${sr.emph ? "<b>" : ""}${sr.label}: ${(y.tipFmt || y.fmt || String)(p.y)}${sr.emph ? "</b>" : ""}`).join("");
    const left = box.left + X(xv) * (box.width / W);
    tip.show({ getBoundingClientRect: () => ({ left, top: box.top + m.t * (box.height / H), width: 0 }) }, html);
  });
  hover.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
  root.replaceChildren(el("div", { class: "chart" }, svg));
}

/**
 * Dot rows: one row per item, marks along a shared x-axis.
 * rows: [{label, emph, marks: [{v, color, open, r, jitter, tip}], range?: [lo, hi]}]; x: {label, min, max, fmt};
 * refs: [{x, label}]. Used for the headline dot plot, coverage per group and seed spread.
 */
export function dotRows(root, { rows, x, refs = [], tip, rowH = 34, labelW, ariaLabel, onRow }) {
  const W = width(root);
  const lw = labelW ?? Math.min(240, Math.max(110, W * 0.3));
  const m = { l: lw, r: 22, t: refs.some((r) => r.label) ? 24 : 8, b: 44 };
  const H = m.t + m.b + rows.length * rowH;
  const vals = rows.flatMap((r) => r.marks.map((k) => k.v)).concat(refs.map((r) => r.x));
  const xt = ticks(x.min ?? 0, Math.max(x.max ?? 0, ...vals) > (x.max ?? -Infinity) ? niceMax(Math.max(...vals)) : x.max, W < 520 ? 4 : 6);
  const x0 = Math.min(xt[0], x.min ?? xt[0]), x1 = Math.max(xt[xt.length - 1], ...vals);
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": ariaLabel || x.label });
  rows.forEach((r, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    if (r.emph) svg.append(s("rect", { class: "rowband emph", x: 0, y: cy - rowH / 2 + 2, width: W, height: rowH - 4, rx: 8 }));
    const lab = s("text", { class: `rowlab${r.emph ? " emph" : ""}`, x: 10, y: cy + 4 }, r.label);
    if (onRow) { lab.style.cursor = "pointer"; lab.addEventListener("click", () => onRow(r)); }
    svg.append(lab);
  });
  for (const v of xt) svg.append(s("line", { class: "gl", x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }),
    s("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle" }, (x.fmt || String)(v)));
  svg.append(s("line", { class: "ax", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }),
    s("text", { class: "title-x", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, x.label));
  const sorted = [...refs].sort((a, c) => a.x - c.x);
  sorted.forEach((r, i) => {
    svg.append(s("line", { class: "ref", x1: X(r.x), x2: X(r.x), y1: m.t - 4, y2: H - m.b }));
    if (!r.label) return;
    // Neighbouring labels closer than their width go on either side of their line instead of centred.
    const near = (j) => sorted[j] && Math.abs(X(sorted[j].x) - X(r.x)) < 7 * (r.label.length + sorted[j].label.length) / 2;
    const anchor = near(i - 1) ? "start" : near(i + 1) ? "end" : "middle";
    const dx = anchor === "start" ? 4 : anchor === "end" ? -4 : 0;
    svg.append(s("text", { class: "reflab", x: X(r.x) + dx, y: m.t - 9, "text-anchor": anchor }, r.label));
  });
  rows.forEach((r, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    if (r.range) svg.append(s("line", { class: "whisk", x1: X(r.range[0]), x2: X(r.range[1]), y1: cy, y2: cy }));
    for (const k of r.marks) {
      const yy = cy + (k.jitter || 0) * (rowH * 0.32);
      const g = s("g", { class: "mark", style: `--col:${k.color}` },
        s("circle", { cx: X(k.v), cy: yy, r: 10, fill: "transparent" }),
        s("circle", { class: `pt${k.open ? " open" : ""}`, cx: X(k.v), cy: yy, r: k.r || 5.5 }));
      if (k.tip) attachTip(g, tip, k.tip);
      svg.append(g);
    }
  });
  root.replaceChildren(el("div", { class: "chart" }, svg));
}

/** Scatter. points: [{x, y, color, open, r, label, tip}]; diag draws y = x; labels only for points with `label`. */
export function scatter(root, { points, x, y, diag = false, tip, height, ariaLabel, square = false }) {
  const W = width(root), H = height || (square ? Math.min(W, 420) : (W < 520 ? 280 : 340));
  const m = { l: 54, r: 18, t: 14, b: 46 };
  const xv = points.map((p) => p.x), yv = points.map((p) => p.y);
  let lo = Math.min(...xv, ...(diag ? yv : [])), hi = Math.max(...xv, ...(diag ? yv : []));
  const pad = (hi - lo) * 0.06 || 0.1;
  const xt = niceTicks(x.min ?? lo - pad, x.max ?? hi + pad, W < 520 ? 4 : 6);
  const yt = diag ? xt : niceTicks(y.min ?? Math.min(...yv) - pad, y.max ?? Math.max(...yv) + pad, 5);
  const [x0, x1] = [xt[0], xt[xt.length - 1]], [y0, y1] = [yt[0], yt[yt.length - 1]];
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": ariaLabel || `${y.label} against ${x.label}` });
  for (const v of yt) svg.append(s("line", { class: "gl", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }),
    s("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, (y.fmt || String)(v)));
  for (const v of xt) svg.append(s("line", { class: "gl", x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }),
    s("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle" }, (x.fmt || String)(v)));
  svg.append(s("line", { class: "ax", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }),
    s("text", { class: "title-x", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, x.label),
    s("text", { class: "title-y", x: 14, y: (m.t + H - m.b) / 2, transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle" }, y.label));
  if (diag) {
    const a = Math.max(x0, y0), b = Math.min(x1, y1);
    svg.append(s("line", { class: "ref", x1: X(a), y1: Y(a), x2: X(b), y2: Y(b) }),
      s("text", { class: "reflab", x: X(b) - 4, y: Y(b) + 14, "text-anchor": "end" }, diag === true ? "y = x" : diag));
  }
  const many = points.length > 300;
  for (const p of points) {
    const c = s("circle", { class: `pt${p.open ? " open" : ""}`, cx: X(p.x), cy: Y(p.y), r: p.r || (many ? 2.5 : 5.5),
      style: `--col:${p.color}${many ? ";stroke:none;opacity:.45" : ""}` });
    if (!many && p.tip) {
      const g = s("g", { class: "mark", style: `--col:${p.color}` }, s("circle", { cx: X(p.x), cy: Y(p.y), r: 11, fill: "transparent" }), c);
      attachTip(g, tip, p.tip);
      svg.append(g);
    } else svg.append(c);
  }
  for (const p of points.filter((q) => q.label)) svg.append(s("text", { class: "lab", x: X(p.x) + 9, y: Y(p.y) - 8 }, p.label));
  root.replaceChildren(el("div", { class: "chart" }, svg));
}

/**
 * Matrix as an HTML table (accessible, keyboard-focusable cells).
 * rows: [{key, label, emph}], cols: [{key, label}], cell(r, c) -> {v, tip} | null, fill(v) -> {bg, fg},
 * best: "min" marks the lowest value per column with a dot.
 */
export function matrix(root, { rows, cols, cell, fill, fmtV = fmt, best = null, tip, onCell, selCol, corner = "" }) {
  const bestIn = {};
  if (best) for (const c of cols) {
    let bv = Infinity, bk = null;
    for (const r of rows) { const v = cell(r, c)?.v; if (v !== null && v !== undefined && v < bv) { bv = v; bk = r.key; } }
    bestIn[c.key] = bk;
  }
  const head = el("tr", {}, el("th", { text: corner }), cols.map((c) => el("th", { class: "col" + (selCol === c.key ? " sel" : ""), scope: "col", text: c.label })));
  if (onCell) head.querySelectorAll("th.col").forEach((th, i) => th.addEventListener("click", () => onCell(null, cols[i])));
  const body = rows.map((r) => el("tr", { class: r.emph ? "proposed" : null }, el("th", { scope: "row", text: r.label }),
    cols.map((c) => {
      const v = cell(r, c);
      if (!v || v.v === null || v.v === undefined || Number.isNaN(v.v)) return el("td", { class: "cell na", text: "–" });
      const f = fill(v.v);
      const td = el("td", { class: "cell" + (selCol === c.key ? " sel" : ""), style: `--bg:${f.bg};--fg:${f.fg}`,
        "aria-label": `${r.label}, ${c.label}: ${fmtV(v.v)}` },
      bestIn[c.key] === r.key ? el("span", { class: "dot", "aria-hidden": "true", text: "●" }) : null, fmtV(v.v));
      if (v.tip) attachTip(td, tip, v.tip);
      if (onCell) {
        const pick = () => onCell(r, c);
        td.addEventListener("click", pick);
        td.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      }
      return td;
    })));
  root.replaceChildren(el("div", { class: "scroll" }, el("table", { class: "heat" }, el("thead", {}, head), el("tbody", {}, body))));
}

/** Legend for a continuous fill: low label, ramp, high label. */
export function rampLegend(lo, hi, gradient, note = "") {
  return el("div", { class: "legend" }, el("span", { text: lo }),
    el("span", { class: "ramp", style: `background:${gradient}`, "aria-hidden": "true" }), el("span", { text: hi }),
    note ? el("span", { text: `· ${note}` }) : null);
}
export const SEQ_GRADIENT = "linear-gradient(90deg, color-mix(in oklab, var(--seq-1) 75%, var(--seq-0)), var(--seq-1), var(--seq-2), var(--seq-3), var(--seq-4), var(--seq-5))";
export const DIV_GRADIENT = "linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos))";
