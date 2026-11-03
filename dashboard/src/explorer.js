// Data explorer: every exported table behind the figures, with its columns, row count, a filterable preview and a
// download link, plus the source files and hashes from manifest.json.
import { PATHS } from "../config.js";
import { el } from "./views.js";
import { load } from "./tabs.js";
import { TABLES } from "./content.js";

const PREVIEW = 200;

async function readManifest() {
  try {
    const r = await fetch(`${PATHS.data}manifest.json`);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

/** Preview of one table: a text filter over all columns and the first PREVIEW matching rows. */
async function preview(host, name, local) {
  host.replaceChildren(el("p", { class: "sub", text: "Loading…" }));
  let rows;
  try { rows = await load(name, local); } catch (e) {
    host.replaceChildren(el("p", { class: "notice", text: local
      ? `${name} is built from the local data folder and is not in this copy (${e.message}). Run python scripts/export_dashboard_data.py with the data folder present.`
      : `${name} could not be read (${e.message}).` }));
    return;
  }
  const cols = rows.length ? Object.keys(rows[0]) : [];
  const input = el("input", { type: "search", placeholder: "Filter rows (matches any column)", "aria-label": `Filter ${name}` });
  const count = el("span", { class: "sub" });
  const tbody = el("tbody");
  const draw = () => {
    const q = input.value.trim().toLowerCase();
    const hit = q ? rows.filter((r) => cols.some((c) => String(r[c]).toLowerCase().includes(q))) : rows;
    count.textContent = `${hit.length.toLocaleString()} of ${rows.length.toLocaleString()} rows${hit.length > PREVIEW ? `, first ${PREVIEW} shown` : ""}`;
    tbody.replaceChildren(...hit.slice(0, PREVIEW).map((r) => el("tr", {}, cols.map((c) => el("td", { text: r[c] })))));
  };
  input.addEventListener("input", draw);
  host.replaceChildren(el("div", { class: "xtools" }, input, count),
    el("div", { class: "scroll stable tall" }, el("table", { class: "xtable" }, el("thead", {}, el("tr", {}, cols.map((c) => el("th", { text: c })))), tbody)));
  draw();
}

export async function renderExplorer(root) {
  const manifest = await readManifest();
  const counts = manifest?.row_counts || {};
  const list = TABLES.map(([name, where, desc]) => {
    const local = where === "local";
    const href = `${local ? PATHS.local : PATHS.data}${name}.csv`;
    const body = el("div", { class: "xbody" });
    const det = el("details", { class: "xitem", id: `x-${name}` },
      el("summary", {},
        el("span", { class: "xname", text: name }),
        el("span", { class: "xrows", text: counts[name] !== undefined ? `${counts[name].toLocaleString()} rows` : "" }),
        el("span", { class: `xwhere ${where}`, text: local ? "local data folder" : "in Git" }),
        el("span", { class: "xdesc", text: desc })),
      el("p", { class: "sub" }, el("a", { href, download: `${name}.csv`, text: `Download ${name}.csv` }), local ? " (built locally; not published with the repository)" : ""),
      body);
    det.addEventListener("toggle", () => { if (det.open && !body.childNodes.length) preview(body, name, local); });
    return det;
  });
  const sources = manifest ? Object.entries(manifest.sources || {}) : [];
  root.replaceChildren(
    el("header", { class: "pagehead" }, el("h1", { text: "Data explorer" }),
      el("p", { class: "deck", text: "Every table behind the dashboard, as exported by scripts/export_dashboard_data.py. Open a table to see its columns and rows, filter them, or download the CSV. Tables are long format: one row per run, task, group or check." })),
    el("section", { class: "sect" },
      el("h2", { text: "Tables" }),
      manifest ? el("p", { class: "lede", text: `Exported on ${manifest.generated_utc.slice(0, 10)} from commit ${String(manifest.commit).slice(0, 7)}.` }) : null,
      el("div", { class: "xlist" }, list)),
    el("section", { class: "sect" },
      el("h2", { text: "Source files" }),
      el("p", { class: "lede", text: "Each results file the export read, with its SHA-256 (line endings normalised), so any number can be traced to a file and a commit." }),
      sources.length
        ? el("div", { class: "scroll stable tall" }, el("table", { class: "xtable" },
          el("thead", {}, el("tr", {}, el("th", { text: "File" }), el("th", { text: "SHA-256" }))),
          el("tbody", {}, sources.map(([f, h]) => el("tr", {}, el("td", { text: f }), el("td", { class: "hash", text: h }))))))
        : el("p", { class: "notice", text: "manifest.json could not be read." })));
}
