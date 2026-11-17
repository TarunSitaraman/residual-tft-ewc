// Tableau Embedding API v3 wrapper: load the API once, mount a <tableau-viz>, push filters, report mark selections.
import { TABLEAU, FIELDS, ALIASES } from "../config.js";

let apiPromise = null;

function loadApi() {
  if (!apiPromise) {
    apiPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.type = "module";
      s.src = TABLEAU.scriptUrl;
      s.onload = resolve;
      s.onerror = () => reject(new Error("Tableau Embedding API could not be loaded"));
      document.head.append(s);
    });
  }
  return apiPromise;
}

// Marks of a selection event as plain objects {fieldName: formattedValue}.
async function marksAsRows(event) {
  const marks = await event.detail.getMarksAsync();
  const rows = [];
  for (const table of marks.data || []) {
    const names = table.columns.map((c) => c.fieldName);
    for (const r of table.data) rows.push(Object.fromEntries(r.map((v, i) => [names[i], v.formattedValue ?? v.value])));
  }
  return rows;
}

// Show the whole dashboard: size the frame to the dashboard's own fixed (or maximum) height plus the toolbar.
// Dashboards sized "Automatic" have no height of their own and keep TABLEAU.heightPx.
const TOOLBAR_PX = 30;
function fitHeight(viz) {
  try {
    const s = viz.workbook.activeSheet.size;
    const max = s.maxSize || s.minSize;
    if (!(max?.height > 0)) return;
    // A dashboard sized as a range is scaled down to the frame width, keeping its proportions; a frame as tall as
    // the maximum left blank space under it.
    const w = viz.getBoundingClientRect().width;
    const scale = max.width > 0 && w > 0 ? Math.min(1, w / max.width) : 1;
    const h = Math.max(s.minSize?.height || 0, Math.round(max.height * scale));
    viz.setAttribute("height", `${h + TOOLBAR_PX}px`);
  } catch { /* size not readable: keep the default height */ }
}

/**
 * Mount a Tableau Public view in `container`.
 * Calls onReady(viz) when it is interactive, onFail(reason) if the API or the view does not load in time,
 * and onMarks(rows) whenever the selection changes.
 */
export async function mountViz(container, url, { onReady, onFail, onMarks }) {
  try {
    await loadApi();
  } catch (e) {
    onFail(e.message);
    return null;
  }
  const viz = document.createElement("tableau-viz");
  viz.setAttribute("src", url);
  viz.setAttribute("toolbar", "bottom");
  viz.setAttribute("hide-tabs", "");
  viz.setAttribute("width", "100%");
  viz.setAttribute("height", `${TABLEAU.heightPx}px`);
  let done = false;
  // A view in a hidden pane does not render, so the clock only runs out while the view is on screen.
  let timer = 0;
  const arm = () => {
    timer = setTimeout(() => {
      if (done) return;
      if (!viz.isConnected || viz.offsetParent === null) { arm(); return; }
      done = true;
      onFail("the view did not respond in time");
    }, TABLEAU.timeoutS * 1000);
  };
  arm();
  viz.addEventListener("firstinteractive", () => {
    if (done) return;
    done = true;
    clearTimeout(timer);
    fitHeight(viz);
    onReady(viz);
  });
  viz.addEventListener("markselectionchanged", async (ev) => {
    try { onMarks(await marksAsRows(ev)); } catch { /* selection cleared or not readable */ }
  });
  container.replaceChildren(viz);
  return viz;
}

/**
 * Apply the filter-bar state to every worksheet of the view (fields a sheet does not have are skipped).
 * Only the filters in `keys` (the ones the pane reads) are applied; the others are cleared, so a filter the
 * pane greys out cannot empty its view (the D38 view has no S1/S6 rows).
 */
export async function applyState(viz, state, keys = Object.keys(FIELDS)) {
  const sheet = viz?.workbook?.activeSheet;
  if (!sheet) return;
  const sheets = sheet.sheetType === "dashboard" ? sheet.worksheets : [sheet];
  const jobs = [];
  for (const ws of sheets) {
    for (const [key, field] of Object.entries(FIELDS)) {
      const v = keys.includes(key) ? ALIASES[key]?.[state[key]] ?? state[key] : "all";
      const job = v === "all" ? ws.clearFilterAsync(field) : ws.applyFilterAsync(field, [v], "replace");
      jobs.push(job.catch(() => null));
    }
  }
  // Parameters with the same names (if the workbook uses parameters instead of filters).
  jobs.push(viz.workbook.getParametersAsync().then((params) => Promise.all(params
    .filter((p) => Object.values(FIELDS).includes(p.name))
    .map((p) => {
      const key = Object.keys(FIELDS).find((k) => FIELDS[k] === p.name);
      return state[key] === "all" || !keys.includes(key) ? null : p.changeValueAsync(state[key]).catch(() => null);
    }))).catch(() => null));
  await Promise.all(jobs);
}
