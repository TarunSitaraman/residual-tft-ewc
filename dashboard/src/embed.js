// Tableau Embedding API v3 wrapper: load the API once, mount a <tableau-viz>, push filters, report mark selections.
import { TABLEAU, FIELDS } from "../config.js";

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
    const h = (s.maxSize || s.minSize)?.height;
    if (h > 0) viz.setAttribute("height", `${h + TOOLBAR_PX}px`);
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
  const timer = setTimeout(() => { if (!done) { done = true; onFail("the view did not respond in time"); } }, TABLEAU.timeoutS * 1000);
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

/** Apply the filter-bar state to every worksheet of the view (fields a sheet does not have are skipped). */
export async function applyState(viz, state) {
  const sheet = viz?.workbook?.activeSheet;
  if (!sheet) return;
  const sheets = sheet.sheetType === "dashboard" ? sheet.worksheets : [sheet];
  const jobs = [];
  for (const ws of sheets) {
    for (const [key, field] of Object.entries(FIELDS)) {
      const v = state[key];
      const job = v === "all" ? ws.clearFilterAsync(field) : ws.applyFilterAsync(field, [v], "replace");
      jobs.push(job.catch(() => null));
    }
  }
  // Parameters with the same names (if the workbook uses parameters instead of filters).
  jobs.push(viz.workbook.getParametersAsync().then((params) => Promise.all(params
    .filter((p) => Object.values(FIELDS).includes(p.name))
    .map((p) => {
      const key = Object.keys(FIELDS).find((k) => FIELDS[k] === p.name);
      return state[key] === "all" ? null : p.changeValueAsync(state[key]).catch(() => null);
    }))).catch(() => null));
  await Promise.all(jobs);
}
