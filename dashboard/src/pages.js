// Project pages around the dashboard: Home, Project, Data, Methods, Results and Tableau.
// Fixed text lives in content.js; anything that changes with new results is read from summary.json.
import { PATHS, TABLEAU } from "../config.js";
import { el, fmt, groupInfo } from "./views.js";
import { renderKpis, kpiNote, load } from "./tabs.js";
import * as C from "./content.js";
import { renderStudies, STUDIES } from "./studies.js";

// ---------- building blocks ----------
const section = (id, title, lede, ...kids) => el("section", { class: "sect", id: `s-${id}`, "aria-labelledby": `h-${id}` },
  el("h2", { id: `h-${id}`, text: title }), lede ? el("p", { class: "lede", text: lede }) : null, ...kids);
const table = (head, rows, caption) => el("div", { class: "scroll" }, el("table", { class: "info" },
  caption ? el("caption", { text: caption }) : null,
  el("thead", {}, el("tr", {}, head.map((h) => el("th", { text: h })))),
  el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el(i ? "td" : "th", { scope: i ? null : "row", class: "prose", text: c })))))));
const defs = (pairs) => el("dl", { class: "defs" }, pairs.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
const bullets = (items) => el("ul", { class: "bul" }, items.map((t) => el("li", { text: t })));
const figure = (src, alt, caption) => el("figure", { class: "pfig" }, el("img", { src: `${PATHS.figures}${src}`, alt, loading: "lazy" }),
  el("figcaption", { text: caption }));
const link = (href, text, ext = true) => el("a", { href, text, ...(ext ? { target: "_blank", rel: "noopener" } : {}) });
const pageHead = (title, deck) => el("header", { class: "pagehead" }, el("h1", { text: title }), el("p", { class: "deck", text: deck }));
/** In-page contents for long pages. */
const toc = (items) => el("nav", { class: "toc", "aria-label": "On this page" }, items.map(([id, t]) => el("a", { href: `#page=${currentPage}&sec=${id}`, "data-sec": id, text: t })));
let currentPage = "home";

// ---------- pages ----------
function home(root, ctx) {
  const { summary, state } = ctx;
  const kpis = el("div", { class: "kpis" });
  renderKpis(kpis, summary, state);
  const panes = [["project", "Project", "The question, the contributions, the terms used throughout and the limitations."],
    ["studies", "Studies", "Every experiment from D32 back to the first benchmark, newest first, with its verdicts."],
    ["results", "Results", "The paper's result tables, what was learned, and every registered verdict."],
    ["dashboard", "Dashboard", "Interactive figures: capacity fade, forecast error, continual learning, reuse classes and intervals."],
    ["data", "Data", "The public datasets, the usage conditions, the integrity rules and the Spark pipeline."],
    ["methods", "Methods", "Pipeline, forecasting task, models, continual-learning strategies, protocols and statistics."],
    ["explorer", "Data explorer", "Every exported table behind the figures, with a preview and a download."],
    ["tableau", "Tableau", "How the Tableau Public workbook is built, published and driven by this site."]];
  const h = state.setting === "e1" ? "1 to 3" : "1 to 6";
  root.replaceChildren(
    el("header", { class: "hero" },
      el("p", { class: "kicker", text: "Research project, IEEE conference paper in preparation" }),
      el("h1", { text: "Forecasting the health of second-life batteries, one new group at a time" }),
      el("p", { class: "hero-claim", text: "Of the models compared, the residual TFT with EWC is the only one with no weak side: level with the best baseline on the continual stream, and ahead of extrapolation on usage conditions it has never seen." }),
      el("div", { class: "actions" },
        el("a", { class: "btn primary", href: "#page=dashboard", text: "Open the dashboard" }),
        el("a", { class: "btn", href: "#page=studies", text: "Browse the studies" }),
        el("a", { class: "btn", href: C.PROJECT.links.paper, target: "_blank", rel: "noopener", text: "Read the paper (PDF)" }))),
    el("section", { class: "headline", "aria-label": "Headline numbers" }, kpis,
      el("p", { class: "kpi-note", text: `Test RMSE of SOH in percentage points on S6 (seven LFP usage conditions), ${h} checks ahead, main seed set of 50 runs; read live from the committed results. ${kpiNote(summary, state)}` })),
    section("latest", "Latest studies", "Each study was fixed in writing before it ran; negative results are reported with the positive ones.",
      el("ol", { class: "latest" }, STUDIES.slice(0, 3).map((s) => el("li", {},
        el("a", { href: `#page=studies&sec=study-${s.id}` }, el("span", { class: "sid", text: s.id }), el("b", { text: s.title }), el("span", { text: s.reading })))))),
    section("findings", "What the study found", null,
      el("ol", { class: "findings" }, C.FINDINGS.map(([t, d]) => el("li", {}, el("b", { text: t }), el("span", { text: d }))))),
    section("question", "The question", C.PROJECT.question),
    section("glance", "The study at a glance", null, defs(C.AT_A_GLANCE)),
    section("site", "On this site", null, el("div", { class: "cards" }, panes.map(([p, t, d]) =>
      el("a", { class: "card-link", href: `#page=${p}` }, el("b", { text: t }), el("span", { text: d }))))),
    section("status", "Status", C.PROJECT.status));
}

function project(root) {
  root.replaceChildren(
    pageHead("The project", "Why second-life batteries need a model that keeps learning, what this study contributes, and where its claims stop."),
    toc([["question", "Question"], ["contrib", "Contributions"], ["framework", "Framework"], ["terms", "Terms"], ["limits", "Limitations"], ["team", "Team and citation"]]),
    section("question", "The question", C.PROJECT.question),
    section("contrib", "Contributions", null, bullets(C.CONTRIBUTIONS)),
    section("framework", "Framework", "From the public archives to a reuse class: every stage keeps battery groups separate and uses only what is known at the forecast origin.",
      figure("framework.png", "Framework: public data, verification, labels and samples, purged splits, protocols, models, evaluation and reuse assessment", "Study framework (paper Fig. 1)."),
      figure("architecture.png", "Residual TFT-EWC architecture: TFT correction network on top of linear extrapolation, trained with an EWC penalty", "The proposed model: a TFT predicts a correction to linear extrapolation, shrunk per horizon and trained with EWC; conformal intervals and reuse classes follow.")),
    section("terms", "Terms used on this site", null, defs(C.GLOSSARY)),
    section("limits", "Limitations and honest claims", null, defs(C.LIMITATIONS)),
    section("team", "Team, citation and licence", null,
      defs([["Authors", C.PROJECT.authors.join(", ")], ["Manuscript", "Residual TFT-EWC: Continual Learning for State-of-Health Forecasting and Reuse Screening of Heterogeneous Second-Life EV Batteries (in preparation)"],
        ["Licence", "Code, documentation, results and manuscript are source-available, all rights reserved: you may read them and run the code unmodified to reproduce or verify the results. The datasets keep their own licences (CC BY 4.0)."]]),
      el("p", {}, link(C.PROJECT.links.paper, "Paper (PDF)"), " · ", link(C.PROJECT.links.supplement, "Supplement (PDF)"), " · ", link(C.PROJECT.links.repo, "Repository"))));
}

function data(root, ctx) {
  const { summary } = ctx;
  const s6 = summary.groups.filter((g) => g.dataset === "S6");
  const s1 = summary.groups.filter((g) => g.dataset === "S1");
  root.replaceChildren(
    pageHead("The data", "Public battery-ageing datasets, downloaded from their original archives and verified file by file. The two main datasets are never pooled, so no cross-chemistry transfer is claimed."),
    toc([["sources", "Sources"], ["groups", "Battery groups"], ["rules", "Integrity rules"], ["spark", "Spark pipeline"]]),
    section("sources", "Sources", "Not used: S3 (first-life data of the S1 cells), S4 (synthetic duty-cycle profiles, cited only) and S5 (no licence).",
      table(["ID", "Dataset", "Chemistry and size", "Role"], C.SOURCES)),
    section("groups", "Battery groups", "A group is the unit kept apart between training and test: one usage condition in S6, one cell in S1. Read from dim_group.",
      table(["Group", "Name", "Charge", "Discharge", "Depth of discharge", "Description"], s6.map((g) => {
        const i = groupInfo(summary, g.group);
        return [g.group, i.name, `${g.charge_mode} to ${g.v_max} V at ${Number(Number(g.charge_c_rate).toFixed(2))}C`, g.discharge, `${Math.round(g.dod * 100)} %`, g.description];
      }), `S6: ${s6.length} usage conditions of A123 LFP cells (Wheeler et al. 2025), all at 50 °C.`),
      table(["Group", "Chemistry", "Description"], s1.map((g) => [g.group, g.chemistry, g.description]), `S1: ${s1.length} second-life NMC cells (Moy et al. 2024).`)),
    section("rules", "Integrity rules", "Rules applied to every table before any model sees it.", defs(C.DATA_RULES)),
    section("spark", "Big-data processing with Apache Spark", "The S6 between-check ageing data is the project's big-data workload. Each file is independent, which is the shape of a map-reduce job.",
      defs(C.SPARK), figure("spark_pipeline.png", "Spark pipeline: map over 827 blocks, cache, Spark SQL reduce, validate against pandas", "The Spark job (scripts/spark_ageing_features.py).")));
}

function methods(root) {
  root.replaceChildren(
    pageHead("Methods", "How samples are built, which models and continual-learning strategies are compared, and how a result is judged."),
    toc([["pipeline", "Pipeline"], ["baselines", "Baselines"], ["neural", "Neural models"], ["protocols", "Protocols"], ["stats", "Evaluation and statistics"], ["reuse", "Reuse screening"]]),
    section("pipeline", "End-to-end pipeline", null, el("ol", { class: "steps" }, C.PIPELINE.map(([t, d]) => el("li", {}, el("b", { text: t }), el("span", { text: d }))))),
    section("baselines", "Baselines", "Every learned model is shown beside these; persistence is the floor every forecaster must beat.", defs(C.BASELINES)),
    section("neural", "Neural models and continual-learning strategies",
      "Two backbones: a small MLP with explicit missing indicators, and a TFT over the previous four checks of the same cell. No battery ID is an input, so every method stays task-agnostic. The residual formulation (D23) predicts only the correction to linear extrapolation; it is what made the TFT competitive.",
      defs(C.STRATEGIES)),
    section("protocols", "Protocols", null, table(["Protocol", "Name", "Setting"], C.PROTOCOLS)),
    section("stats", "Evaluation and statistics", null, defs(C.STATS)),
    section("reuse", "Reuse screening", "Forecast SOH is thresholded into C0-C3 with the same rule as measured SOH. The classes rest on capacity relative to each cell's first check, so they screen for fade; a reuse decision would also need power capability, safety and absolute capacity. They are only informative in P2 on S6, where test periods span several classes.",
      table(["Class", "Fade since first check", "SOH"], [["C0", "< 5 pp", "> 95 %"], ["C1", "5-10 pp", "90-95 %"], ["C2", "10-20 pp", "80-90 %"], ["C3", "≥ 20 pp", "≤ 80 %"]])));
}

async function results(root, ctx) {
  const { summary } = ctx;
  root.replaceChildren(
    pageHead("Results", "The paper's result tables, what the study learned, and every registered comparison with its verdict. Full provenance: docs/RESULTS.md."),
    toc([["stream", "Continual stream"], ["unseen", "Unseen conditions"], ["learned", "What we learned"], ["verdicts", "Registered verdicts"]]),
    section("stream", "Continual stream (P1)", null, table(C.STREAM_TABLE.head, C.STREAM_TABLE.rows, C.STREAM_TABLE.caption)),
    section("unseen", "Unseen usage conditions (P2)", null, table(C.UNSEEN_TABLE.head, C.UNSEEN_TABLE.rows, C.UNSEEN_TABLE.caption)),
    section("learned", "What we learned", null, el("ol", { class: "findings plain" }, C.LEARNED.map((t) => el("li", { text: t })))),
    section("verdicts", "Registered verdicts", "Every pre-registered comparison in fact_verdicts.csv. Replicated means p_Holm < 0.05 in both seed sets with the same sign and a difference of at least 0.05 pp.", el("p", { class: "sub", text: "Loading…" })));
  const host = root.querySelector("#s-verdicts");
  try {
    const [rows, methods] = await Promise.all([load("fact_verdicts"), load("dim_method")]);
    const label = Object.fromEntries(methods.map((m) => [m.method, m.label]));
    const cls = (v) => (v.includes("better") ? "better" : v.includes("worse") ? "worse" : "");
    host.querySelector(".sub").replaceWith(el("div", { class: "scroll stable tall" }, el("table", {},
      el("thead", {}, el("tr", {}, ["Study", "Setting", "A", "B", "Metric", "A − B (main / replicate)", "p_Holm (main / replicate)", "Verdict"].map((h, i) => el("th", { class: i === 5 || i === 6 ? "r" : null, text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", {},
        el("td", { text: r.experiment }), el("td", { text: r.setting.toUpperCase() }),
        el("td", { class: "prose", text: label[r.a] || r.a }), el("td", { class: "prose", text: label[r.b] || r.b }), el("td", { text: r.metric }),
        el("td", { class: "r", text: `${fmt(Number(r.diff_main), 3)} / ${fmt(Number(r.diff_replicate), 3)}` }),
        el("td", { class: "r", text: `${Number(r.p_holm_main).toPrecision(2)} / ${Number(r.p_holm_replicate).toPrecision(2)}` }),
        el("td", {}, el("span", { class: `verdict ${cls(r.verdict)}`, text: r.verdict }))))))),
    el("p", { class: "sub", text: `${rows.length} comparisons. Summary built on ${summary.generated_utc.slice(0, 10)}.` }));
  } catch (e) {
    host.querySelector(".sub").textContent = `The verdict table could not be read (${e.message}).`;
  }
}

function tableau(root) {
  const views = Object.entries(TABLEAU.views);
  const live = views.filter(([, u]) => u).length;
  const flow = [["Export", "scripts/export_dashboard_data.py writes tidy CSVs (results/dashboard/, in Git; data/dashboard/, local) plus summary.json and manifest.json with source hashes and the commit."],
    ["Build the kit", "scripts/build_tableau_workbook.py writes tableau_source.xlsx (one pre-filtered sheet per view, filter fields already named), a starter residual_tft_ewc.twb (data sources, calculated fields, 12 worksheets, 5 dashboards) and AUTHORING.md."],
    ["Author", "Open the .twb in Tableau Desktop Public Edition, check each view against WORKBOOK_SPEC.md and fix what Tableau reports. The .twb was generated without Tableau."],
    ["Publish", "Save to Tableau Public as an extract (Tableau Public has no live connections). Each of the five dashboards gets a public URL."],
    ["Configure", "Paste the five URLs into dashboard/config.js (TABLEAU.views)."],
    ["Embed", "This site loads the Tableau Embedding API v3 and mounts a <tableau-viz> under the native figures of the matching dashboard pane."]];
  const runtime = [["Loading", "The API script (tableau.embedding.3.latest.min.js from public.tableau.com) is loaded once, only when a pane has a URL."],
    ["Mounting", "A <tableau-viz> element with the view URL, toolbar at the bottom, tabs hidden, full width and 640 px high."],
    ["Readiness", `The view must fire firstinteractive within ${TABLEAU.timeoutS} s; otherwise a notice replaces it and the native figures stand alone.`],
    ["Filters out", "On every filter change the site calls applyFilterAsync(field, [value], \"replace\") on each worksheet of the dashboard, or clearFilterAsync for \"All\". Worksheets without the field are skipped; parameters with the same name are set too."],
    ["Selections in", "markselectionchanged events are read with getMarksAsync; a selected mark's Group opens the details panel, the same panel the native heatmap cells open."],
    ["Shared state", "Filters and the open pane live in the URL hash, so a link reproduces the same view in the native figures and in Tableau."]];
  root.replaceChildren(
    pageHead("Tableau implementation", "The dashboard's analytical views have two layers: native figures drawn by this site from the exported tables, and a Tableau Public workbook embedded under them. The site is complete without the workbook."),
    section("status", "Current status", live ? `${live} of ${views.length} dashboards are published and embedded.` : "No dashboard is published yet: every pane shows the native figures only. The workbook kit is generated, but it has not been opened in Tableau Desktop, which is not installed on the machine that built it.",
      table(["Dashboard pane", "Tableau dashboard", "View URL"], C.TABLEAU_CONTRACT.dashboards.map(([name, key]) => [key, name, TABLEAU.views[key] || "not published"]))),
    section("flow", "How a Tableau view gets onto this site", null, el("ol", { class: "steps" }, flow.map(([t, d]) => el("li", {}, el("b", { text: t }), el("span", { text: d }))))),
    section("runtime", "What happens in the browser", "Implemented in dashboard/src/embed.js and main.js.", defs(runtime)),
    section("contract", "Contract between the site and the workbook", "The site relies only on these names; everything else in the workbook is free. Every per-group worksheet keeps Group in the view so selections can be read.",
      table(["Field", "Values"], C.TABLEAU_CONTRACT.fields)),
    section("why", "Why Tableau Public", "After export the data are small (about 850 S6 checks, 111 S1 RPTs, 38.5k saved forecasts, 42k run × task rows) and public (CC BY 4.0). Tableau Public is free to publish and its Embedding API lets a page set filters and read selections. Power BI's JavaScript embedding needs paid capacity and Azure AD. The trade-off: Tableau Public views are public and use extracts."),
    section("files", "Files", null, defs([["dashboard/tableau/WORKBOOK_SPEC.md", "Build sheet: contract, data model, calculated fields, the five dashboards and the publishing checklist"],
      ["data/dashboard/tableau/ (local)", "tableau_source.xlsx, residual_tft_ewc.twb and AUTHORING.md from build_tableau_workbook.py"],
      ["dashboard/config.js", "View URLs (TABLEAU.views), field names (FIELDS), API script and timeout"],
      ["dashboard/src/embed.js", "Loads the API, mounts views, applies filters, reads selections"]])));
}

const PAGES = { home, project, data, methods, studies: renderStudies, results, tableau };
export const PAGE_TITLES = { home: "Home", project: "Project", data: "Data", methods: "Methods", studies: "Studies", results: "Results", dashboard: "Dashboard", explorer: "Data explorer", tableau: "Tableau" };

/** Render a content page into `root`; `sec` scrolls to a section once drawn. */
export async function renderPage(name, root, ctx, sec) {
  currentPage = name;
  await PAGES[name](root, ctx);
  const target = sec && root.querySelector(`#s-${sec}`);
  if (target) target.scrollIntoView(); else scrollTo({ top: 0 });
}
