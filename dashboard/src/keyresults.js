// Key results: the claims the paper makes, each with its number, verdict and where to see it.
// Fixed numbers are copied from paper/main.tex (abstract, Results) and docs/RESULTS.md; keep them in step.
// The headline table is read live from summary.json (the same rows as paper/tables/headline.tex).
import { PATHS } from "../config.js";
import { el, fmt } from "./views.js";

// [big number, unit, claim, detail, verdict, status, dashboard link, link text]
const CLAIMS = [
  ["0.32", "pp", "Level with the best stream baseline",
    "On the S6 continual stream (h = 1-3) the proposed model reaches 0.32 / 0.31 pp test RMSE against 0.34 pp for linear extrapolation, which no learned model beats.",
    "practically equivalent", "Registered (D25)", "#page=dashboard&tab=overview&protocol=P1", "Overview"],
  ["0.24-0.46", "pp", "Ahead of gradient boosting on the stream",
    "Under the same no-storage rule (gradient boosting refitted on the latest condition) the proposed model is 0.24-0.46 pp more accurate; 0.13-0.41 pp against gradient boosting that stores every condition.",
    "replicated, better", "Post hoc (D35)", "#page=dashboard&tab=accuracy&protocol=P1", "Forecast error"],
  ["2/3", "less", "EWC cuts forgetting by two thirds",
    "At horizons 1-6, forgetting of the residual TFT falls from 0.28 / 0.22 pp without EWC to 0.09 pp with it.",
    "replicated, better", "Registered (D23)", "#page=dashboard&tab=continual&setting=e2", "Continual learning"],
  ["0.43-0.68", "pp", "Ahead of extrapolation on unseen conditions",
    "With a whole usage condition held out of training (P2), the proposed model is 0.43-0.68 pp more accurate than linear extrapolation (0.92 against 1.37 pp at h = 1-3).",
    "replicated, better", "Registered (D26)", "#page=dashboard&tab=overview", "Overview"],
  ["95.5", "%", "Correct reuse class on unseen conditions",
    "Forecast SOH puts 95.5 % of held-out S6 samples in their measured fade class (extrapolation 93.9 %); every error is off by one class only.",
    null, "Registered (D30)", "#page=dashboard&tab=reuse", "Reuse and intervals"],
  ["64 / 64", "conditions", "Ahead of gradient boosting on every S10 condition",
    "On a further dataset (S10, 64 held-out conditions) the proposed model beats gradient boosting held to the same no-storage rule on every condition, for both horizon sets (p_Holm < 1e-10). On S7 the same holds for h = 1-3.",
    "replicated, better", "Registered (D36, D34)", "#page=dashboard&tab=many", "Many conditions"],
  ["40", "rows", "A small replay memory closes most of the gap",
    "Keeping 40 rows per condition (replay ensemble V2), the model shows no replicated difference from the jointly trained TFT and reaches pooled 90 % interval coverage of 0.89-0.91. All five registered questions hold on S10.",
    "replicated, better", "Post hoc design, protocol fixed before running (D38)", "#page=dashboard&tab=d38", "Replay ensemble (D38)"],
  ["0.90", "coverage", "Intervals hold when calibrated on held-out conditions",
    "Residuals from seen conditions give 0.66-0.68 coverage at nominal 90 %; residuals from held-out conditions give 0.90. Refitting as a new condition's checks arrive cuts the interval score by 29 %.",
    null, "Registered (D27, D28)", "#page=dashboard&tab=reuse", "Reuse and intervals"],
];

// Where the proposed model does not win: reported in the paper beside the claims above.
const LIMITS = [
  ["Stored data wins on unseen conditions", "Gradient boosting that stores every earlier condition (0.54 pp) and the jointly trained TFT (0.57 pp) are more accurate than the proposed model (0.92 pp) on unseen S6 conditions. Stored-data gradient boosting is also better on 52 of 64 S10 and 50 of 60 S7 conditions; its higher S10 mean comes from a few calendar conditions where it fails."],
  ["Pretraining did not help", "Pretraining on 36,711 first-life samples made S6 forecasts worse (+0.22 pp, replicated)."],
  ["Spark features did not help", "The 21 block-level inputs from 902 million raw rows made the model 0.14-0.19 pp worse (replicated): a negative result."],
  ["S1 is level", "On the six second-life NMC cells all methods are within 0.46-0.48 pp and every sample stays in class C0."],
  ["Out of range is not fixed", "With 100 % depth of discharge held out, every training condition runs at 70 %, so neither forecasts nor intervals can be trusted there."],
];

// Paper figures that carry the claims, in paper order.
const FIGS = [
  ["overview.png", "Error on the stream against error on unseen conditions", "Only the proposed model has no weak setting (S6): extrapolation is good on the stream only, gradient boosting refitted on the latest condition on unseen conditions only."],
  ["dash_s6.png", "Forgetting and per-condition error on S6", "S6 in detail: what is forgotten, and where the error comes from (dashboard views)."],
  ["dash_horizon.png", "Skill against persistence and error by horizon", "Forecasting pays off most at long horizons (S6, P2)."],
  ["dash_unseen.png", "Interval coverage and 124 further unseen conditions", "Unseen conditions: intervals on S6 and the 124 conditions of S7 and S10 (P2, h = 1-3)."],
  ["dash_d38.png", "Verdict grid of the replay ensemble", "Bounded memory: the registered questions for the replay ensemble V2 on S7 and S10."],
];

const cls = (v) => (v?.includes("better") ? "better" : v?.includes("worse") ? "worse" : "");
const verdict = (v) => el("span", { class: `verdict ${cls(v)}`, text: v });

const COLS = [["Continual stream (P1)", "E1 (h = 1-3)"], ["Continual stream (P1)", "E2 (h = 1-6)"],
  ["Unseen usage condition (P2)", "E1 (h = 1-3)"], ["Unseen usage condition (P2)", "E2 (h = 1-6)"]];

/** Paper headline table from summary.headline: RMSE rows, then proposed-minus-baseline rows with verdicts. */
function headlineTable(rows) {
  const cell = (m, [p, s]) => rows.find((r) => r.method === m && r.protocol === p && r.setting === s);
  const methods = [...new Set(rows.filter((r) => !r.method.startsWith("vs ")).map((r) => r.method))];
  const diffs = [...new Set(rows.filter((r) => r.method.startsWith("vs ")).map((r) => r.method))];
  const label = (m) => rows.find((r) => r.method === m).label;
  const pair = (a, b, d = 2) => (a === b ? fmt(a, d) : `${fmt(a, d)} / ${fmt(b, d)}`);
  const signed = (x) => `${x > 0 ? "+" : ""}${fmt(x, 2)}`;
  return el("div", { class: "scroll" }, el("table", { class: "keytable" },
    el("caption", { text: "Test RMSE of SOH (pp) on S6. Proposed model: mean of 50 runs, main / replicate seed sets; others: single deterministic fits. Lower is better. Same numbers as the paper's headline table." }),
    el("thead", {},
      el("tr", {}, el("th", { rowspan: 2, text: "Method" }), el("th", { colspan: 2, class: "c", text: "Continual stream (P1)" }), el("th", { colspan: 2, class: "c", text: "Unseen usage condition (P2)" })),
      el("tr", {}, COLS.map(([, s]) => el("th", { class: "r", text: s.slice(4, -1) })))),
    el("tbody", {},
      methods.map((m) => el("tr", { class: m.includes("tft") ? "proposed" : null }, el("th", { scope: "row", text: label(m) }),
        COLS.map((c) => { const r = cell(m, c); return el("td", { class: "r", text: r ? pair(r.rmse_main, r.rmse_replicate) : "–" }); }))),
      el("tr", { class: "group" }, el("th", { colspan: 5, text: "Proposed minus baseline (pp, main / replicate; negative favours the proposed model)" })),
      diffs.map((m) => el("tr", {}, el("th", { scope: "row", text: label(m).replace("Proposed vs ", "vs ") }),
        COLS.map((c) => { const r = cell(m, c); return el("td", { class: "r" }, r ? `${signed(r.diff_main)} / ${signed(r.diff_replicate)}` : "–", r ? verdict(r.verdict) : null); }))))));
}

export function renderKeyResults(root, { summary }) {
  const section = (id, title, lede, ...kids) => el("section", { class: "sect", id: `s-${id}`, "aria-labelledby": `h-${id}` },
    el("h2", { id: `h-${id}`, text: title }), lede ? el("p", { class: "lede", text: lede }) : null, ...kids);
  root.replaceChildren(
    el("header", { class: "pagehead" }, el("h1", { text: "Key results" }),
      el("p", { class: "deck", text: "The results reported in the paper, each with its number, its verdict under the replication rule and where to see it. A difference counts only if it holds in both seed sets (p_Holm < 0.05, same sign, at least 0.05 pp)." })),
    section("claims", "What the paper claims", null,
      el("ol", { class: "claims" }, CLAIMS.map(([n, u, t, d, v, status, href, where], i) => el("li", { class: "claim" },
        el("div", { class: "claim-num" }, el("span", { class: "idx", text: String(i + 1).padStart(2, "0") }), el("b", { text: n }), el("small", { text: u })),
        el("div", { class: "claim-body" }, el("h3", { text: t }), el("p", { text: d }),
          el("p", { class: "claim-meta" }, v ? verdict(v) : null, el("span", { text: status }), el("a", { href, text: `See: ${where} →` }))))))),
    section("headline", "Headline table", "The proposed model against the strongest traditional forecasters on S6, in both settings, read from the committed results.",
      headlineTable(summary.headline)),
    section("limits", "Where it does not win", "Reported in the paper beside the claims above.",
      el("ol", { class: "findings" }, LIMITS.map(([t, d]) => el("li", {}, el("b", { text: t }), el("span", { text: d }))))),
    section("figures", "Figures in the paper", "The dashboard views as they appear in the manuscript.",
      el("div", { class: "keyfigs" }, FIGS.map(([src, alt, cap]) => el("figure", { class: "pfig" },
        el("img", { src: `${PATHS.figures}${src}`, alt, loading: "lazy" }), el("figcaption", { text: cap }))))));
}
