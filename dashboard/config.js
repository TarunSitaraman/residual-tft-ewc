// Dashboard configuration: the Tableau Public views and the field names the page drives.
// Leave a view URL empty until the workbook is published; that tab then shows the paper figures instead.
// Field and parameter names are the contract with the workbook (dashboard/tableau/WORKBOOK_SPEC.md).

export const TABLEAU = {
  // Tableau's own distribution of the Embedding API v3. Tableau Public serves the version that matches its
  // server, which is how Tableau recommends loading it. The npm package @tableau/embedding-api ships no
  // browser bundle.
  scriptUrl: "https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js",
  // Seconds to wait for a view to become interactive before falling back to the static figures.
  timeoutS: 25,
  views: {
    overview: "",
    soh: "",
    accuracy: "",
    continual: "",
    reuse: "",
  },
};

// Filter bar key -> Tableau field name (applied to every worksheet that has the field).
export const FIELDS = {
  dataset: "Dataset",
  setting: "Setting",
  protocol: "Protocol",
  group: "Group",
  family: "Method Family",
};

// Where the page reads its data and figures, relative to dashboard/index.html (serve from the repository root).
export const PATHS = {
  summary: "../results/dashboard/summary.json",
  // Tidy tables behind the native charts (in Git), and the local, git-ignored tables built from the data folder.
  data: "../results/dashboard/",
  local: "../data/dashboard/",
  // Optional, git-ignored: SOH trajectories built from the data folder (scripts/export_dashboard_data.py).
  soh: "../data/dashboard/fact_soh.csv",
  figures: "../paper/figures/",
  repo: "https://github.com/TarunSitaraman/residual-tft-ewc/blob/main/",
  platform: "../docs/spark_dashboard.html",
};
