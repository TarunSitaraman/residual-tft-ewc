# Results dashboard

An interactive view of the project's findings: SOH trajectories, forecast accuracy against baselines, continual-learning adaptation and retention, reuse classes and interval coverage. It has two layers:

- **Tableau Public workbook**: the analytical views, embedded with the Tableau Embedding API v3. Specification: [`tableau/WORKBOOK_SPEC.md`](tableau/WORKBOOK_SPEC.md).
- **JS page** (`index.html`, `src/`, no build step): five sections in a top bar, one filter strip, a details panel, and native SVG charts on every section, captioned and numbered like the paper's figures, drawn from `results/dashboard/summary.json` and the CSVs in `results/dashboard/`. Each section opens with a one-sentence finding computed from the data (the verdict wording comes from the stored replication verdicts). The SOH trajectories (`data/dashboard/fact_soh.csv`) and the reuse-class confusion and parity plots (`data/dashboard/fact_predictions.csv`) need the local data folder; without it those cards say how to build them. When a section has a published Tableau view, it is embedded below the native charts and follows the same filters. The page is complete without the workbook.

The data-platform page (`scripts/platform_server.py`, `docs/spark_dashboard.html`) remains the place for datasets, lineage and Spark runs. This dashboard covers results only.

## Why Tableau Public

After export the data are small: about 850 S6 checks, 111 S1 RPTs, 38.5k saved forecasts and 42k run × task rows. They are public (CC BY 4.0), and the repository is public. Tableau Public is free to publish, runs on Windows and macOS, and its Embedding API lets this page set filters and read mark selections. Power BI's JavaScript embedding needs paid Pro or Embedded capacity and Azure AD; its free "publish to web" is an iframe with no API, and Power BI Desktop runs only on Windows. The trade-off: Tableau Public views are public and use extracts, with no live connection.

## Steps

1. **Export the tables** (from the repository root):
   ```bash
   python scripts/export_dashboard_data.py
   ```
   This writes `results/dashboard/*.csv`, `summary.json` and `manifest.json` (source hashes, commit, row counts). With the data folder present (`data/README.md`), it also writes `data/dashboard/fact_soh.csv` and `fact_predictions.csv`. Without the data folder, those two are skipped with a message. Re-run after any new results file.
2. **Build the Tableau kit**:
   ```bash
   python scripts/build_tableau_workbook.py
   ```
   This writes `data/dashboard/tableau/` (git-ignored). It contains `tableau_source.xlsx`, one self-contained, pre-filtered sheet per view with the filter fields already named as the page expects, and `residual_tft_ewc.twb`, a generated starter workbook with the data sources, calculated fields, 12 worksheets and the five named dashboards. It also writes `AUTHORING.md`. Views whose inputs are missing (SOH, forecasts) get empty placeholder sheets; re-run after building the data.
3. **Author and publish** in Tableau Desktop Public Edition. The `.twb` is generated and has never been opened by Tableau before you open it, so check every view against `tableau/WORKBOOK_SPEC.md` and fix what Tableau reports. Then save it to Tableau Public. `AUTHORING.md` lists the steps.
4. **Configure** the five dashboard URLs in `config.js` (`TABLEAU.views`).
5. **Serve** the page from the repository root, because it reads `../results/dashboard/` and `../paper/figures/`:
   ```bash
   python -m http.server 8000
   ```
   Then open <http://localhost:8000/dashboard/>. It can also be hosted on GitHub Pages from the repository root.

## How the page works

- `config.js`: Tableau view URLs, the field names the filters drive (the contract in the workbook spec), and paths.
- `src/main.js`: state (kept in the URL hash, so views can be shared and the back button works), sections, filters, the SOH section, and the optional Tableau card of each section.
- `src/tabs.js`: the Overview headline chart and finding, and the Forecast accuracy, Continual learning and Reuse & uncertainty sections.
- `src/charts.js`: chart primitives (line chart, dot rows, scatter, matrix, table view) following the validated reference palette: categorical colours in fixed order, one-hue sequential blue, blue-grey-red diverging.
- `src/embed.js`: loads the Embedding API once, mounts `<tableau-viz>`, applies filters to every worksheet, and reads mark selections.
- `src/views.js`: KPI tiles, the headline table with verdicts, caveats and the details panel.
- `src/soh.js`: the SOH trajectory chart (one line per cell, reuse-class boundaries, hover, legend selection, table view).
- `styles.css`: design tokens for light and dark themes (each selected, not inverted); the theme button in the top bar cycles auto, light and dark.

Filters: Dataset, Setting (E1 h = 1–3, E2 h = 1–6), Protocol (P1 continual stream, P2 unseen usage condition), Group and Method family. Selecting a heatmap cell or a Tableau mark opens the details panel: the group's condition, every method's error with its seed spread, and a link to the result in `docs/RESULTS.md`.

## Verify

```bash
python -m pytest tests/test_export_dashboard_data.py
```

Then serve the page and check, at desktop and phone widths in both themes:

- every section renders its charts with no console errors, in both themes;
- the numbers match `docs/RESULTS.md` and paper Table III (linear extrapolation 0.34 pp on the stream; gradient boosting 0.54 pp, joint residual TFT 0.57 pp and the proposed model 0.92 / 0.94 pp on unseen conditions, E1);
- each finding sentence matches the numbers in the charts below it;
- once views are configured, changing Group re-filters every embedded view, and selecting a mark opens the details panel.
