// Screenshot every chart card of the results dashboard (dashboard/index.html) for the paper figures.
//
// Usage (from the repository root, with the site served on port 8765):
//   python -m http.server 8765 --bind 127.0.0.1 &
//   node scripts/capture_dashboard_cards.js <out_dir>
// Then compose the paper figures with scripts/make_dashboard_figures.py <out_dir>.
//
// Light theme, 1400 px viewport (the dashboard's desktop layout), device scale 3. The card captions, "Show table"
// links and the Tableau fallback card are hidden: the paper captions replace them. Charts use the default filters
// (S6, E1 = horizons 1-3) and read only the committed exports in results/dashboard/.
const { chromium } = require('playwright');
const path = require('path');

const OUT = process.argv[2] || 'dashboard_cards';
const URL = process.env.DASHBOARD_URL || 'http://127.0.0.1:8765/dashboard/#page=dashboard';
const PANES = ['overview', 'accuracy', 'continual', 'reuse', 'many', 'd38', 'evidence'];

(async () => {
  const fs = require('fs');
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 3, colorScheme: 'light' });
  await page.addInitScript(() => {
    document.addEventListener('DOMContentLoaded', () => {
      const s = document.createElement('style');
      s.textContent = 'figure.fig figcaption, figure.fig details, figure.fig .show-table, figure.fig button.link'
        + ' { display: none !important; } figure.fig { box-shadow: none !important; }';
      document.head.append(s);
    });
  });
  await page.goto(URL);
  await page.waitForTimeout(800);
  for (const pane of PANES) {
    await page.click('#t-' + pane);
    await page.waitForTimeout(2500);
    // Nudge the viewport so every chart re-lays out after the captions were hidden.
    await page.setViewportSize({ width: 1399, height: 1000 });
    await page.waitForTimeout(300);
    await page.setViewportSize({ width: 1400, height: 1000 });
    await page.waitForTimeout(800);
    // Interactive links ("Show table", "Show every row as a table") mean nothing on paper.
    await page.evaluate(() => document.querySelectorAll('figure.fig a, figure.fig button').forEach((e) => {
      if (/^\s*Show/.test(e.textContent)) e.style.visibility = 'hidden';
    }));
    let i = 0;
    for (const head of await page.$$('#tab-' + pane + ' .fig-head')) {
      const title = await head.evaluate((e) => e.textContent);
      if (/Tableau/.test(title)) continue;
      const card = await head.evaluateHandle((e) => e.parentElement);
      const file = path.join(OUT, `${pane}_${i}.png`);
      await card.screenshot({ path: file });
      console.log(file, '|', title.slice(0, 90));
      i++;
    }
  }
  await browser.close();
})();
