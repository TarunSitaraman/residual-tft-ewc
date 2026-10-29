"""Training curves of the residual TFT + EWC on the S6 stream (D23 E1 set-up, first evaluation order, seed 0).

Re-runs one committed D23 stream with the tuned learning rate and EWC lambda, records the mean training loss
(MSE on the scaled target plus the EWC penalty) and the validation RMSE (pp) after every epoch of every task,
and checks that the run reproduces the committed final average error of that stream. Illustration of training
only; it adds no result. Writes results/epoch_curves_d23_e1.json and paper/figures/epoch_curves.png.

    python scripts/run_epoch_curves.py          # run, check and plot
    python scripts/run_epoch_curves.py --plot   # replot from the JSON
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_d23  # noqa: E402
from src import forecasting as FC  # noqa: E402
from src import p3  # noqa: E402
from src.data_pipeline import environment_info  # noqa: E402

ORDER, SEED, TOL = 0, 0, 1e-6


def main() -> int:
    committed = json.loads((ROOT / "results" / "forecasting_d23_e1.json").read_text())
    tuned = committed["tuning"]["tft_residual"]
    r = run_d23.Runner("e1", "paper")
    order = r.orders[ORDER]
    frame = r.frames["residual"]
    s = p3.build_strategy("tft_ewc", r.features, SEED, frame,
                          r.settings("tft", tuned["lr"], {"ewc_lambda": tuned["ewc_lambda"]}))
    out = p3.run_stream(s, FC.forecast_stream(frame, order), None, eval_split="test")

    ref = [x for x in committed["runs"] if x["method"] == run_d23.REFERENCE and x["order_id"] == ORDER
           and x["seed"] == SEED and x["set"] == "main"]
    got, want = out["metrics"]["final_avg_error"], ref[0]["final_avg_error"]
    print(f"final average error {got:.6f} pp, committed {want:.6f} pp")
    if abs(got - want) > TOL:
        print("does not reproduce the committed run; not writing outputs")
        return 1

    tasks = [{"task": t, "best_epoch": int(b), "history": [{"train_loss": a, "val_rmse": v} for a, v in h]}
             for t, b, h in zip(s.tasks_seen_, s.best_epoch_, s.epoch_history_)]
    (ROOT / "results" / "epoch_curves_d23_e1.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(), "environment": environment_info(),
        "source": "results/forecasting_d23_e1.json", "method": run_d23.REFERENCE, "order": order, "seed": SEED,
        "lr": tuned["lr"], "ewc_lambda": tuned["ewc_lambda"], "final_avg_error": got, "tasks": tasks}, indent=1))

    plot()
    print("wrote results/epoch_curves_d23_e1.json and paper/figures/epoch_curves.png")
    return 0


def plot() -> None:
    """Seven panels of validation RMSE per epoch (one per task) and one panel of all training losses."""
    d = json.loads((ROOT / "results" / "epoch_curves_d23_e1.json").read_text())
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.4))
    loss_ax = axes.flat[-1]
    for k, (t, ax) in enumerate(zip(d["tasks"], axes.flat)):
        v = np.array([e["val_rmse"] for e in t["history"]])
        a = np.array([e["train_loss"] for e in t["history"]])
        ep = np.arange(1, len(v) + 1)
        c = plt.cm.viridis(k / (len(d["tasks"]) - 1))
        ax.plot(ep, v, color=c, lw=1.1)
        if t["best_epoch"] >= 0:
            ax.axvline(t["best_epoch"] + 1, color="k", ls=":", lw=0.8)
        loss_ax.plot(ep, a, color=c, lw=1.0, label=str(k + 1))
        ax.set_title(f"task {k + 1} ({t['task']})", fontsize=7.5)
        if k % 4 == 0:
            ax.set_ylabel("val. RMSE (pp)", fontsize=7)
    loss_ax.set_yscale("log")
    loss_ax.set_title("training loss, all tasks", fontsize=7.5)
    loss_ax.legend(title="task", fontsize=5.5, title_fontsize=6, ncol=2, frameon=False)
    for k, ax in enumerate(axes.flat):
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.3)
        if k >= 4:
            ax.set_xlabel("epoch", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "paper" / "figures" / "epoch_curves.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
        raise SystemExit(0)
    raise SystemExit(main())
