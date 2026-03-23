"""Generate paper-local figures and tables for the ALIFE 2026 submission."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train_parameter_predictor import FEATURE_COLUMNS, TARGET_PARAMS

FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

PARAM_ORDER = ["tau_v", "tau_w", "a", "b", "v_scale", "R_off", "R_on", "alpha"]
PARAM_LABELS = {
    "tau_v": r"$\tau_v$",
    "tau_w": r"$\tau_w$",
    "a": r"$a$",
    "b": r"$b$",
    "v_scale": r"$v_{scale}$",
    "R_off": r"$R_{off}$",
    "R_on": r"$R_{on}$",
    "alpha": r"$\alpha$",
}


def copy_existing_figures():
    """Copy reusable figures into the paper folder for stable LaTeX paths."""
    mapping = {
        ROOT / "optimization_study_results" / "fig1_viable_parameter_range.png":
            FIG_DIR / "fig_viable_range.png",
        ROOT / "figures" / "characterization_protocols_figure.pdf":
            FIG_DIR / "fig_characterization_protocols.pdf",
        ROOT / "ablation_study_results" / "fig6_protocol_ablation.png":
            FIG_DIR / "fig_protocol_ablation.png",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)


def generate_prediction_summary():
    """Create a compact figure for predictor quality and protocol ablation."""
    eval_path = ROOT / "ml_models" / "evaluation_results_20260219_232802.csv"
    ablation_path = ROOT / "ablation_study_results" / "ablation_r2.csv"

    eval_df = pd.read_csv(eval_path)
    eval_df = eval_df[eval_df["parameter"].isin(PARAM_ORDER)].copy()
    eval_df["parameter"] = pd.Categorical(eval_df["parameter"], PARAM_ORDER, ordered=True)
    eval_df = eval_df.sort_values(["parameter", "model"])

    ablation_df = pd.read_csv(ablation_path, index_col=0)
    ablation_df = ablation_df[PARAM_ORDER]
    ablation_df.columns = [PARAM_LABELS[p] for p in PARAM_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.05, 1.1]})

    ax = axes[0]
    plot_df = eval_df.copy()
    plot_df["label"] = plot_df["parameter"].map(PARAM_LABELS)
    sns.barplot(
        data=plot_df,
        x="label",
        y="r2",
        hue="model",
        palette={"random_forest": "#4c72b0", "mlp": "#dd8452"},
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Parameter Prediction Accuracy", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(r"$R^2$")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")

    ax = axes[1]
    row_labels = {
        "step_only": "Step",
        "pp_only": "Paired-pulse",
        "tri_only": "Triangle",
        "step_pp": "Step+PP",
        "step_tri": "Step+Tri",
        "pp_tri": "PP+Tri",
        "all": "All",
    }
    heatmap_df = ablation_df.copy()
    heatmap_df.index = [row_labels.get(i, i) for i in heatmap_df.index]
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        vmin=-0.25,
        vmax=0.9,
        linewidths=0.5,
        cbar_kws={"label": r"$R^2$"},
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Protocol Ablation", fontweight="bold")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Feature subset")

    plt.suptitle("Predictor Performance and Protocol Contributions", fontweight="bold")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIG_DIR / f"fig_prediction_ablation.{ext}", dpi=200, bbox_inches="tight")
    plt.close()


def generate_reduced_placeholder():
    """Create a placeholder figure for reduced rediscovery if results are unavailable."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.60,
        "Reduced rediscovery results pending",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.38,
        "Run reduced_rediscovery_study.py to populate\nwaveform-mismatch and parameter-error summaries.",
        ha="center",
        va="center",
        fontsize=11,
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIG_DIR / f"fig_reduced_rediscovery_placeholder.{ext}", dpi=200, bbox_inches="tight")
    plt.close()


def write_tables():
    """Write compact LaTeX tables from the current results."""
    eval_df = pd.read_csv(ROOT / "ml_models" / "evaluation_results_20260219_232802.csv")
    eval_df = eval_df[
        (eval_df["model"] == "random_forest") & (eval_df["parameter"].isin(PARAM_ORDER))
    ].copy()
    eval_df["parameter"] = pd.Categorical(eval_df["parameter"], PARAM_ORDER, ordered=True)
    eval_df = eval_df.sort_values("parameter")

    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Parameter & $R^2$ & RMSE \\",
        r"\midrule",
    ]
    for _, row in eval_df.iterrows():
        lines.append(
            f"{PARAM_LABELS[row['parameter']]} & {row['r2']:.3f} & {row['rmse']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TAB_DIR / "ml_results_table.tex").write_text("\n".join(lines))

    # Compare held-out performance overall vs within each parameter's viable interval.
    csv_path = ROOT / "characterization_study_results" / "characterization_results_20260219_145311.csv"
    df = pd.read_csv(csv_path)
    df = df[df["characterization_success"] == True].copy()
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    X = df[available_features].copy().replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_PARAMS].copy().replace([np.inf, -np.inf], np.nan)
    valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_ts = "20260219_232802"
    with open(ROOT / "optimization_study_results" / "viable_param_ranges.json") as f:
        viable = json.load(f)["param_ranges"]

    rf_rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Parameter & Overall $R^2$ & Overall RMSE & Viable $n$ & Viable $R^2$ & Viable RMSE \\",
        r"\midrule",
    ]
    raw_rows = []
    for param in PARAM_ORDER:
        model = joblib.load(ROOT / "ml_models" / f"random_forest_{param}_{model_ts}.pkl")
        preds = model.predict(X_test.values)
        true_vals = y_test[param].values
        overall_r2 = float(r2_score(true_vals, preds))
        overall_rmse = float(mean_squared_error(true_vals, preds) ** 0.5)
        lo = viable[param]["viable_low"]
        hi = viable[param]["viable_high"]
        mask = (true_vals >= lo) & (true_vals <= hi)
        viable_n = int(mask.sum())
        if viable_n >= 2:
            viable_r2 = float(r2_score(true_vals[mask], preds[mask]))
            viable_rmse = float(mean_squared_error(true_vals[mask], preds[mask]) ** 0.5)
        else:
            viable_r2 = np.nan
            viable_rmse = np.nan

        rf_rows.append(
            f"{PARAM_LABELS[param]} & {overall_r2:.3f} & {overall_rmse:.3f} & {viable_n} & "
            f"{viable_r2:.3f} & {viable_rmse:.3f} \\\\"
        )
        raw_rows.append(
            {
                "parameter": param,
                "overall_r2": overall_r2,
                "overall_rmse": overall_rmse,
                "viable_n": viable_n,
                "viable_r2": viable_r2,
                "viable_rmse": viable_rmse,
            }
        )
    rf_rows.extend([r"\bottomrule", r"\end{tabular}"])
    (TAB_DIR / "ml_viable_results_table.tex").write_text("\n".join(rf_rows))
    pd.DataFrame(raw_rows).to_csv(TAB_DIR / "ml_viable_results_table.csv", index=False)

    with open(ROOT / "optimization_study_results" / "viable_param_ranges.json") as f:
        viable = json.load(f)
    meta = viable["metadata"]
    summary_lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Optimization trials & {meta['n_total_trials']} \\\\",
        f"Viable trials & {meta['n_viable_trials']} \\\\",
        f"Viable threshold & {meta['score_threshold']:.3f} \\\\",
        f"Viable fraction & {100 * meta['viable_fraction']:.0f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TAB_DIR / "optimization_summary_table.tex").write_text("\n".join(summary_lines))


def main():
    copy_existing_figures()
    generate_prediction_summary()
    generate_reduced_placeholder()
    write_tables()
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TAB_DIR}")


if __name__ == "__main__":
    main()
