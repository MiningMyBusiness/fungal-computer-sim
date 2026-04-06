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
REDISCOVERY_DIR = ROOT / "reduced_rediscovery_results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
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
CONDITION_ORDER = ["oracle", "ml_only", "ml_refine"]
CONDITION_LABELS = {
    "oracle": "Oracle",
    "ml_only": "ML only",
    "ml_refine": "ML+refine",
}
CONDITION_COLORS = {
    "oracle": "#4c72b0",
    "ml_only": "#dd8452",
    "ml_refine": "#55a868",
}


def copy_existing_figures():
    """Copy reusable figures into the paper folder for stable LaTeX paths."""
    mapping = {
        ROOT / "optimization_study_results" / "fig1_viable_parameter_range.png":
            FIG_DIR / "fig_viable_range.png",
        ROOT / "figures" / "characterization_protocols_figure.pdf":
            FIG_DIR / "fig_characterization_protocols.pdf",
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

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"width_ratios": [1.05, 1.1]})

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
    ax.set_title("Parameter Prediction Accuracy", fontweight="bold", fontsize=14)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel(r"$R^2$", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(title="", fontsize=10)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    # Clamp y-axis to a readable range and annotate any clipped bars
    Y_MIN, Y_MAX = -1.0, 1.05
    ax.set_ylim(Y_MIN, Y_MAX)
    alpha_label = PARAM_LABELS["alpha"]
    for container in ax.containers:
        for bar in container:
            bar_height = bar.get_height()
            if bar_height < Y_MIN:
                # Bar is clipped — annotate with the actual value
                ax.annotate(
                    f"{bar_height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, Y_MIN + 0.02),
                    ha="center", va="bottom",
                    fontsize=9, color="black",
                    rotation=90,
                )
    # Add a note under alpha's x-tick label to hint at truncation
    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    if alpha_label in x_labels:
        alpha_idx = x_labels.index(alpha_label)
        ax.get_xticklabels()[alpha_idx].set_fontstyle("italic")
    ax.annotate(
        "* axis truncated",
        xy=(1.0, 0.01), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9, color="gray", style="italic",
    )

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
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title("Protocol Ablation", fontweight="bold", fontsize=14)
    ax.set_xlabel("Parameter", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature subset", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(r"$R^2$", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle("Predictor Performance and Protocol Contributions", fontweight="bold", fontsize=16)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIG_DIR / f"fig_prediction_ablation.{ext}", dpi=200, bbox_inches="tight")
    plt.close()


def load_reduced_rediscovery_results(
    exclude_failed_specimens: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all reduced rediscovery checkpoints, keeping the newest duplicate rows."""
    checkpoints = sorted(REDISCOVERY_DIR.glob("checkpoint_*.csv"))
    if not checkpoints:
        return pd.DataFrame(), pd.DataFrame()

    frames = []
    for checkpoint in checkpoints:
        df = pd.read_csv(checkpoint)
        df["source_timestamp"] = checkpoint.stem.replace("checkpoint_", "")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["specimen_seed", "condition", "source_timestamp"])
    merged = merged.drop_duplicates(["specimen_seed", "condition"], keep="last").copy()

    failed_mask = merged["success"].fillna(False) != True
    failed_specimens = set(merged.loc[failed_mask, "specimen_seed"].tolist())

    filtered = merged.copy()
    if exclude_failed_specimens and failed_specimens:
        filtered = filtered[~filtered["specimen_seed"].isin(failed_specimens)].copy()

    return filtered, merged


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


def write_reduced_rediscovery_table(summary_df: pd.DataFrame):
    """Write the reduced rediscovery condition summary table."""
    csv_path = TAB_DIR / "reduced_rediscovery_summary_table.csv"
    summary_df.to_csv(csv_path, index=False)

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & $n$ & Waveform mismatch & Mean rel.\ err.\ (all) & Mean rel.\ err.\ (core) \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['label']} & {int(row['n'])} & "
            f"{row['waveform_mismatch_mean']:.3f} $\\pm$ {row['waveform_mismatch_sd']:.3f} & "
            f"{row['mean_rel_err_all_mean']:.1f} $\\pm$ {row['mean_rel_err_all_sd']:.1f} & "
            f"{row['mean_rel_err_core_mean']:.1f} $\\pm$ {row['mean_rel_err_core_sd']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TAB_DIR / "reduced_rediscovery_summary_table.tex").write_text("\n".join(lines))


def generate_reduced_rediscovery_assets():
    """Create a figure and summary tables for the reduced rediscovery study."""
    successful, merged = load_reduced_rediscovery_results(exclude_failed_specimens=True)
    if successful.empty:
        generate_reduced_placeholder()
        return

    successful = successful[successful["success"] == True].copy()
    successful["condition_label"] = successful["condition"].map(CONDITION_LABELS)

    condition_summary = (
        successful.groupby("condition")
        .agg(
            n=("specimen_seed", "nunique"),
            waveform_mismatch_mean=("waveform_mismatch", "mean"),
            waveform_mismatch_sd=("waveform_mismatch", "std"),
            mean_rel_err_all_mean=("mean_rel_err_all", "mean"),
            mean_rel_err_all_sd=("mean_rel_err_all", "std"),
            mean_rel_err_core_mean=("mean_rel_err_core", "mean"),
            mean_rel_err_core_sd=("mean_rel_err_core", "std"),
            duration_hours_mean=("duration_seconds", lambda s: s.mean() / 3600.0),
        )
        .reset_index()
    )
    condition_summary["condition"] = pd.Categorical(
        condition_summary["condition"], CONDITION_ORDER, ordered=True
    )
    condition_summary = condition_summary.sort_values("condition").reset_index(drop=True)
    condition_summary["label"] = condition_summary["condition"].map(CONDITION_LABELS)
    write_reduced_rediscovery_table(condition_summary)

    wide = successful.pivot(index="specimen_seed", columns="condition")
    paired_metrics = [
        ("waveform_mismatch", "Waveform mismatch"),
        ("mean_rel_err_all", "Mean relative error (all)"),
        ("mean_rel_err_core", "Mean relative error (core)"),
    ]
    stat_rows = []
    for metric, label in paired_metrics:
        ml_only = wide[(metric, "ml_only")].astype(float)
        ml_refine = wide[(metric, "ml_refine")].astype(float)
        stat, pvalue = wilcoxon(ml_only, ml_refine, alternative="greater")
        stat_rows.append(
            {
                "metric": metric,
                "label": label,
                "n": len(ml_only),
                "ml_only_mean": float(ml_only.mean()),
                "ml_refine_mean": float(ml_refine.mean()),
                "mean_delta": float((ml_only - ml_refine).mean()),
                "mean_reduction_pct": float(100.0 * (ml_only.mean() - ml_refine.mean()) / ml_only.mean()),
                "wilcoxon_stat": float(stat),
                "wilcoxon_pvalue": float(pvalue),
            }
        )
    pd.DataFrame(stat_rows).to_csv(TAB_DIR / "reduced_rediscovery_stats.csv", index=False)

    refine_rows = successful[successful["condition"] == "ml_refine"].copy()
    param_rows = []
    for param in PARAM_ORDER:
        init = refine_rows[f"initial_rel_err_{param}"].astype(float)
        final = refine_rows[f"rel_err_{param}"].astype(float)
        stat, pvalue = wilcoxon(init, final, alternative="greater")
        param_rows.append(
            {
                "parameter": param,
                "initial_mean_rel_err": float(init.mean()),
                "final_mean_rel_err": float(final.mean()),
                "mean_delta": float((init - final).mean()),
                "wilcoxon_stat": float(stat),
                "wilcoxon_pvalue": float(pvalue),
            }
        )
    param_stats = pd.DataFrame(param_rows)
    param_stats.to_csv(TAB_DIR / "reduced_rediscovery_parameter_stats.csv", index=False)

    fig = plt.figure(figsize=(13.5, 5.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])

    ax = fig.add_subplot(grid[0, 0])
    order = [c for c in CONDITION_ORDER if c in successful["condition"].unique()]
    sns.boxplot(
        data=successful,
        x="condition",
        y="waveform_mismatch",
        order=order,
        palette=CONDITION_COLORS,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=successful,
        x="condition",
        y="waveform_mismatch",
        order=order,
        color="black",
        alpha=0.75,
        size=4.5,
        jitter=0.12,
        ax=ax,
    )
    ax.set_xticklabels([CONDITION_LABELS[c] for c in order])
    ax.set_xlabel("")
    ax.set_ylabel("Waveform mismatch")
    ax.set_title("A. Waveform matching across conditions", fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    x_max = 1.10 * max(
        param_stats["initial_mean_rel_err"].max(),
        param_stats["final_mean_rel_err"].max(),
    )
    for idx, param in enumerate(PARAM_ORDER):
        row = param_stats[param_stats["parameter"] == param].iloc[0]
        y = len(PARAM_ORDER) - 1 - idx
        ax.plot(
            [row["initial_mean_rel_err"], row["final_mean_rel_err"]],
            [y, y],
            color="#999999",
            linewidth=2.0,
            zorder=1,
        )
        ax.scatter(
            row["initial_mean_rel_err"],
            y,
            color=CONDITION_COLORS["ml_only"],
            s=65,
            zorder=2,
            label="ML only" if idx == 0 else None,
        )
        ax.scatter(
            row["final_mean_rel_err"],
            y,
            color=CONDITION_COLORS["ml_refine"],
            s=65,
            zorder=3,
            label="ML+refine" if idx == 0 else None,
        )
    ax.set_xlim(0, x_max)
    ax.set_yticks(range(len(PARAM_ORDER)))
    ax.set_yticklabels([PARAM_LABELS[p] for p in PARAM_ORDER[::-1]])
    ax.set_xlabel("Mean relative parameter error (%)")
    ax.set_title("B. Parameter error before and after refinement", fontweight="bold")
    ax.legend(frameon=True, loc="lower right")

    plt.suptitle("Reduced Rediscovery Validation", fontweight="bold")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIG_DIR / f"fig_reduced_rediscovery.{ext}", dpi=200, bbox_inches="tight")
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
    generate_reduced_rediscovery_assets()
    write_tables()
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TAB_DIR}")


if __name__ == "__main__":
    main()
