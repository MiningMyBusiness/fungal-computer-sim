"""Reduced rediscovery study for ALIFE: parameter + waveform recovery only.

This script evaluates a smaller, safer validation slice of the full digital-twin
pipeline on pre-optimized specimens from systematic_optimization_study.py.

It compares three conditions:
  - oracle:    true tuned parameters (upper bound)
  - ml_only:   direct ML prediction from characterization features
  - ml_refine: ML prediction followed by waveform-matching refinement

Unlike batch_rediscovery_study_from_opt.py, this script does not report XOR
transfer metrics. It focuses only on:
  - per-parameter absolute/relative error
  - waveform mismatch before/after refinement

This is intended for a reduced ALIFE validation section.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from realistic_sim import RealisticFungalComputer
from rediscover_fungal_parameters import (
    FUNGAL_PARAMS,
    characterize_specimen,
    collect_response_waveforms,
    compute_waveform_mismatch,
    create_twin,
    load_models,
    predict_parameters,
    refine_parameters_optimization,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


OUTPUT_DIR = Path("reduced_rediscovery_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_OPT_RESULTS_DIR = Path("optimization_study_results")
DEFAULT_MODEL_DIR = Path("ml_models")
DEFAULT_MODEL_TYPE = "random_forest"
DEFAULT_OPT_METHOD = "dual_annealing"
DEFAULT_SCORE_PERCENTILE = 75
DEFAULT_CONDITIONS = ["oracle", "ml_only", "ml_refine"]
CORE_IDENTIFIABLE_PARAMS = ["tau_v", "tau_w", "a", "b", "alpha"]
DEFAULT_MAX_NODES = 50


def find_latest_opt_results() -> Optional[Path]:
    """Return the most recent optimization_results_*.csv, or None."""
    candidates = sorted(
        DEFAULT_OPT_RESULTS_DIR.glob("optimization_results_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_optimized_specimens(
    opt_results_path: Path,
    score_percentile: int = DEFAULT_SCORE_PERCENTILE,
    max_specimens: Optional[int] = None,
    max_nodes: Optional[int] = DEFAULT_MAX_NODES,
) -> pd.DataFrame:
    """Load top-scoring pre-optimized specimens from an optimization study CSV."""
    df = pd.read_csv(opt_results_path)
    df = df[(df["success"] == True) & df["tuned_score"].notna()].copy()
    if len(df) == 0:
        return df

    if max_nodes is not None:
        before = len(df)
        df = df[df["num_nodes"] <= max_nodes].copy()
        logger.info(
            "Applied node-count filter: kept %d/%d specimens with num_nodes <= %d",
            len(df),
            before,
            max_nodes,
        )
        if len(df) == 0:
            return df

    threshold = df["tuned_score"].quantile(score_percentile / 100.0)
    viable = df[df["tuned_score"] >= threshold].sort_values(
        "tuned_score", ascending=False
    )
    if max_specimens is not None:
        viable = viable.head(max_specimens)

    logger.info(
        "Loaded %d of %d successful specimens at or above p%d (tuned_score >= %.4f)",
        len(viable),
        len(df),
        score_percentile,
        threshold,
    )
    return viable.reset_index(drop=True)


def compute_error_metrics(true_params: Dict, est_params: Dict) -> Dict:
    """Compute per-parameter abs/rel error and aggregate summaries."""
    metrics = {}
    rel_errors_all = []
    rel_errors_core = []

    for param in FUNGAL_PARAMS:
        true_val = float(true_params[param])
        est_val = float(est_params[param])
        abs_err = abs(est_val - true_val)
        rel_err = abs_err / (abs(true_val) + 1e-10) * 100.0
        metrics[f"abs_err_{param}"] = abs_err
        metrics[f"rel_err_{param}"] = rel_err
        rel_errors_all.append(rel_err)
        if param in CORE_IDENTIFIABLE_PARAMS:
            rel_errors_core.append(rel_err)

    metrics["mean_rel_err_all"] = float(np.mean(rel_errors_all))
    metrics["median_rel_err_all"] = float(np.median(rel_errors_all))
    metrics["mean_rel_err_core"] = float(np.mean(rel_errors_core))
    return metrics


def _run_specimen_worker(args: Dict) -> List[Dict]:
    """Run all reduced-study conditions for a single optimized specimen."""
    row = args["specimen_row_dict"]
    conditions = args["conditions"]
    model_dir = Path(args["model_dir"])
    model_type = args["model_type"]
    opt_method = args["opt_method"]
    specimen_idx = args["specimen_idx"]

    num_nodes = int(row["num_nodes"])
    seed = int(row["random_state"])

    true_params = {
        "tau_v": float(row["tuned_tau_v"]),
        "tau_w": float(row["tuned_tau_w"]),
        "a": float(row["tuned_a"]),
        "b": float(row["tuned_b"]),
        "v_scale": float(row["tuned_v_scale"]),
        "R_off": float(row["tuned_R_off"]),
        "R_on": float(row["tuned_R_on"]),
        "alpha": float(row["tuned_alpha"]),
    }

    shared_meta = {
        "specimen_idx": specimen_idx,
        "specimen_seed": seed,
        "num_nodes": num_nodes,
        "num_edges": int(row.get("num_edges", 0)),
        "network_density": float(row.get("network_density", np.nan)),
        "tuned_score": float(row["tuned_score"]),
    }
    for param, val in true_params.items():
        shared_meta[f"true_{param}"] = float(val)

    try:
        specimen = RealisticFungalComputer(num_nodes=num_nodes, random_seed=seed)
        for param, val in true_params.items():
            setattr(specimen, param, val)

        features = characterize_specimen(specimen)
        specimen_waveforms = collect_response_waveforms(specimen)
        models, scaler = load_models(model_dir, model_type)
    except Exception as exc:
        return [
            {
                **shared_meta,
                "condition": condition,
                "success": False,
                "error_message": f"Specimen setup failed: {exc}",
            }
            for condition in conditions
        ]

    results = []
    for condition in conditions:
        t0 = time.time()
        record = {**shared_meta, "condition": condition}
        try:
            if condition == "oracle":
                final_params = true_params.copy()
                record["param_source"] = "true"

            elif condition == "ml_only":
                predicted_params, _ = predict_parameters(
                    features, models, scaler, model_type
                )
                final_params = predicted_params
                record["param_source"] = "ml_predicted"
                record.update(compute_error_metrics(true_params, predicted_params))

            elif condition == "ml_refine":
                predicted_params, _ = predict_parameters(
                    features, models, scaler, model_type
                )
                record["param_source"] = "ml_refined"

                # Store pre-refinement errors for direct before/after comparison.
                initial_metrics = compute_error_metrics(true_params, predicted_params)
                for key, value in initial_metrics.items():
                    record[f"initial_{key}"] = value

                twin_ml = create_twin(specimen, predicted_params)
                waveforms_ml = collect_response_waveforms(twin_ml)
                record["waveform_mismatch_before"] = float(
                    compute_waveform_mismatch(specimen_waveforms, waveforms_ml)
                )

                refined_params, opt_info = refine_parameters_optimization(
                    specimen,
                    specimen_waveforms,
                    predicted_params,
                    method=opt_method,
                    use_full_bounds=False,
                )
                final_params = refined_params
                record["opt_success"] = bool(opt_info.get("success", False))
                record["opt_n_evals"] = int(opt_info.get("n_evaluations", 0))
                record["opt_final_mismatch"] = float(
                    opt_info.get("final_mismatch", np.nan)
                )
                record.update(compute_error_metrics(true_params, refined_params))

            else:
                raise ValueError(f"Unknown condition: {condition}")

            twin = create_twin(specimen, final_params)
            twin_waveforms = collect_response_waveforms(twin)
            record["waveform_mismatch"] = float(
                compute_waveform_mismatch(specimen_waveforms, twin_waveforms)
            )

            if condition == "oracle":
                record.update(compute_error_metrics(true_params, final_params))
                record["waveform_mismatch_before"] = 0.0
            elif condition == "ml_only":
                record["waveform_mismatch_before"] = record["waveform_mismatch"]

            record["success"] = True
            record["error_message"] = None
        except Exception as exc:
            record["success"] = False
            record["error_message"] = str(exc)

        record["duration_seconds"] = time.time() - t0
        results.append(record)

    return results


def save_summary_figure(df: pd.DataFrame, output_dir: Path, timestamp: str):
    """Save a compact figure for the reduced rediscovery section."""
    successful = df[df["success"] == True].copy()
    if len(successful) == 0:
        logger.warning("No successful results to plot.")
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    order = [c for c in DEFAULT_CONDITIONS if c in successful["condition"].unique()]

    # Panel A: waveform mismatch by condition.
    ax = axes[0]
    sns.boxplot(
        data=successful,
        x="condition",
        y="waveform_mismatch",
        order=order,
        palette={"oracle": "#4c72b0", "ml_only": "#dd8452", "ml_refine": "#55a868"},
        ax=ax,
    )
    ax.set_title("Waveform Mismatch by Condition", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Mismatch")

    # Panel B: per-parameter relative error, ml_only vs ml_refine.
    ax = axes[1]
    plot_rows = []
    for condition in ["ml_only", "ml_refine"]:
        sub = successful[successful["condition"] == condition]
        for param in FUNGAL_PARAMS:
            col = f"rel_err_{param}"
            if col not in sub.columns:
                continue
            for value in sub[col].dropna():
                plot_rows.append(
                    {"condition": condition, "parameter": param, "rel_err": value}
                )
    if plot_rows:
        plot_df = pd.DataFrame(plot_rows)
        sns.barplot(
            data=plot_df,
            x="parameter",
            y="rel_err",
            hue="condition",
            estimator=np.mean,
            errorbar="sd",
            palette={"ml_only": "#dd8452", "ml_refine": "#55a868"},
            ax=ax,
        )
        ax.set_title("Relative Parameter Error", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Relative Error (%)")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.set_visible(False)

    plt.suptitle("Reduced Rediscovery Validation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = output_dir / f"reduced_rediscovery_summary_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", fig_path)


def print_summary(df: pd.DataFrame):
    """Print condition-level summary statistics."""
    successful = df[df["success"] == True]
    if len(successful) == 0:
        logger.warning("No successful records to summarise.")
        return

    logger.info("\n%s", "=" * 70)
    logger.info("REDUCED REDISCOVERY SUMMARY")
    logger.info("%s", "=" * 70)
    logger.info(
        "%-12s %-5s %-14s %-16s %-16s",
        "Condition",
        "N",
        "Mismatch",
        "Mean Rel Err",
        "Core Rel Err",
    )
    logger.info("%s", "-" * 70)

    for condition in DEFAULT_CONDITIONS:
        sub = successful[successful["condition"] == condition]
        if len(sub) == 0:
            continue
        logger.info(
            "%-12s %-5d %-14.4f %-16.2f %-16.2f",
            condition,
            len(sub),
            sub["waveform_mismatch"].mean(),
            sub["mean_rel_err_all"].mean(),
            sub["mean_rel_err_core"].mean(),
        )
        if condition == "ml_refine":
            logger.info(
                "  refinement gain: mismatch %.4f -> %.4f",
                sub["waveform_mismatch_before"].mean(),
                sub["waveform_mismatch"].mean(),
            )


def run_reduced_rediscovery(
    opt_results_path: Path,
    score_percentile: int = DEFAULT_SCORE_PERCENTILE,
    n_specimens: int = 20,
    max_nodes: Optional[int] = DEFAULT_MAX_NODES,
    conditions: Optional[List[str]] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    model_type: str = DEFAULT_MODEL_TYPE,
    opt_method: str = DEFAULT_OPT_METHOD,
    resume: bool = False,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Run the reduced parameter/waveform rediscovery study."""
    if conditions is None:
        conditions = DEFAULT_CONDITIONS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"reduced_rediscovery_{timestamp}.csv"
    checkpoint_file = OUTPUT_DIR / f"checkpoint_{timestamp}.csv"
    config_file = OUTPUT_DIR / f"config_{timestamp}.json"

    all_results = []
    completed_keys = set()

    if resume:
        checkpoints = sorted(
            OUTPUT_DIR.glob("checkpoint_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            checkpoint_file = checkpoints[0]
            timestamp = checkpoint_file.stem.replace("checkpoint_", "")
            results_file = OUTPUT_DIR / f"reduced_rediscovery_{timestamp}.csv"
            config_file = OUTPUT_DIR / f"config_{timestamp}.json"
            if config_file.exists():
                with open(config_file) as f:
                    prev_config = json.load(f)
                prev_max_nodes = prev_config.get("max_nodes")
                if prev_max_nodes != max_nodes:
                    raise ValueError(
                        "Checkpoint configuration mismatch: "
                        f"checkpoint max_nodes={prev_max_nodes}, "
                        f"current max_nodes={max_nodes}. "
                        "Start a new run or use the same node filter."
                    )
            prev = pd.read_csv(checkpoint_file)
            all_results = prev.to_dict("records")
            for row in all_results:
                if row.get("success", False):
                    completed_keys.add((int(row["specimen_idx"]), row["condition"]))
            logger.info(
                "Resuming from %s with %d completed records",
                checkpoint_file.name,
                len(completed_keys),
            )

    specimens_df = load_optimized_specimens(
        opt_results_path,
        score_percentile=score_percentile,
        max_specimens=n_specimens,
        max_nodes=max_nodes,
    )
    if len(specimens_df) == 0:
        raise ValueError("No optimized specimens available for reduced rediscovery.")

    config = {
        "opt_results_path": str(opt_results_path),
        "score_percentile": score_percentile,
        "n_specimens": len(specimens_df),
        "max_nodes": max_nodes,
        "conditions": conditions,
        "model_dir": str(model_dir),
        "model_type": model_type,
        "opt_method": opt_method,
        "timestamp": timestamp,
    }
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    pending_tasks = []
    for specimen_idx, row in specimens_df.iterrows():
        remaining_conditions = [
            condition
            for condition in conditions
            if (int(specimen_idx), condition) not in completed_keys
        ]
        if not remaining_conditions:
            continue
        pending_tasks.append(
            {
                "specimen_row_dict": row.to_dict(),
                "conditions": remaining_conditions,
                "model_dir": str(model_dir),
                "model_type": model_type,
                "opt_method": opt_method,
                "specimen_idx": int(specimen_idx),
            }
        )

    logger.info("%s", "=" * 70)
    logger.info("REDUCED REDISCOVERY STUDY")
    logger.info("%s", "=" * 70)
    logger.info("Opt results: %s", opt_results_path)
    logger.info("Specimens: %d", len(specimens_df))
    logger.info("Max nodes: %s", max_nodes if max_nodes is not None else "none")
    logger.info("Conditions: %s", conditions)
    logger.info("Workers: %d", n_workers)

    study_start = time.time()
    completed_specimens = 0

    def handle_results(cond_results: List[Dict]):
        nonlocal completed_specimens
        completed_specimens += 1
        for metrics in cond_results:
            all_results.append(metrics)
            if metrics.get("success"):
                completed_keys.add((int(metrics["specimen_idx"]), metrics["condition"]))
        pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
        elapsed = time.time() - study_start
        avg = elapsed / completed_specimens if completed_specimens else 0.0
        eta = avg * (len(specimens_df) - completed_specimens)
        logger.info(
            "  specimen %d/%d done | elapsed=%.1fmin ETA=%.1fmin",
            completed_specimens,
            len(specimens_df),
            elapsed / 60.0,
            eta / 60.0,
        )

    if n_workers <= 1:
        for task in pending_tasks:
            handle_results(_run_specimen_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(_run_specimen_worker, task): task
                for task in pending_tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    cond_results = future.result()
                except Exception as exc:
                    cond_results = [
                        {
                            "specimen_idx": task["specimen_idx"],
                            "condition": condition,
                            "success": False,
                            "error_message": str(exc),
                        }
                        for condition in task["conditions"]
                    ]
                handle_results(cond_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_file, index=False)
    save_summary_figure(results_df, OUTPUT_DIR, timestamp)
    print_summary(results_df)
    logger.info(
        "Finished in %.1f min | results: %s",
        (time.time() - study_start) / 60.0,
        results_file,
    )
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Reduced parameter/waveform rediscovery study for ALIFE",
    )
    parser.add_argument(
        "--opt-results",
        type=Path,
        default=None,
        help="Path to optimization_results_*.csv (defaults to latest)",
    )
    parser.add_argument(
        "--score-percentile",
        type=int,
        default=DEFAULT_SCORE_PERCENTILE,
        help="Keep specimens at or above this tuned_score percentile",
    )
    parser.add_argument(
        "--n-specimens",
        type=int,
        default=20,
        help="Maximum number of top optimized specimens to evaluate",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=DEFAULT_MAX_NODES,
        help="Only evaluate specimens with num_nodes <= this value "
             f"(default: {DEFAULT_MAX_NODES})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing trained ML models",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["random_forest", "mlp"],
        help="Model family to use",
    )
    parser.add_argument(
        "--opt-method",
        type=str,
        default=DEFAULT_OPT_METHOD,
        help="Optimizer used for waveform refinement",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in reduced_rediscovery_results",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    args = parser.parse_args()

    opt_results_path = args.opt_results or find_latest_opt_results()
    if opt_results_path is None or not opt_results_path.exists():
        raise FileNotFoundError(
            "Could not find optimization results. Run systematic_optimization_study.py first."
        )

    run_reduced_rediscovery(
        opt_results_path=opt_results_path,
        score_percentile=args.score_percentile,
        n_specimens=args.n_specimens,
        max_nodes=args.max_nodes,
        model_dir=args.model_dir,
        model_type=args.model_type,
        opt_method=args.opt_method,
        resume=args.resume,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
