#!/usr/bin/env python3
"""
Brain Age Prediction - HTCondor Stacking Training
Version: 0.2 (Without Token - Basic Parallelization)

This version uses only outer CV parallelization (no recursive).
For recursive parallelization (max_recursion_level=1), token is required.
Author:
    Fatma Karateke
    fatma.karateke@hhu.de
"""

import argparse
# import cProfile
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_config

# HTCondor backend
from joblib_htcondor import register_htcondor
from julearn import run_cross_validation
from julearn.config import set_config
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging, logger
from sklearn.linear_model import ElasticNetCV


def load_data():
    """Load preprocessed regional voxel data"""
    logger.info("Loading data...")
    with open(features_dir / "regional_voxels.pkl", "rb") as f:
        regional_voxels = pickle.load(f)

    with open(features_dir / "regional_voxels_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    logger.info(
        f"✓ Loaded {metadata['n_subjects']} subjects, "
        f"{len(regional_voxels)} regions available"
    )
    return regional_voxels, metadata


def prepare_data(metadata, regional_voxels, n_regions):
    """Prepare DataFrame with features from selected regions"""
    logger.info(f"Preparing {n_regions} regions...")

    ages = np.array(metadata["ages"])
    subject_ids = metadata["subject_ids"]

    X_types = {}
    feature_names = []
    all_data = {}

    for region_id in range(1, n_regions + 1):
        voxels = regional_voxels[region_id]

        # Create feature names for this region
        this_feature_names = [
            f"region_{region_id}_voxel_{i}" for i in range(voxels.shape[1])
        ]
        feature_names.extend(this_feature_names)

        # Register region for julearn
        X_types[f"region_{region_id}"] = f"region_{region_id}_voxel_.*"

        # Add to DataFrame
        for j, fname in enumerate(this_feature_names):
            all_data[fname] = voxels[:, j]

    # Create DataFrame
    data = pd.DataFrame(all_data)
    data["subject_id"] = subject_ids
    data["age"] = ages

    logger.info(f"✓ Data shape: {data.shape}")

    return data, feature_names, X_types


def build_model(n_regions, l1_ratio, n_alphas, l0_cv, max_iter, tol, seed):
    """Build stacking ensemble model"""
    logger.info("Building stacking model...")

    # Level-0: Region-specific models
    l0_models = []

    for region_id in range(1, n_regions + 1):
        model = PipelineCreator(
            problem_type="regression", apply_to=f"region_{region_id}"
        )

        # Keep only this region's features
        model.add("filter_columns", apply_to="*", keep=f"region_{region_id}")

        # Z-score normalization
        model.add("zscore")

        # ElasticNetCV with grid search
        model.add(
            ElasticNetCV(
                l1_ratio=l1_ratio,
                n_alphas=n_alphas,
                cv=l0_cv,
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
                n_jobs=1,  # CRITICAL: Must be 1 to avoid job explosion
            ),
        )

        l0_models.append((f"region_{region_id}", model))

    # Level-1: Meta-model
    l1_meta = PipelineCreator(problem_type="regression")
    l1_meta.add("zscore", apply_to="*")
    l1_meta.add(
        ElasticNetCV(
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            cv=l0_cv,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            n_jobs=1,
        )
    )

    # Stacking ensemble
    stacking = PipelineCreator(problem_type="regression")
    stacking.add(
        "stacking",
        estimators=[l0_models],
        final_estimator=l1_meta,
        cv=l0_cv,
        apply_to="*",
        n_jobs=-1,
    )

    logger.info(f"✓ Stacking model built: {len(l0_models)} Level-0 models")
    return stacking


def _save_results(model, scores, mae, r2, rmse, elapsed):
    """Save trained model and results"""
    logger.info("Saving results...")

    # Save model
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scores
    with open(output_dir / "scores.pkl", "wb") as f:
        pickle.dump(scores, f)

    # Save summary
    summary = pd.DataFrame(
        [
            {
                "n_regions": n_regions,
                "mae": mae,
                "r2": r2,
                "rmse": rmse,
                "training_time_min": elapsed / 60,
            }
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    logger.info("✓ Results saved")


if __name__ == "__main__":
    register_htcondor("INFO")

    set_config("disable_x_verbose", True)
    set_config("disable_xtypes_verbose", True)
    set_config("disable_xtypes_check", True)
    set_config("disable_x_check", True)

    parser = argparse.ArgumentParser(
        description="Brain Age Stacking Training with HTCondor"
    )
    parser.add_argument(
        "features_dir",
        type=str,
        help="Directory containing regional_voxels.pkl and metadata",
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory for model and results"
    )
    parser.add_argument(
        "--n_regions",
        type=int,
        default=10,
        help="Number of brain regions to use (default: 10)",
    )
    parser.add_argument(
        "--n_alphas",
        type=int,
        default=100,
        help="Number of alpha values for ElasticNetCV (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    """Setup logging configuration"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"training_{timestamp}.log"
    configure_logging(level="INFO", fname=log_file)

    # Training parameters
    features_dir = Path(args.features_dir)
    n_regions = args.n_regions
    n_alphas = args.n_alphas
    seed = args.seed

    # Paper-compliant parameters (Cole et al. 2017)
    l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    n_alphas = n_alphas
    max_iter = 100000
    tol = 1e-7
    l0_cv = 3  # Inner CV
    outer_cv = 5  # Outer CV

    logger.info("=" * 70)
    logger.info("STACKING TRAINER (HTCondor - No Token)")
    logger.info("=" * 70)
    logger.info(f"N regions: {n_regions}")
    logger.info(f"ElasticNetCV: n_alphas={n_alphas}, l1_ratio={len(l1_ratio)} values")
    logger.info(f"Inner CV: {l0_cv}-fold, Outer CV: {outer_cv}-fold")
    logger.info("=" * 70)

    regional_voxels, metadata = load_data()
    data, X, X_types = prepare_data(
        metadata=metadata, regional_voxels=regional_voxels, n_regions=n_regions
    )

    model = build_model(
        n_regions=n_regions,
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        l0_cv=l0_cv,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
    )

    # pr = cProfile.Profile()
    # pr.enable()

    start_time = time.time()

    # HTCondor configuration (without token - no recursion)
    # Note: For recursive parallelization, token is required
    # Job size: 1.6Gb
    # With max_recursion_level=1, we would have 6 CV splits (5 outer +
    # final model) * 800 ROIs (max) = 4800 jobs.
    # With a shared dir of 400Gb -> max 250 jobs in parallel
    # (400Gb / 1.6Gb per job).
    # Throttle first level = 6
    # Throttle second level = 250 (shared dir limit) / 6 = ~40
    with parallel_config(
        backend="htcondor",
        n_jobs=-1,  # Maximum parallelization
        request_cpus=1,  # 1 CPU per job (easier slot matching)
        request_memory="8GB",  # Memory per job
        request_disk="5GB",  # Scratch disk per job
        shared_data_dir="/data/group/appliedml/fkarateke_joblib_htcondor",  # NFS shared directory
        pool="head2.htc.inm7.de",  # HTCondor scheduler
        # max_recursion_level NOT set (defaults to 0 - no recursion)
        # Token required for max_recursion_level=1
        max_recursion_level=1,  # Outer CV + Stacking
        export_metadata=True,  # to visualize progress
        throttle=[6, 40],  # Throttle levels
        delete_task_file_on_load=True,  # Free disk space after loading
    ):
        scores, trained_model = run_cross_validation(
            X=X,
            X_types=X_types,
            y="age",
            data=data,
            model=model,
            cv=outer_cv,
            scoring=["neg_mean_absolute_error", "r2", "neg_mean_squared_error"],
            return_estimator="final",
            seed=seed,
        )

    elapsed = time.time() - start_time

    # pr.disable()
    # pr.dump_stats(output_dir / "training_profile.prof")

    # Calculate metrics
    mae = -scores["test_neg_mean_absolute_error"].mean()
    r2 = scores["test_r2"].mean()
    rmse = np.sqrt(-scores["test_neg_mean_squared_error"].mean())

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info(f"MAE: {mae:.3f} years")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.3f} years")
    logger.info(f"Training time: {elapsed / 60:.1f} minutes")
    logger.info("=" * 70)

    # Save results
    _save_results(
        model=trained_model, scores=scores, mae=mae, r2=r2, rmse=rmse, elapsed=elapsed
    )
