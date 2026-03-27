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
import logging
import pickle
import sys
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
from julearn.utils import configure_logging as julearn_configure_logging
from sklearn.linear_model import ElasticNetCV

register_htcondor("INFO")

set_config("disable_x_verbose", True)
set_config("disable_xtypes_verbose", True)
set_config("disable_xtypes_check", True)
set_config("disable_x_check", True)


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )

    julearn_configure_logging(level="INFO")
    return logging.getLogger(__name__)


class StackingTrainer:
    """
    Region-wise stacking ensemble trainer for brain age prediction.

    Based on Cole et al. 2017 methodology:
    - ElasticNetCV for each brain region (Level-0)
    - ElasticNetCV meta-model (Level-1)
    - Nested cross-validation
    """

    def __init__(
        self,
        features_dir: str,
        output_dir: str,
        n_regions: int = 10,
        n_alphas: int = 100,
        seed: int = 42,
    ):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.n_regions = n_regions
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.output_dir)

        # Paper-compliant parameters (Cole et al. 2017)
        self.l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        self.n_alphas = n_alphas
        self.max_iter = 100000
        self.tol = 1e-7
        self.l0_cv = 3  # Inner CV
        self.outer_cv = 5  # Outer CV

        self.logger.info("=" * 70)
        self.logger.info("STACKING TRAINER (HTCondor - No Token)")
        self.logger.info("=" * 70)
        self.logger.info(f"N regions: {self.n_regions}")
        self.logger.info(
            f"ElasticNetCV: n_alphas={self.n_alphas}, l1_ratio={len(self.l1_ratio)} values"
        )
        self.logger.info(f"Inner CV: {self.l0_cv}-fold, Outer CV: {self.outer_cv}-fold")
        self.logger.info("=" * 70)

    def load_data(self):
        """Load preprocessed regional voxel data"""
        self.logger.info("Loading data...")

        try:
            with open(self.features_dir / "regional_voxels.pkl", "rb") as f:
                self.regional_voxels = pickle.load(f)

            with open(self.features_dir / "regional_voxels_metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)

            self.logger.info(
                f"✓ Loaded {self.metadata['n_subjects']} subjects, "
                f"{len(self.regional_voxels)} regions available"
            )

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def prepare_data(self):
        """Prepare DataFrame with features from selected regions"""
        self.logger.info(f"Preparing {self.n_regions} regions...")

        try:
            ages = np.array(self.metadata["ages"])
            subject_ids = self.metadata["subject_ids"]

            self.X_types = {}
            all_data = {}

            for region_id in range(1, self.n_regions + 1):
                voxels = self.regional_voxels[region_id]

                # Create feature names for this region
                feature_names = [
                    f"region_{region_id}_voxel_{i}" for i in range(voxels.shape[1])
                ]

                # Register region for julearn
                self.X_types[f"region_{region_id}"] = feature_names

                # Add to DataFrame
                for j, fname in enumerate(feature_names):
                    all_data[fname] = voxels[:, j]

            # Create DataFrame
            self.data = pd.DataFrame(all_data)
            self.data["subject_id"] = subject_ids
            self.data["age"] = ages

            self.logger.info(f"✓ Data shape: {self.data.shape}")

        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            raise

    def build_model(self):
        """Build stacking ensemble model"""
        self.logger.info("Building stacking model...")

        # Level-0: Region-specific models
        l0_models = []

        for region_id in range(1, self.n_regions + 1):
            model = PipelineCreator(
                problem_type="regression", apply_to=f"region_{region_id}"
            )

            # Keep only this region's features
            model.add("filter_columns", apply_to="*", keep=f"region_{region_id}")

            # Z-score normalization
            model.add("zscore", apply_to=f"region_{region_id}")

            # ElasticNetCV with grid search
            model.add(
                ElasticNetCV(
                    l1_ratio=self.l1_ratio,
                    n_alphas=self.n_alphas,
                    cv=self.l0_cv,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.seed,
                    n_jobs=1,  # CRITICAL: Must be 1 to avoid job explosion
                ),
                apply_to=f"region_{region_id}",
            )

            l0_models.append((f"region_{region_id}", model))

        # Level-1: Meta-model
        l1_meta = PipelineCreator(problem_type="regression")
        l1_meta.add("zscore", apply_to="*")
        l1_meta.add(
            ElasticNetCV(
                l1_ratio=self.l1_ratio,
                n_alphas=self.n_alphas,
                cv=self.l0_cv,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.seed,
                n_jobs=1,
            )
        )

        # Stacking ensemble
        stacking = PipelineCreator(problem_type="regression")
        stacking.add(
            "stacking",
            estimators=[l0_models],
            final_estimator=l1_meta,
            cv=self.l0_cv,
            apply_to="*",
            n_jobs=-1,
        )

        self.logger.info(f"✓ Stacking model built: {len(l0_models)} Level-0 models")
        return stacking

    def train(self, model):
        """Train model using HTCondor parallelization"""
        self.logger.info("Training with HTCondor backend...")

        try:
            # Collect all features
            all_features = []
            for features in self.X_types.values():
                all_features.extend(features)

            self.logger.info(
                f"Features: {len(all_features):,}, Subjects: {len(self.data)}"
            )
            self.logger.info("Submitting to HTCondor...")

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
                request_memory="4GB",  # Memory per job
                request_disk="5GB",  # Scratch disk per job
                shared_data_dir="/data/group/appliedml/fkarateke_joblib_htcondor",  # NFS shared directory
                pool="head2.htc.inm7.de",  # HTCondor scheduler
                # max_recursion_level NOT set (defaults to 0 - no recursion)
                # Token required for max_recursion_level=1
                max_recursion_level=1,  # Outer CV + Stacking
                export_metadata=True,  # to visualize progress
                throttle=[6, 40]  # Throttle levels
            ):
                scores, trained_model = run_cross_validation(
                    X=all_features,
                    X_types=self.X_types,
                    y="age",
                    data=self.data,
                    model=model,
                    cv=self.outer_cv,
                    scoring=["neg_mean_absolute_error", "r2", "neg_mean_squared_error"],
                    return_estimator="final",
                    seed=self.seed,
                )

            elapsed = time.time() - start_time

            # Calculate metrics
            mae = -scores["test_neg_mean_absolute_error"].mean()
            r2 = scores["test_r2"].mean()
            rmse = np.sqrt(-scores["test_neg_mean_squared_error"].mean())

            self.logger.info("=" * 70)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info(f"MAE: {mae:.3f} years")
            self.logger.info(f"R²: {r2:.4f}")
            self.logger.info(f"RMSE: {rmse:.3f} years")
            self.logger.info(f"Training time: {elapsed / 60:.1f} minutes")
            self.logger.info("=" * 70)

            # Save results
            self._save_results(trained_model, scores, mae, r2, rmse, elapsed)

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

    def _save_results(self, model, scores, mae, r2, rmse, elapsed):
        """Save trained model and results"""
        self.logger.info("Saving results...")

        # Save model
        with open(self.output_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Save scores
        with open(self.output_dir / "scores.pkl", "wb") as f:
            pickle.dump(scores, f)

        # Save summary
        summary = pd.DataFrame(
            [
                {
                    "n_regions": self.n_regions,
                    "mae": mae,
                    "r2": r2,
                    "rmse": rmse,
                    "training_time_min": elapsed / 60,
                }
            ]
        )
        summary.to_csv(self.output_dir / "summary.csv", index=False)

        self.logger.info("✓ Results saved")

    def run(self):
        """Execute full training pipeline"""
        self.load_data()
        self.prepare_data()
        model = self.build_model()
        self.train(model)

        self.logger.info("=" * 70)
        self.logger.info("ALL DONE!")
        self.logger.info("=" * 70)


def main():
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

    trainer = StackingTrainer(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        n_regions=args.n_regions,
        n_alphas=args.n_alphas,
        seed=args.seed,
    )

    success = trainer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
