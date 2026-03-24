#!/usr/bin/env python3
"""
Brain Age Prediction - HTCondor Joblib Version 0.1
Clean and minimal version for stacking + parallelization
"""

import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# HTCondor backend
from joblib_htcondor import register_htcondor
register_htcondor()

from joblib import parallel_config
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging as julearn_configure_logging
from sklearn.linear_model import ElasticNetCV


# -------------------------
# Logging
# -------------------------
def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )

    julearn_configure_logging(level="WARNING")
    return logging.getLogger(__name__)


# -------------------------
# Trainer
# -------------------------
class StackingTrainer:
    def __init__(self, features_dir, output_dir, n_regions=10, seed=42):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.n_regions = n_regions
        self.seed = seed

        self.logger = setup_logging(self.output_dir)

        # Model params
        self.l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        self.n_alphas = 100
        self.l0_cv = 3
        self.outer_cv = 5

    # -------------------------
    def load_data(self):
        self.logger.info("Loading data...")

        with open(self.features_dir / "regional_voxels.pkl", "rb") as f:
            self.regional_voxels = pickle.load(f)

        with open(self.features_dir / "regional_voxels_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    # -------------------------
    def prepare_data(self):
        self.logger.info(f"Preparing {self.n_regions} regions...")

        ages = np.array(self.metadata["ages"])
        subject_ids = self.metadata["subject_ids"]

        self.X_types = {}
        all_data = {}

        for region_id in range(1, self.n_regions + 1):
            voxels = self.regional_voxels[region_id]

            feature_names = [
                f"region_{region_id}_voxel_{i}" for i in range(voxels.shape[1])
            ]

            self.X_types[f"region_{region_id}"] = feature_names

            for j, fname in enumerate(feature_names):
                all_data[fname] = voxels[:, j]

        self.data = pd.DataFrame(all_data)
        self.data["subject_id"] = subject_ids
        self.data["age"] = ages

        self.logger.info(f"Data shape: {self.data.shape}")

    # -------------------------
    def build_model(self):
        self.logger.info("Building model...")

        l0_models = []

        for region_id in range(1, self.n_regions + 1):
            model = PipelineCreator(problem_type="regression", apply_to=f"region_{region_id}")

            model.add("filter_columns", apply_to="*", keep=f"region_{region_id}")
            model.add("zscore", apply_to=f"region_{region_id}")

            model.add(
                ElasticNetCV(
                    l1_ratio=self.l1_ratio,
                    n_alphas=self.n_alphas,
                    cv=self.l0_cv,
                    random_state=self.seed,
                    n_jobs=1
                ),
                apply_to=f"region_{region_id}"
            )

            l0_models.append((f"region_{region_id}", model))

        # Meta model
        l1_meta = PipelineCreator(problem_type="regression")
        l1_meta.add("zscore", apply_to="*")
        l1_meta.add(
            ElasticNetCV(
                l1_ratio=self.l1_ratio,
                n_alphas=self.n_alphas,
                cv=self.l0_cv,
                random_state=self.seed,
                n_jobs=1
            )
        )

        stacking = PipelineCreator(problem_type="regression")
        stacking.add(
            "stacking",
            estimators=[l0_models],
            final_estimator=l1_meta,
            cv=self.l0_cv,
            apply_to="*"
        )

        return stacking

    # -------------------------
    def train(self, model):
        self.logger.info("Starting training...")

        all_features = []
        for f in self.X_types.values():
            all_features.extend(f)

        start = time.time()

        with parallel_config(
            backend="htcondor",
            n_jobs=-1,
            request_cpus=1,
            request_memory="4GB",
            request_disk="5GB",
            shared_data_dir="/home/fkarateke/brainage-stacking",
            pool="head2.htc.inm7.de",
            max_recursion_level=1,
            throttle=[25, 50],
        ):
            scores, trained = run_cross_validation(
                X=all_features,
                X_types=self.X_types,
                y="age",
                data=self.data,
                model=model,
                cv=self.outer_cv,
                scoring=["neg_mean_absolute_error", "r2"],
                return_estimator="final",
                seed=self.seed
            )

        elapsed = time.time() - start

        mae = -scores["test_neg_mean_absolute_error"].mean()
        r2 = scores["test_r2"].mean()

        self.logger.info(f"MAE: {mae:.3f}")
        self.logger.info(f"R2: {r2:.4f}")
        self.logger.info(f"Time (min): {elapsed/60:.2f}")

        # Save
        with open(self.output_dir / "model.pkl", "wb") as f:
            pickle.dump(trained, f)

        pd.DataFrame({
            "mae": [mae],
            "r2": [r2],
            "time_min": [elapsed / 60]
        }).to_csv(self.output_dir / "summary.csv", index=False)

    # -------------------------
    def run(self):
        self.load_data()
        self.prepare_data()
        model = self.build_model()
        self.train(model)


# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--n_regions", type=int, default=10)

    args = parser.parse_args()

    trainer = StackingTrainer(
        args.features_dir,
        args.output_dir,
        n_regions=args.n_regions
    )

    trainer.run()


if __name__ == "__main__":
    main()
