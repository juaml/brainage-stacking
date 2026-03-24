#!/usr/bin/env python3
"""
Brain Age Prediction - Module 3 (Batch Version - Paper Compliant)
Uses ElasticNetCV with automatic hyperparameter tuning (glmnet equivalent)
"""

import os
import sys
import pickle
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import warnings

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging as julearn_configure_logging

from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import VarianceThreshold


class BatchLogger:
    def __init__(self, output_dir: Path, batch_id: str):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = output_dir / f"batch_{batch_id}_{timestamp}.log"
        
        self.logger = logging.getLogger(f"batch_{batch_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
        
        julearn_configure_logging(level="WARNING")
        
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)


PAPER_SETTINGS = {
    "paper_compliant": {
        "description": "Paper-compliant using ElasticNetCV (automatic tuning)",
        "n_alphas": 100,
        "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        "max_iter": 100000,
        "tol": 1e-7,
        "variance_threshold": 0.0,
        "l0_cv": 3,
        "outer_cv": 5
    },
    "fast_test": {
        "description": "Fast test",
        "n_alphas": 10,
        "l1_ratio": [0.5, 0.7, 0.9],
        "max_iter": 10000,
        "tol": 1e-4,
        "variance_threshold": 0.01,
        "l0_cv": 2,
        "outer_cv": 3
    }
}


class BatchTrainer:
    def __init__(self, features_dir: str, output_dir: str, region_start: int, region_end: int, 
                 batch_id: str, preset: str = "paper_compliant", seed: int = 42):
        
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.region_start = region_start
        self.region_end = region_end
        self.batch_id = batch_id
        self.seed = seed
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = BatchLogger(self.output_dir, batch_id)
        
        config = PAPER_SETTINGS[preset]
        self.n_alphas = config['n_alphas']
        self.l1_ratio = config['l1_ratio']
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.variance_threshold = config['variance_threshold']
        self.l0_cv = config['l0_cv']
        self.outer_cv = config['outer_cv']
        
        self.regional_voxels = None
        self.metadata = None
        self.data = None
        self.X_types = None
        
        self.logger.info("="*70)
        self.logger.info(f"BATCH TRAINER: {batch_id}")
        self.logger.info(f"Regions: {region_start}-{region_end}")
        self.logger.info(f"Preset: {preset} - {config['description']}")
        self.logger.info(f"ElasticNetCV: n_alphas={self.n_alphas}, l1_ratio={len(self.l1_ratio)} values")
        self.logger.info("="*70)
        
    def load_data(self) -> bool:
        self.logger.info("Loading data...")
        try:
            with open(self.features_dir / "regional_voxels.pkl", 'rb') as f:
                self.regional_voxels = pickle.load(f)
            
            with open(self.features_dir / "regional_voxels_metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.logger.info(f"✓ Loaded {self.metadata['n_subjects']} subjects")
            return True
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            return False
    
    def prepare_data(self) -> bool:
        self.logger.info(f"Preparing regions {self.region_start}-{self.region_end}...")
        try:
            ages = np.array(self.metadata['ages'])
            subject_ids = self.metadata['subject_ids']
            
            regions = list(range(self.region_start, self.region_end + 1))
            self.X_types = {}
            all_data = {}
            
            for region_id in regions:
                voxels = self.regional_voxels[region_id]
                feature_names = [f'region_{region_id}_voxel_{v}' for v in range(voxels.shape[1])]
                self.X_types[f'region_{region_id}'] = feature_names
                
                for j, fname in enumerate(feature_names):
                    all_data[fname] = voxels[:, j]
            
            self.data = pd.DataFrame(all_data)
            self.data['subject_id'] = subject_ids
            self.data['age'] = ages
            
            self.logger.info(f"✓ Data shape: {self.data.shape}")
            return True
        except Exception as e:
            self.logger.error(f"Prepare failed: {e}")
            return False
    
    def build_model(self):
        self.logger.info("Building stacking model with ElasticNetCV...")
        
        l0_models = []
        for region_id in range(self.region_start, self.region_end + 1):
            model = PipelineCreator(problem_type="regression", apply_to=f'region_{region_id}')
            model.add("filter_columns", apply_to="*", keep=f'region_{region_id}')
            
            if self.variance_threshold > 0:
                model.add(VarianceThreshold(threshold=self.variance_threshold), apply_to=f'region_{region_id}')
            
            model.add("zscore", apply_to=f'region_{region_id}')
            
            # ElasticNetCV with automatic hyperparameter tuning (paper-compliant!)
            model.add(
                ElasticNetCV(
                    l1_ratio=self.l1_ratio,
                    n_alphas=self.n_alphas,
                    cv=self.l0_cv,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.seed,
                    n_jobs=1
                ),
                apply_to=f'region_{region_id}'
            )
            
            l0_models.append((f'region_{region_id}', model))
        
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
                n_jobs=1
            )
        )
        
        stacking = PipelineCreator(problem_type="regression")
        stacking.add("stacking", estimators=[l0_models], final_estimator=l1_meta, cv=self.l0_cv, apply_to="*")
        
        self.logger.info(f"✓ Model built: {len(l0_models)} regions with ElasticNetCV")
        return stacking
    
    def train(self, model) -> bool:
        self.logger.info("Training started...")
        try:
            all_features = []
            for features in self.X_types.values():
                all_features.extend(features)
            
            self.logger.info(f"Features: {len(all_features):,}, Subjects: {len(self.data)}")
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                start = time.time()
                scores, trained = run_cross_validation(
                    X=all_features, X_types=self.X_types, y='age', data=self.data,
                    model=model, cv=self.outer_cv,
                    scoring=['neg_mean_absolute_error', 'r2', 'neg_mean_squared_error'],
                    return_estimator='final', seed=self.seed
                )
                elapsed = time.time() - start
                
                if w:
                    self.logger.warning(f"{len(w)} warnings during training")
            
            self.logger.info(f"✓ Training done! Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            
            mae = -scores['test_neg_mean_absolute_error'].mean()
            r2 = scores['test_r2'].mean()
            rmse = np.sqrt(-scores['test_neg_mean_squared_error'].mean())
            
            self.logger.info(f"MAE: {mae:.3f}, R²: {r2:.4f}, RMSE: {rmse:.3f}")
            
            # Save
            with open(self.output_dir / f'model_batch_{self.batch_id}.pkl', 'wb') as f:
                pickle.dump(trained, f)
            
            with open(self.output_dir / f'scores_batch_{self.batch_id}.pkl', 'wb') as f:
                pickle.dump(scores, f)
            
            summary = pd.DataFrame([{
                'batch_id': self.batch_id,
                'region_start': self.region_start,
                'region_end': self.region_end,
                'mae': mae, 'r2': r2, 'rmse': rmse,
                'training_time_sec': elapsed,
                'training_time_min': elapsed/60
            }])
            summary.to_csv(self.output_dir / f'summary_batch_{self.batch_id}.csv', index=False)
            
            self.logger.info("✓ Results saved")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run(self) -> bool:
        if not self.load_data():
            return False
        if not self.prepare_data():
            return False
        
        model = self.build_model()
        
        if not self.train(model):
            return False
        
        self.logger.info("="*70)
        self.logger.info(f"BATCH {self.batch_id} COMPLETED!")
        self.logger.info("="*70)
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Training (Paper-Compliant)")
    parser.add_argument("features_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--region_start", type=int, required=True)
    parser.add_argument("--region_end", type=int, required=True)
    parser.add_argument("--batch_id", type=str, required=True)
    parser.add_argument("--preset", type=str, default="paper_compliant", choices=list(PAPER_SETTINGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    trainer = BatchTrainer(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        region_start=args.region_start,
        region_end=args.region_end,
        batch_id=args.batch_id,
        preset=args.preset,
        seed=args.seed
    )
    
    success = trainer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
