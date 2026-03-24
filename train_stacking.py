#!/usr/bin/env python3
"""
Brain Age Prediction - Module 3
Stacking Model Training (Paper-Compliant with Detailed Logging)
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
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import time
import warnings
from contextlib import contextmanager

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging as julearn_configure_logging

from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import VarianceThreshold


class MultiLevelLogger:
    def __init__(self, output_dir: Path, log_name: str = "training"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.main_log = output_dir / f"{log_name}_{timestamp}.log"
        self.error_log = output_dir / f"errors_{timestamp}.log"
        self.progress_log = output_dir / f"progress_{timestamp}.csv"
        
        self.logger = logging.getLogger(f"{log_name}_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_format)
        
        file_handler = logging.FileHandler(self.main_log)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        
        error_handler = logging.FileHandler(self.error_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        
        self.progress_data = []
        julearn_configure_logging(level="WARNING")
        
    def log_progress(self, region_id: int, status: str, **metrics):
        entry = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'region_id': region_id, 'status': status, **metrics}
        self.progress_data.append(entry)
        
    def save_progress(self):
        if self.progress_data:
            df = pd.DataFrame(self.progress_data)
            df.to_csv(self.progress_log, index=False)
            
    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)


PAPER_SETTINGS = {
    "full_paper": {
        "description": "Full paper-compliant",
        "n_alpha": 100,
        "alpha_range": (-4, 2),
        "n_l1_ratio": 11,
        "l1_ratio_range": (0.0, 1.0),
        "max_iter": 100000,
        "tol": 1e-7,
        "variance_threshold": 0.0,
        "l0_cv": 3,
        "outer_cv": 5
    },
    "fast_test": {
        "description": "Fast test",
        "n_alpha": 5,
        "alpha_range": (-3, 2),
        "n_l1_ratio": 3,
        "l1_ratio_range": (0.3, 0.7),
        "max_iter": 10000,
        "tol": 1e-4,
        "variance_threshold": 0.01,
        "l0_cv": 2,
        "outer_cv": 3
    }
}


class BrainAgeStackingTrainer:
    def __init__(self, features_dir: str, output_dir: str, n_regions: int = 100, preset: str = "full_paper", seed: int = 42, **kwargs):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.n_regions = n_regions
        self.seed = seed
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MultiLevelLogger(self.output_dir / "logs")
        
        self._load_config(preset, **kwargs)
        
        self.regional_voxels = None
        self.metadata = None
        self.data = None
        self.X_types = None
        self.stacking_model = None
        self.trained_model = None
        self.scores = None
        
        self.training_stats = {'start_time': datetime.now(), 'convergence_warnings': 0}
        self._print_initialization()
        
    def _load_config(self, preset: str, **kwargs):
        if preset in PAPER_SETTINGS:
            config = PAPER_SETTINGS[preset]
            self.logger.info(f"Using preset: '{preset}' - {config['description']}")
            
            self.n_alpha = kwargs.get('n_alpha', config['n_alpha'])
            self.alpha_range = config['alpha_range']
            self.n_l1_ratio = kwargs.get('n_l1_ratio', config['n_l1_ratio'])
            self.l1_ratio_range = config['l1_ratio_range']
            self.max_iter = kwargs.get('max_iter', config['max_iter'])
            self.tol = kwargs.get('tol', config['tol'])
            self.variance_threshold = kwargs.get('variance_threshold', config['variance_threshold'])
            self.l0_cv = kwargs.get('l0_cv', config['l0_cv'])
            self.outer_cv = kwargs.get('outer_cv', config['outer_cv'])
        else:
            self.logger.warning(f"Unknown preset '{preset}'")
            self.n_alpha = 100
            self.alpha_range = (-4, 2)
            self.n_l1_ratio = 11
            self.l1_ratio_range = (0.0, 1.0)
            self.max_iter = 100000
            self.tol = 1e-7
            self.variance_threshold = 0.0
            self.l0_cv = 3
            self.outer_cv = 5
        
        self.alpha_values = np.logspace(self.alpha_range[0], self.alpha_range[1], self.n_alpha)
        self.l1_ratio_values = np.linspace(self.l1_ratio_range[0], self.l1_ratio_range[1], self.n_l1_ratio)
        self.total_hyperparam_combos = len(self.alpha_values) * len(self.l1_ratio_values)
        
    def _print_initialization(self):
        self.logger.info("="*70)
        self.logger.info("  BRAIN AGE STACKING TRAINER INITIALIZED")
        self.logger.info("="*70)
        self.logger.info(f"Features dir : {self.features_dir}")
        self.logger.info(f"Output dir   : {self.output_dir}")
        self.logger.info(f"N regions    : {self.n_regions}")
        self.logger.info(f"Alpha        : {self.n_alpha} values")
        self.logger.info(f"L1 ratio     : {self.n_l1_ratio} values")
        self.logger.info(f"Max iter     : {self.max_iter:,}")
        self.logger.info(f"L0 CV        : {self.l0_cv}-fold")
        self.logger.info(f"Outer CV     : {self.outer_cv}-fold")
        self.logger.info(f"Total combos : {self.total_hyperparam_combos:,}")
        
    def load_data(self) -> bool:
        self.logger.info("="*70)
        self.logger.info("  LOADING DATA")
        self.logger.info("="*70)
        
        try:
            voxels_path = self.features_dir / "regional_voxels.pkl"
            if not voxels_path.exists():
                self.logger.error(f"File not found: {voxels_path}")
                return False
            
            with open(voxels_path, 'rb') as f:
                self.regional_voxels = pickle.load(f)
            self.logger.info(f"✓ Loaded {len(self.regional_voxels)} regions")
            
            metadata_path = self.features_dir / "regional_voxels_metadata.pkl"
            if not metadata_path.exists():
                self.logger.error(f"File not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            self.logger.info(f"✓ Loaded {self.metadata['n_subjects']} subjects")
            
            if self.n_regions > self.metadata['n_regions']:
                self.logger.error(f"Requested {self.n_regions} but only {self.metadata['n_regions']} available!")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def prepare_data_for_julearn(self) -> bool:
        self.logger.info("="*70)
        self.logger.info("  PREPARING DATA")
        self.logger.info("="*70)
        
        try:
            ages = np.array(self.metadata['ages'])
            subject_ids = self.metadata['subject_ids']
            
            selected_regions = list(range(1, self.n_regions + 1))
            self.X_types = {}
            all_data = {}
            
            total_voxels = 0
            
            for region_id in selected_regions:
                voxels = self.regional_voxels[region_id]
                n_voxels = voxels.shape[1]
                total_voxels += n_voxels
                
                feature_names = [f'region_{region_id}_voxel_{v}' for v in range(n_voxels)]
                self.X_types[f'region_{region_id}'] = feature_names
                
                for j, fname in enumerate(feature_names):
                    all_data[fname] = voxels[:, j]
            
            self.data = pd.DataFrame(all_data)
            self.data['subject_id'] = subject_ids
            self.data['age'] = ages
            
            self.logger.info(f"✓ Data shape: {self.data.shape}")
            self.logger.info(f"✓ Total voxels: {total_voxels:,}")
            
            return True
        except Exception as e:
            self.logger.error(f"Prepare failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def build_l0_models(self) -> List[Tuple[str, PipelineCreator]]:
        self.logger.info(f"Building {self.n_regions} L0 models...")
        l0_models = []
        
        for region_id in range(1, self.n_regions + 1):
            model = PipelineCreator(problem_type="regression", apply_to=f'region_{region_id}')
            model.add("filter_columns", apply_to="*", keep=f'region_{region_id}')
            
            if self.variance_threshold > 0:
                model.add(VarianceThreshold(threshold=self.variance_threshold), apply_to=f'region_{region_id}')
            
            model.add("zscore", apply_to=f'region_{region_id}')
            model.add(ElasticNet(max_iter=self.max_iter, tol=self.tol, random_state=self.seed),
                     alpha=self.alpha_values, l1_ratio=self.l1_ratio_values, apply_to=f'region_{region_id}')
            
            l0_models.append((f'region_{region_id}', model))
        
        self.logger.info(f"✓ {len(l0_models)} L0 models configured")
        return l0_models
    
    def build_l1_model(self) -> PipelineCreator:
        self.logger.info("Building L1 meta-model...")
        l1_meta = PipelineCreator(problem_type="regression")
        l1_meta.add("zscore", apply_to="*")
        l1_meta.add(ElasticNet(max_iter=self.max_iter, tol=self.tol, random_state=self.seed),
                   alpha=self.alpha_values, l1_ratio=self.l1_ratio_values)
        self.logger.info("✓ L1 meta-model configured")
        return l1_meta
    
    def build_stacking_model(self) -> PipelineCreator:
        self.logger.info("="*70)
        self.logger.info("  BUILDING STACKING MODEL")
        self.logger.info("="*70)
        
        l0_models = self.build_l0_models()
        l1_meta = self.build_l1_model()
        
        stacking_model = PipelineCreator(problem_type="regression")
        stacking_model.add("stacking", estimators=[l0_models], final_estimator=l1_meta, cv=self.l0_cv, apply_to="*")
        
        self.stacking_model = stacking_model
        self.logger.info("✓ Stacking ensemble configured")
        return stacking_model
    
    def train(self) -> bool:
        self.logger.info("="*70)
        self.logger.info("  TRAINING")
        self.logger.info("="*70)
        
        try:
            all_features = []
            for region_features in self.X_types.values():
                all_features.extend(region_features)
            
            self.logger.info(f"Features: {len(all_features):,}")
            self.logger.info(f"Subjects: {len(self.data)}")
            self.logger.info("Training started...")
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                train_start = time.time()
                
                self.scores, self.trained_model = run_cross_validation(
                    X=all_features, X_types=self.X_types, y='age', data=self.data,
                    model=self.stacking_model, cv=self.outer_cv,
                    scoring=['neg_mean_absolute_error', 'r2', 'neg_mean_squared_error'],
                    return_estimator='final', seed=self.seed
                )
                
                train_time = time.time() - train_start
                
                if w:
                    self.logger.warning(f"Captured {len(w)} warnings")
                    self.training_stats['convergence_warnings'] = len(w)
            
            self.logger.info(f"✓ Training completed! Time: {train_time:.1f}s")
            self.training_stats['training_time'] = train_time
            
            return True
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def print_results(self):
        self.logger.info("="*70)
        self.logger.info("  RESULTS")
        self.logger.info("="*70)
        
        try:
            mae = -self.scores['test_neg_mean_absolute_error'].mean()
            mae_std = self.scores['test_neg_mean_absolute_error'].std()
            r2 = self.scores['test_r2'].mean()
            r2_std = self.scores['test_r2'].std()
            rmse = np.sqrt(-self.scores['test_neg_mean_squared_error'].mean())
            
            self.logger.info(f"MAE:  {mae:.3f} ± {mae_std:.3f} years")
            self.logger.info(f"RMSE: {rmse:.3f} years")
            self.logger.info(f"R²:   {r2:.4f} ± {r2_std:.4f}")
        except Exception as e:
            self.logger.error(f"Print results failed: {e}")
    
    def save_results(self) -> bool:
        self.logger.info("="*70)
        self.logger.info("  SAVING RESULTS")
        self.logger.info("="*70)
        
        try:
            with open(self.output_dir / 'stacking_model.pkl', 'wb') as f:
                pickle.dump(self.trained_model, f)
            self.logger.info("✓ Model saved")
            
            with open(self.output_dir / 'cv_scores.pkl', 'wb') as f:
                pickle.dump(self.scores, f)
            self.logger.info("✓ Scores saved")
            
            mae = -self.scores['test_neg_mean_absolute_error'].mean()
            r2 = self.scores['test_r2'].mean()
            rmse = np.sqrt(-self.scores['test_neg_mean_squared_error'].mean())
            
            summary = pd.DataFrame([{
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'n_regions': self.n_regions,
                'mae': mae, 'rmse': rmse, 'r2': r2,
                'training_time_sec': self.training_stats.get('training_time', 0)
            }])
            summary.to_csv(self.output_dir / 'results_summary.csv', index=False)
            self.logger.info("✓ Summary saved")
            
            self.logger.save_progress()
            return True
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_pipeline(self) -> bool:
        start = time.time()
        
        try:
            if not self.load_data():
                return False
            if not self.prepare_data_for_julearn():
                return False
            
            self.build_stacking_model()
            
            if not self.train():
                return False
            
            self.print_results()
            
            if not self.save_results():
                return False
            
            total_time = time.time() - start
            self.logger.info("="*70)
            self.logger.info(f"  PIPELINE COMPLETED! Total time: {total_time:.1f}s")
            self.logger.info("="*70)
            
            return True
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Module 3: Stacking Training")
    parser.add_argument("features_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--n_regions", type=int, default=100)
    parser.add_argument("--preset", type=str, default="full_paper", choices=list(PAPER_SETTINGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_arguments()
    trainer = BrainAgeStackingTrainer(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        n_regions=args.n_regions,
        preset=args.preset,
        seed=args.seed
    )
    
    success = trainer.run_pipeline()
    
    if success:
        print("\n" + "="*70)
        print("  ✅ TRAINING COMPLETED!")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("  ❌ TRAINING FAILED!")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()