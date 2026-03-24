"""
============================================================
BRAIN AGE PREDICTION - MODULE 3
Region-wise Stacking Ensemble (Julearn Native)
============================================================

Paper: Region-wise stacking ensembles for estimating brain-age
       using structural MRI (More et al., 2025)
       https://www.sciencedirect.com/science/article/pii/S0010482525015355

Implementation:
    - Julearn native stacking
    - X_types based (one type per region)
    - Paper-compliant GLMnet (ElasticNet)
    - Nested CV: 3-fold L0, 5-fold outer

Author: Brain Age Prediction Project
Version: 3.0.0 (Julearn Native)
============================================================
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging as julearn_configure_logging

from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import VarianceThreshold

# ============================================================
# LOGGING
# ============================================================

def setup_logging(output_dir: Path):
    """Setup logging."""
    log_file = output_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    julearn_configure_logging(level="INFO")
    return logging.getLogger(__name__)

# ============================================================
# PAPER SETTINGS
# ============================================================

PAPER_SETTINGS = {
    "full_paper": {
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
    "balanced": {
        "n_alpha": 20,
        "alpha_range": (-4, 2),
        "n_l1_ratio": 9,
        "l1_ratio_range": (0.1, 0.9),
        "max_iter": 50000,
        "tol": 1e-4,
        "variance_threshold": 0.01,
        "l0_cv": 3,
        "outer_cv": 5
    },
    "fast_test": {
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

# ============================================================
# TRAINER CLASS
# ============================================================

class BrainAgeStackingTrainer:
    """Julearn-native stacking trainer for brain age prediction."""
    
    def __init__(self, features_dir: str, output_dir: str, n_regions: int = 100,
                 preset: str = "full_paper", seed: int = 42):
        
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.n_regions = n_regions
        self.seed = seed
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.output_dir)
        
        # Load preset
        config = PAPER_SETTINGS[preset]
        self.n_alpha = config["n_alpha"]
        self.alpha_range = config["alpha_range"]
        self.n_l1_ratio = config["n_l1_ratio"]
        self.l1_ratio_range = config["l1_ratio_range"]
        self.max_iter = config["max_iter"]
        self.tol = config["tol"]
        self.variance_threshold = config["variance_threshold"]
        self.l0_cv = config["l0_cv"]
        self.outer_cv = config["outer_cv"]
        
        # Generate hyperparameters
        self.alpha_values = np.logspace(self.alpha_range[0], self.alpha_range[1], self.n_alpha)
        self.l1_ratio_values = np.linspace(self.l1_ratio_range[0], self.l1_ratio_range[1], self.n_l1_ratio)
        
        # Data containers
        self.regional_voxels = None
        self.metadata = None
        self.data = None
        self.X_types = None
        self.all_features = None
        self.stacking_model = None
        self.scores = None
        self.trained_model = None
        
        self.logger.info("=" * 60)
        self.logger.info("BRAIN AGE STACKING TRAINER (Julearn Native)")
        self.logger.info("=" * 60)
        self.logger.info(f"Features dir : {self.features_dir}")
        self.logger.info(f"Output dir   : {self.output_dir}")
        self.logger.info(f"N regions    : {self.n_regions}")
        self.logger.info(f"Preset       : {preset}")
        self.logger.info(f"Alpha        : {self.n_alpha} values")
        self.logger.info(f"L1 ratio     : {self.n_l1_ratio} values")
        self.logger.info(f"Max iter     : {self.max_iter:,}")
        self.logger.info(f"L0 CV        : {self.l0_cv}-fold")
        self.logger.info(f"Outer CV     : {self.outer_cv}-fold")
    
    def load_data(self) -> bool:
        """Load Module 2 outputs."""
        self.logger.info("=" * 60)
        self.logger.info("LOADING DATA")
        self.logger.info("=" * 60)
        
        voxels_path = self.features_dir / "regional_voxels.pkl"
        metadata_path = self.features_dir / "regional_voxels_metadata.pkl"
        
        if not voxels_path.exists() or not metadata_path.exists():
            self.logger.error("Regional voxels or metadata not found!")
            return False
        
        with open(voxels_path, 'rb') as f:
            self.regional_voxels = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.logger.info(f"✓ Loaded {self.metadata['n_subjects']} subjects")
        self.logger.info(f"✓ Total regions available: {self.metadata['n_regions']}")
        
        if self.n_regions > self.metadata['n_regions']:
            self.logger.error(f"Requested {self.n_regions} > available {self.metadata['n_regions']}!")
            return False
        
        return True
    
    def prepare_data_for_julearn(self) -> bool:
        """Prepare data in Julearn X_types format."""
        self.logger.info("=" * 60)
        self.logger.info("PREPARING DATA (X_types)")
        self.logger.info("=" * 60)
        
        ages = np.array(self.metadata['ages'])
        subject_ids = self.metadata['subject_ids']
        n_subjects = len(ages)
        
        # Build X_types: {region_1: [voxel features], region_2: [...], ...}
        self.X_types = {}
        all_data = {}
        
        self.logger.info(f"Building X_types for {self.n_regions} regions...")
        
        for region_id in range(1, self.n_regions + 1):
            voxels = self.regional_voxels[region_id]
            n_voxels = voxels.shape[1]
            
            # Feature names for this region
            feature_names = [f'region_{region_id}_voxel_{v}' for v in range(n_voxels)]
            
            # Register in X_types
            self.X_types[f'region_{region_id}'] = feature_names
            
            # Store voxel data
            for i, fname in enumerate(feature_names):
                all_data[fname] = voxels[:, i]
            
            if region_id % 100 == 0 or region_id == self.n_regions:
                self.logger.info(f"  Processed {region_id}/{self.n_regions} regions")
        
        # Create DataFrame
        self.data = pd.DataFrame(all_data)
        self.data['subject_id'] = subject_ids
        self.data['age'] = ages
        
        # All feature names (for run_cross_validation)
        self.all_features = [f for features in self.X_types.values() for f in features]
        
        self.logger.info("")
        self.logger.info("✓ Data preparation complete")
        self.logger.info(f"  DataFrame shape: {self.data.shape}")
        self.logger.info(f"  X_types: {len(self.X_types)} region types")
        self.logger.info(f"  Total features: {len(self.all_features):,}")
        
        return True
    
    def build_l0_models(self):
        """Build L0 models (one per region) - Julearn native."""
        self.logger.info("=" * 60)
        self.logger.info("BUILDING L0 MODELS")
        self.logger.info("=" * 60)
        self.logger.info(f"Creating {self.n_regions} region-specific models...")
        
        l0_estimators = []
        
        for region_id in range(1, self.n_regions + 1):
            region_name = f'region_{region_id}'
            
            # Create pipeline for this region
            model = PipelineCreator(problem_type="regression", apply_to=region_name)
            
            # Step 1: Filter - keep only this region's voxels
            model.add("filter_columns", apply_to="*", keep=region_name)
            
            # Step 2: Variance threshold
            model.add(VarianceThreshold(threshold=self.variance_threshold), apply_to=region_name)
            
            # Step 3: Z-score
            model.add("zscore", apply_to=region_name)
            
            # Step 4: ElasticNet (GLMnet)
            model.add(
                ElasticNet(max_iter=self.max_iter, tol=self.tol, random_state=self.seed),
                alpha=self.alpha_values,
                l1_ratio=self.l1_ratio_values,
                apply_to=region_name
            )
            
            l0_estimators.append((region_name, model))
            
            if region_id % 100 == 0 or region_id == self.n_regions:
                self.logger.info(f"  Configured {region_id}/{self.n_regions} L0 models")
        
        self.logger.info("")
        self.logger.info(f"✓ {len(l0_estimators)} L0 models configured")
        self.logger.info(f"  Hyperparameters per model: {self.n_alpha * self.n_l1_ratio}")
        
        return l0_estimators
    
    def build_l1_model(self):
        """Build L1 meta-model."""
        self.logger.info("=" * 60)
        self.logger.info("BUILDING L1 META-MODEL")
        self.logger.info("=" * 60)
        
        l1_meta = PipelineCreator(problem_type="regression")
        
        # Z-score normalization of L0 predictions
        l1_meta.add("zscore", apply_to="*")
        
        # ElasticNet meta-learner
        l1_meta.add(
            ElasticNet(max_iter=self.max_iter, tol=self.tol, random_state=self.seed),
            alpha=self.alpha_values,
            l1_ratio=self.l1_ratio_values
        )
        
        self.logger.info("✓ L1 meta-model configured")
        self.logger.info(f"  Input: {self.n_regions} L0 predictions")
        self.logger.info(f"  Hyperparameters: {self.n_alpha * self.n_l1_ratio}")
        
        return l1_meta
    
    def build_stacking_model(self):
        """Build complete stacking ensemble."""
        self.logger.info("=" * 60)
        self.logger.info("BUILDING STACKING ENSEMBLE")
        self.logger.info("=" * 60)
        
        # Build L0 and L1
        l0_estimators = self.build_l0_models()
        l1_meta = self.build_l1_model()
        
        # Create stacking model
        stacking = PipelineCreator(problem_type="regression")
        stacking.add(
            "stacking",
            estimators=[l0_estimators],
            final_estimator=l1_meta,
            cv=self.l0_cv,  # 3-fold CV for L0 OOS predictions
            apply_to="*"
        )
        
        self.stacking_model = stacking
        
        self.logger.info("")
        self.logger.info("✓ Stacking ensemble configured")
        self.logger.info(f"  L0: {self.n_regions} region models")
        self.logger.info(f"  L0 CV: {self.l0_cv}-fold (OOS predictions)")
        self.logger.info(f"  L1: 1 meta-model")
        
        return stacking
    
    def train(self) -> bool:
        """Train with nested CV."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Features: {len(self.all_features):,}")
        self.logger.info(f"Subjects: {len(self.data)}")
        self.logger.info(f"Outer CV: {self.outer_cv}-fold")
        self.logger.info("")
        self.logger.info("⚠️  This will take a long time!")
        self.logger.info(f"   Estimated: {self.n_regions * 5}-{self.n_regions * 15} minutes")
        self.logger.info("")
        
        try:
            self.scores, self.trained_model = run_cross_validation(
                X=self.all_features,
                X_types=self.X_types,
                y='age',
                data=self.data,
                model=self.stacking_model,
                cv=self.outer_cv,
                scoring=['neg_mean_absolute_error', 'r2', 'neg_mean_squared_error'],
                return_estimator='final',
                seed=self.seed
            )
            
            self.logger.info("")
            self.logger.info("✓ Training completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def print_results(self):
        """Print results."""
        mae = -self.scores['test_neg_mean_absolute_error'].mean()
        mae_std = self.scores['test_neg_mean_absolute_error'].std()
        r2 = self.scores['test_r2'].mean()
        r2_std = self.scores['test_r2'].std()
        rmse = np.sqrt(-self.scores['test_neg_mean_squared_error'].mean())
        
        self.logger.info("=" * 60)
        self.logger.info("RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"MAE:  {mae:.2f} ± {mae_std:.2f} years")
        self.logger.info(f"R²:   {r2:.3f} ± {r2_std:.3f}")
        self.logger.info(f"RMSE: {rmse:.2f} years")
    
    def save_results(self) -> bool:
        """Save results."""
        try:
            # Model
            with open(self.output_dir / 'stacking_model.pkl', 'wb') as f:
                pickle.dump(self.trained_model, f)
            
            # Scores
            with open(self.output_dir / 'cv_scores.pkl', 'wb') as f:
                pickle.dump(self.scores, f)
            
            # Summary
            mae = -self.scores['test_neg_mean_absolute_error'].mean()
            r2 = self.scores['test_r2'].mean()
            rmse = np.sqrt(-self.scores['test_neg_mean_squared_error'].mean())
            
            results = pd.DataFrame({
                'metric': ['MAE', 'R2', 'RMSE'],
                'value': [mae, r2, rmse]
            })
            results.to_csv(self.output_dir / 'results_summary.csv', index=False)
            
            self.logger.info("✓ Results saved")
            return True
            
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Run complete pipeline."""
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
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETED")
        self.logger.info("=" * 60)
        return True

# ============================================================
# CLI
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Module 3: Stacking (Julearn Native)")
    parser.add_argument("features_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--n_regions", type=int, default=100)
    parser.add_argument("--preset", type=str, default="full_paper",
                        choices=list(PAPER_SETTINGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main(features_dir=None, output_dir=None, **kwargs):
    if features_dir is None or output_dir is None:
        args = parse_arguments()
        features_dir = args.features_dir
        output_dir = args.output_dir
        kwargs = vars(args)
        kwargs.pop('features_dir')
        kwargs.pop('output_dir')
    
    trainer = BrainAgeStackingTrainer(
        features_dir=features_dir,
        output_dir=output_dir,
        **kwargs
    )
    
    success = trainer.run_pipeline()
    
    if not success:
        print("❌ Pipeline failed")
        return False
    
    print("✅ Pipeline completed!")
    return True

if __name__ == "__main__":
    main()