"""
============================================================
BRAIN AGE PREDICTION TOOL
Module 2: Feature Extraction from CAT12 Processed MRI
============================================================

Description:
    Extracts features from CAT12 processed MRI scans using
    multi-atlas parcellation (Schaefer + Brainnetome + Cerebellar).
    
    Outputs:
    1. Regional mean GMV matrix (for baseline models)
    2. Regional voxel data (for stacking ensemble L0 models)

Atlas Configuration:
    - Schaefer-800: Cortical regions (IDs: 1-800)
    - Brainnetome-36: Subcortical regions (IDs: 801-836)
    - Cerebellar-34: Cerebellar regions (IDs: 837-870)
    
    NOTE: Paper uses 37 cerebellar regions (total 873).
    Current atlas has 34 regions (total 870).
    TODO: Verify correct cerebellar atlas with supervisor.

Usage:
    # From Jupyter Notebook:
    from feature_extractor import BrainAgeFeatureExtractor
    
    extractor = BrainAgeFeatureExtractor(
        merged_csv="IXI_Edit_merged_final.csv",
        output_dir="./features/"
    )
    extractor.run_pipeline()
    
    # From Terminal:
    python feature_extractor.py IXI_Edit_merged_final.csv ./features/

Author:
    Brain Age Prediction Project
Version:
    2.0.0
============================================================
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from nilearn import datasets
from nilearn.image import resample_to_img

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("brainage_feature_extraction.log")
    ]
)

logger = logging.getLogger(__name__)

# ============================================================
# ATLAS CONFIGURATIONS
# ============================================================
# NOTE: These are the expected region counts from the paper.
# Actual counts may differ based on available atlases.
# ============================================================

ATLAS_CONFIG = {
    "schaefer": {
        "n_rois": 800,
        "id_range": (1, 800),
        "offset": 0,
        "type": "cortical"
    },
    "subcortical": {
        "n_rois": 36,
        "id_range": (801, 836),
        "offset": 800,
        "type": "subcortical",
        "brainnetome_filter": (211, 246)  # Brainnetome subcortical region IDs
    },
    "cerebellar": {
        "n_rois": 34,  # Paper: 37, Current: 34 (TODO: verify)
        "id_range": (837, 870),  # Paper: (837, 873)
        "offset": 836,
        "type": "cerebellar"
    }
}

EXPECTED_TOTAL_REGIONS = 870  # Paper: 873

# ============================================================
# FEATURE EXTRACTOR CLASS
# ============================================================

class BrainAgeFeatureExtractor:
    """
    Feature extraction pipeline for brain age prediction.
    
    Extracts regional features from CAT12 processed MRI scans
    using multi-atlas parcellation.
    
    Parameters:
        merged_csv (str): Path to merged CSV from Module 1 (ID, AGE, MRI_PATH).
        output_dir (str): Directory for output files.
        atlas_dir (str): Directory containing atlas files (optional).
        
    Attributes:
        df (pd.DataFrame): Merged participant data.
        combined_atlas (np.ndarray): Combined parcellation atlas.
        n_regions (int): Total number of regions.
    """
    
    def __init__(self, merged_csv: str, output_dir: str = "./features/",
                 atlas_dir: Optional[str] = None):
        self.merged_csv = merged_csv
        self.output_dir = Path(output_dir)
        self.atlas_dir = Path(atlas_dir) if atlas_dir else None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.combined_atlas: Optional[np.ndarray] = None
        self.atlas_img: Optional[nib.Nifti1Image] = None
        self.n_regions: int = 0
        
        logger.info("BrainAgeFeatureExtractor initialized")
        logger.info(f"  Merged CSV : {self.merged_csv}")
        logger.info(f"  Output dir : {self.output_dir}")
        logger.info(f"  Atlas dir  : {self.atlas_dir if self.atlas_dir else 'Using nilearn defaults'}")
    
    # ========================================================
    # 1. LOAD DATA
    # ========================================================
    
    def load_data(self) -> bool:
        """Load merged CSV from Module 1."""
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)
        
        try:
            self.df = pd.read_csv(self.merged_csv)
            logger.info(f"Loaded {len(self.df)} participants")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            # Validate columns
            required_cols = ['IXI_ID', 'AGE', 'MRI_PATH']  # Adjust based on Module 1 output
            # Try flexible column detection
            id_col = self.df.columns[0]  # First column is ID
            
            if 'AGE' not in self.df.columns:
                logger.error("AGE column not found!")
                return False
            
            if 'MRI_PATH' not in self.df.columns:
                logger.error("MRI_PATH column not found!")
                return False
            
            logger.info(f"Using ID column: {id_col}")
            self.df.rename(columns={id_col: 'ID'}, inplace=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    # ========================================================
    # 2. LOAD AND COMBINE ATLASES
    # ========================================================
    
    def load_reference_mri(self) -> nib.Nifti1Image:
        """Load first participant's MRI as reference for resampling."""
        mri_path = self.df.iloc[0]['MRI_PATH']
        logger.info(f"Loading reference MRI: {Path(mri_path).name}")
        
        try:
            ref_img = nib.load(mri_path)
            logger.info(f"Reference MRI shape: {ref_img.shape}")
            return ref_img
        except Exception as e:
            logger.error(f"Failed to load reference MRI: {e}")
            raise
    
    def load_and_resample_atlas(self, atlas_path: str, reference_img: nib.Nifti1Image,
                                 atlas_name: str, region_filter: Optional[Tuple[int, int]] = None,
                                 id_offset: int = 0) -> Tuple[nib.Nifti1Image, np.ndarray]:
        """
        Load atlas and resample to match reference MRI.
        
        Args:
            atlas_path: Path to atlas NIfTI file.
            reference_img: Reference MRI image.
            atlas_name: Name for logging.
            region_filter: (min, max) to filter specific regions.
            id_offset: Offset to add to region IDs.
            
        Returns:
            Tuple of (resampled image, resampled data array).
        """
        logger.info(f"Loading {atlas_name} atlas: {Path(atlas_path).name}")
        
        try:
            # Load atlas
            atlas_img = nib.load(atlas_path)
            atlas_data = atlas_img.get_fdata()
            
            logger.info(f"  Original shape: {atlas_data.shape}")
            logger.info(f"  Unique regions: {len(np.unique(atlas_data[atlas_data > 0]))}")
            
            # Apply region filter if specified
            if region_filter:
                min_id, max_id = region_filter
                mask = (atlas_data >= min_id) & (atlas_data <= max_id)
                # Remap to 1-indexed
                filtered_data = np.where(mask, atlas_data - min_id + 1, 0)
                atlas_img = nib.Nifti1Image(filtered_data.astype(np.int16),
                                            atlas_img.affine, atlas_img.header)
                logger.info(f"  Filtered to regions {min_id}-{max_id}")
                logger.info(f"  Regions after filter: {len(np.unique(filtered_data[filtered_data > 0]))}")
            
            # Resample to reference
            resampled_img = resample_to_img(atlas_img, reference_img, interpolation='nearest')
            resampled_data = resampled_img.get_fdata()
            
            # Apply offset
            if id_offset > 0:
                resampled_data = np.where(resampled_data > 0, resampled_data + id_offset, 0)
            
            logger.info(f"  Resampled shape: {resampled_data.shape}")
            logger.info(f"  Final ID range: {int(np.min(resampled_data[resampled_data > 0]))} - {int(np.max(resampled_data[resampled_data > 0]))}")
            
            return resampled_img, resampled_data
            
        except Exception as e:
            logger.error(f"Failed to load {atlas_name} atlas: {e}")
            raise
    
    def combine_atlases_with_checks(self, schaefer_data: np.ndarray,
                                      subcortical_data: np.ndarray,
                                      cerebellar_data: np.ndarray) -> np.ndarray:
        """
        Combine three atlases with overlap detection.
        
        Priority order (last wins in case of overlap):
        1. Schaefer (cortical)
        2. Subcortical
        3. Cerebellar
        
        Args:
            schaefer_data: Cortical parcellation.
            subcortical_data: Subcortical parcellation (with offset).
            cerebellar_data: Cerebellar parcellation (with offset).
            
        Returns:
            Combined atlas array.
        """
        logger.info("=" * 60)
        logger.info("COMBINING ATLASES")
        logger.info("=" * 60)
        
        combined = np.zeros_like(schaefer_data)
        
        # 1. Schaefer (base layer)
        mask_schaefer = schaefer_data > 0
        combined[mask_schaefer] = schaefer_data[mask_schaefer]
        n_schaefer_voxels = np.sum(mask_schaefer)
        logger.info(f"Schaefer: {n_schaefer_voxels:,} voxels")
        
        # 2. Subcortical (overwrite if overlap)
        mask_subcortical = subcortical_data > 0
        overlap_sub_sch = mask_schaefer & mask_subcortical
        if np.any(overlap_sub_sch):
            logger.warning(f"⚠️  {np.sum(overlap_sub_sch):,} voxels overlap between Schaefer and Subcortical")
            logger.warning("   Subcortical will overwrite Schaefer in overlapping voxels")
        combined[mask_subcortical] = subcortical_data[mask_subcortical]
        n_subcortical_voxels = np.sum(mask_subcortical)
        logger.info(f"Subcortical: {n_subcortical_voxels:,} voxels")
        
        # 3. Cerebellar (overwrite if overlap)
        mask_cerebellar = cerebellar_data > 0
        overlap_cer_sch = mask_schaefer & mask_cerebellar
        overlap_cer_sub = mask_subcortical & mask_cerebellar
        
        if np.any(overlap_cer_sch):
            logger.warning(f"⚠️  {np.sum(overlap_cer_sch):,} voxels overlap between Schaefer and Cerebellar")
            logger.warning("   Cerebellar will overwrite Schaefer in overlapping voxels")
        
        if np.any(overlap_cer_sub):
            logger.warning(f"⚠️  {np.sum(overlap_cer_sub):,} voxels overlap between Subcortical and Cerebellar")
            logger.warning("   Cerebellar will overwrite Subcortical in overlapping voxels")
        
        combined[mask_cerebellar] = cerebellar_data[mask_cerebellar]
        n_cerebellar_voxels = np.sum(mask_cerebellar)
        logger.info(f"Cerebellar: {n_cerebellar_voxels:,} voxels")
        
        # Verify combined atlas
        unique_labels = np.unique(combined[combined > 0])
        self.n_regions = len(unique_labels)
        
        logger.info("=" * 60)
        logger.info("COMBINED ATLAS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total regions: {self.n_regions}")
        logger.info(f"Expected: {EXPECTED_TOTAL_REGIONS} (Paper: 873)")
        logger.info(f"ID range: {int(np.min(unique_labels))} - {int(np.max(unique_labels))}")
        logger.info(f"Total atlas voxels: {np.sum(combined > 0):,}")
        
        if self.n_regions != EXPECTED_TOTAL_REGIONS:
            logger.warning(f"⚠️  Region count mismatch!")
            logger.warning(f"   Expected: {EXPECTED_TOTAL_REGIONS}, Got: {self.n_regions}")
            logger.warning(f"   This may be due to cerebellar atlas difference (34 vs 37 regions)")
        
        return combined
    
    def load_atlases(self) -> bool:
        """Load and combine all atlases."""
        logger.info("=" * 60)
        logger.info("ATLAS LOADING")
        logger.info("=" * 60)
        
        try:
            # Load reference MRI
            reference_img = self.load_reference_mri()
            
            # 1. Schaefer-800 (cortical)
            logger.info("\n--- Schaefer Atlas (Cortical) ---")
            schaefer = datasets.fetch_atlas_schaefer_2018(
                n_rois=ATLAS_CONFIG["schaefer"]["n_rois"],
                yeo_networks=7,
                resolution_mm=1
            )
            _, schaefer_data = self.load_and_resample_atlas(
                atlas_path=schaefer.maps,
                reference_img=reference_img,
                atlas_name="Schaefer-800",
                id_offset=ATLAS_CONFIG["schaefer"]["offset"]
            )
            
            # 2. Brainnetome (subcortical)
            logger.info("\n--- Brainnetome Atlas (Subcortical) ---")
            
            # Try to find Brainnetome atlas
            brainnetome_candidates = [
                "BNA_MPM_thr25_1.25mm.nii.gz",
                "/home/fkarateke/BNA_MPM_thr25_1.25mm.nii.gz",
            ]
            
            if self.atlas_dir:
                brainnetome_candidates.insert(0, str(self.atlas_dir / "BNA_MPM_thr25_1.25mm.nii.gz"))
            
            brainnetome_path = None
            for path in brainnetome_candidates:
                if os.path.exists(path):
                    brainnetome_path = path
                    break
            
            if not brainnetome_path:
                logger.error("Brainnetome atlas not found!")
                logger.error(f"Searched paths: {brainnetome_candidates}")
                return False
            
            _, subcortical_data = self.load_and_resample_atlas(
                atlas_path=brainnetome_path,
                reference_img=reference_img,
                atlas_name="Brainnetome (Subcortical)",
                region_filter=ATLAS_CONFIG["subcortical"]["brainnetome_filter"],
                id_offset=ATLAS_CONFIG["subcortical"]["offset"]
            )
            
            # 3. Cerebellar
            logger.info("\n--- Cerebellar Atlas ---")
            
            cerebellar_candidates = [
                "atl-Anatom_space-MNI_dseg.nii",
                "/home/fkarateke/atl-Anatom_space-MNI_dseg.nii",
            ]
            
            if self.atlas_dir:
                cerebellar_candidates.insert(0, str(self.atlas_dir / "atl-Anatom_space-MNI_dseg.nii"))
            
            cerebellar_path = None
            for path in cerebellar_candidates:
                if os.path.exists(path):
                    cerebellar_path = path
                    break
            
            if not cerebellar_path:
                logger.error("Cerebellar atlas not found!")
                logger.error(f"Searched paths: {cerebellar_candidates}")
                return False
            
            _, cerebellar_data = self.load_and_resample_atlas(
                atlas_path=cerebellar_path,
                reference_img=reference_img,
                atlas_name="Cerebellar",
                id_offset=ATLAS_CONFIG["cerebellar"]["offset"]
            )
            
            # 4. Combine atlases
            self.combined_atlas = self.combine_atlases_with_checks(
                schaefer_data, subcortical_data, cerebellar_data
            )
            
            # Save combined atlas
            combined_atlas_path = self.output_dir / "combined_atlas.nii.gz"
            self.atlas_img = nib.Nifti1Image(
                self.combined_atlas.astype(np.int16),
                reference_img.affine,
                reference_img.header
            )
            nib.save(self.atlas_img, combined_atlas_path)
            logger.info(f"\n✓ Combined atlas saved: {combined_atlas_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Atlas loading failed: {e}")
            return False
    
    # ========================================================
    # 3. FEATURE EXTRACTION
    # ========================================================
    
    def extract_regional_mean_gmv(self) -> pd.DataFrame:
        """
        Extract regional mean GMV for all participants.
        
        This is the baseline approach used in the paper for comparison.
        
        Returns:
            DataFrame with shape (n_subjects, n_regions + 2)
            Columns: ID, region_1, region_2, ..., region_N, AGE
        """
        logger.info("=" * 60)
        logger.info("EXTRACTING REGIONAL MEAN GMV (Baseline)")
        logger.info("=" * 60)
        
        n_subjects = len(self.df)
        regional_gmv = np.zeros((n_subjects, self.n_regions))
        
        successful = 0
        failed = 0
        
        for idx, row in self.df.iterrows():
            subject_id = row['ID']
            mri_path = row['MRI_PATH']
            
            try:
                # Load MRI
                mri_img = nib.load(mri_path)
                mri_data = mri_img.get_fdata()
                
                # Verify shape
                if mri_data.shape != self.combined_atlas.shape:
                    logger.error(f"[{idx+1}/{n_subjects}] ID {subject_id}: Shape mismatch!")
                    logger.error(f"  MRI: {mri_data.shape}, Atlas: {self.combined_atlas.shape}")
                    failed += 1
                    continue
                
                # Extract regional mean GMV
                for region_id in range(1, self.n_regions + 1):
                    mask = (self.combined_atlas == region_id)
                    n_voxels = np.sum(mask)
                    
                    if n_voxels > 0:
                        regional_gmv[idx, region_id - 1] = np.mean(mri_data[mask])
                    else:
                        logger.warning(f"  Region {region_id} has no voxels!")
                
                successful += 1
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx+1}/{n_subjects} participants")
                
            except Exception as e:
                logger.error(f"[{idx+1}/{n_subjects}] ID {subject_id}: Failed - {e}")
                failed += 1
        
        logger.info(f"\n✓ Regional mean GMV extraction complete")
        logger.info(f"  Successful: {successful}, Failed: {failed}")
        
        # Create DataFrame
        region_cols = [f"region_{i+1}" for i in range(self.n_regions)]
        df_regional = pd.DataFrame(regional_gmv, columns=region_cols)
        df_regional.insert(0, 'ID', self.df['ID'].values)
        df_regional['AGE'] = self.df['AGE'].values
        
        # Save
        output_path = self.output_dir / "regional_mean_gmv.csv"
        df_regional.to_csv(output_path, index=False)
        logger.info(f"✓ Saved: {output_path}")
        logger.info(f"  Shape: {df_regional.shape}")
        
        return df_regional
    
    def extract_regional_voxels(self) -> Dict[int, np.ndarray]:
        """
        Extract voxel-wise data for each region across all participants.
        
        This is used for training L0 models in the stacking ensemble.
        
        Returns:
            Dictionary: {region_id: voxel_matrix}
            where voxel_matrix has shape (n_subjects, n_voxels_in_region)
        """
        logger.info("=" * 60)
        logger.info("EXTRACTING REGIONAL VOXELS (for L0 models)")
        logger.info("=" * 60)
        
        # Initialize containers for each region
        regional_voxels = {}
        region_voxel_counts = {}
        
        # Count voxels per region
        for region_id in range(1, self.n_regions + 1):
            mask = (self.combined_atlas == region_id)
            n_voxels = np.sum(mask)
            region_voxel_counts[region_id] = n_voxels
            regional_voxels[region_id] = np.zeros((len(self.df), n_voxels))
        
        logger.info(f"Voxel counts per region:")
        logger.info(f"  Min: {min(region_voxel_counts.values())}")
        logger.info(f"  Max: {max(region_voxel_counts.values())}")
        logger.info(f"  Mean: {np.mean(list(region_voxel_counts.values())):.0f}")
        logger.info(f"  Total: {sum(region_voxel_counts.values()):,}")
        
        # Extract voxels for each participant
        successful = 0
        failed = 0
        
        for idx, row in self.df.iterrows():
            subject_id = row['ID']
            mri_path = row['MRI_PATH']
            
            try:
                # Load MRI
                mri_img = nib.load(mri_path)
                mri_data = mri_img.get_fdata()
                
                # Verify shape
                if mri_data.shape != self.combined_atlas.shape:
                    logger.error(f"[{idx+1}/{len(self.df)}] ID {subject_id}: Shape mismatch!")
                    failed += 1
                    continue
                
                # Extract voxels for each region
                for region_id in range(1, self.n_regions + 1):
                    mask = (self.combined_atlas == region_id)
                    regional_voxels[region_id][idx, :] = mri_data[mask].flatten()
                
                successful += 1
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx+1}/{len(self.df)} participants")
                
            except Exception as e:
                logger.error(f"[{idx+1}/{len(self.df)}] ID {subject_id}: Failed - {e}")
                failed += 1
        
        logger.info(f"\n✓ Regional voxel extraction complete")
        logger.info(f"  Successful: {successful}, Failed: {failed}")
        
        # Save
        output_path = self.output_dir / "regional_voxels.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(regional_voxels, f)
        
        logger.info(f"✓ Saved: {output_path}")
        
        # Save metadata
        metadata = {
            'n_subjects': len(self.df),
            'n_regions': self.n_regions,
            'region_voxel_counts': region_voxel_counts,
            'subject_ids': self.df['ID'].tolist(),
            'ages': self.df['AGE'].tolist()
        }
        
        metadata_path = self.output_dir / "regional_voxels_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✓ Saved metadata: {metadata_path}")
        
        return regional_voxels
    
    # ========================================================
    # 4. MAIN PIPELINE
    # ========================================================
    
    def run_pipeline(self) -> bool:
        """
        Execute the complete Module 2 pipeline.
        
        Steps:
        1. Load merged data from Module 1
        2. Load and combine atlases
        3. Extract regional mean GMV (baseline)
        4. Extract regional voxels (for L0 models)
        
        Returns:
            bool: True if successful, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("BRAIN AGE PREDICTION - MODULE 2")
        logger.info("Feature Extraction")
        logger.info("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Load atlases
        if not self.load_atlases():
            return False
        
        # Step 3: Extract regional mean GMV
        df_regional = self.extract_regional_mean_gmv()
        
        # Step 4: Extract regional voxels
        regional_voxels = self.extract_regional_voxels()
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"\nOutputs saved in: {self.output_dir}")
        logger.info(f"  - combined_atlas.nii.gz")
        logger.info(f"  - regional_mean_gmv.csv")
        logger.info(f"  - regional_voxels.pkl")
        logger.info(f"  - regional_voxels_metadata.pkl")
        logger.info(f"\nReady for Module 3: Model Training")
        
        return True


# ============================================================
# CLI INTERFACE
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Brain Age Prediction - Module 2: Feature Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python feature_extractor.py IXI_Edit_merged_final.csv ./features/
  
  # With custom atlas directory
  python feature_extractor.py data_merged.csv ./features/ --atlas_dir /path/to/atlases/
        """
    )
    
    parser.add_argument("merged_csv", type=str,
                        help="Path to merged CSV from Module 1")
    parser.add_argument("output_dir", type=str,
                        help="Output directory for features")
    parser.add_argument("--atlas_dir", type=str, default=None,
                        help="Directory containing atlas files (optional)")
    
    return parser.parse_args()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main(merged_csv: str = None, output_dir: str = None, atlas_dir: str = None):
    """
    Main execution function. Supports both CLI and Jupyter usage.
    
    Args:
        merged_csv: Path to merged CSV from Module 1.
        output_dir: Output directory for features.
        atlas_dir: Directory containing atlas files (optional).
        
    Returns:
        bool: True if successful, False otherwise.
        
    Examples:
        # Jupyter:
        main("IXI_Edit_merged_final.csv", "./features/")
        
        # Terminal:
        python feature_extractor.py IXI_Edit_merged_final.csv ./features/
    """
    
    # Resolve arguments
    if merged_csv is None or output_dir is None:
        args = parse_arguments()
        merged_csv = args.merged_csv
        output_dir = args.output_dir
        atlas_dir = args.atlas_dir
    
    # Initialize extractor
    extractor = BrainAgeFeatureExtractor(
        merged_csv=merged_csv,
        output_dir=output_dir,
        atlas_dir=atlas_dir
    )
    
    # Run pipeline
    success = extractor.run_pipeline()
    
    if not success:
        logger.error("Pipeline failed. Check log messages above.")
        return False
    
    return True


if __name__ == "__main__":
    main()
