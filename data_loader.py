"""
============================================================
BRAIN AGE PREDICTION TOOL
Module 1: Data Loading, Validation & Matching
============================================================

Description:
    Loads participant data (Excel/CSV), validates and cleans it,
    then matches with CAT12 processed MRI files.
    Fully generalizable across multiple neuroimaging datasets.

    Supported Datasets:
        - IXI
        - AIBL
        - DLBS
        - GSP
        - NKI
        - OASIS
        - SALD

Usage:
    # From Jupyter Notebook:
    from data_loader import BrainAgeDataLoader

    loader = BrainAgeDataLoader(
        data_file="participants.xlsx",
        mri_folder="/path/to/mri/",
        id_regex=r'IXI(\\d+)'           # Change per dataset
    )
    df = loader.run_pipeline()

    # From Terminal:
    python data_loader.py participants.xlsx /path/to/mri/ --id_regex "IXI(\\d+)"

Output:
    <filename>_merged_final.csv with columns:
        - ID (first column of input file)
        - AGE
        - MRI_PATH

Author:
    Brain Age Prediction Project
Version:
    1.0.0
============================================================
"""

import pandas as pd
import os
import glob
import sys
import re
import logging
import nibabel as nib
import argparse
from typing import Optional

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),                          # Console output
        logging.FileHandler("brainage_loader.log")        # Log file
    ]
)

logger = logging.getLogger(__name__)

# ============================================================
# DATASET CONFIGURATIONS
# ============================================================
# Predefined regex patterns for supported datasets.
# Add new datasets here as needed.
# ============================================================

DATASET_CONFIGS = {
    "IXI":    r'IXI(\d+)',
    "AIBL":   r'AIBL(\d+)',
    "DLBS":   r'DLBS(\d+)',
    "GSP":    r'GSP(\d+)',
    "NKI":    r'NKI(\d+)',
    "OASIS":  r'OASIS(\d+)',
    "SALD":   r'SALD(\d+)',
}

# ============================================================
# BRAIN AGE DATA LOADER CLASS
# ============================================================

class BrainAgeDataLoader:
    """
    Generalized data loading and validation pipeline for brain age prediction.

    Supports any neuroimaging dataset by configuring the id_regex parameter.
    Handles Excel/CSV loading, automatic column detection, data cleaning,
    MRI file validation, and participant matching.

    Parameters:
        data_file (str): Path to input Excel (.xlsx, .xls) or CSV (.csv) file.
        mri_folder (str): Path to folder containing .nii.gz MRI files.
        id_regex (str): Regex pattern to extract numeric ID from MRI filenames.
                        Use predefined patterns from DATASET_CONFIGS or custom.
                        Default: IXI pattern

    Example:
        loader = BrainAgeDataLoader(
            data_file="participants.xlsx",
            mri_folder="/data/mri/",
            id_regex=DATASET_CONFIGS['IXI']
        )
        df_final = loader.run_pipeline()
    """

    def __init__(self, data_file: str, mri_folder: str, id_regex: str = r'IXI(\d+)'):
        self.data_file = data_file
        self.mri_folder = mri_folder
        self.id_regex = id_regex

        # Data states
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None

        # MRI states
        self.mri_files: list = []
        self.mri_ids: list = []

        # Detected column names
        self.id_col: Optional[str] = None
        self.age_col: Optional[str] = None
        self.date_col: Optional[str] = None

        # Extract prefix from regex for MRI path matching
        # e.g., IXI pattern -> "IXI"
        prefix_match = re.match(r'([A-Za-z]+)', self.id_regex)
        self.id_prefix = prefix_match.group(1) if prefix_match else ""

        logger.info(f"BrainAgeDataLoader initialized")
        logger.info(f"  Data file : {self.data_file}")
        logger.info(f"  MRI folder: {self.mri_folder}")
        logger.info(f"  ID regex  : {self.id_regex}")
        logger.info(f"  ID prefix : {self.id_prefix}")

    # ============================================================
    # 1. FILE LOADING
    # Supports: .xlsx, .xls, .csv
    # ============================================================

    def load_data(self) -> bool:
        """
        Load input data file.
        Automatically detects file type (.xlsx, .xls, .csv) and loads accordingly.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        try:
            file_ext = os.path.splitext(self.data_file)[1].lower()
            logger.info(f"Detected file type: '{file_ext}'")

            if file_ext in ['.xlsx', '.xls']:
                logger.info("Loading Excel file...")
                self.df_raw = pd.read_excel(self.data_file)
                logger.info(f"Excel file loaded successfully: {self.data_file}")

            elif file_ext == '.csv':
                logger.info("Loading CSV file...")
                self.df_raw = pd.read_csv(self.data_file)
                logger.info(f"CSV file loaded successfully: {self.data_file}")

            else:
                logger.error(f"Unsupported file type '{file_ext}'")
                logger.error("Supported types: .xlsx, .xls, .csv")
                return False

            logger.info(f"Shape: {self.df_raw.shape[0]} rows x {self.df_raw.shape[1]} columns")
            return True

        except FileNotFoundError:
            logger.error(f"File not found: {self.data_file}")
            return False
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return False

    # ============================================================
    # 2. COLUMN DETECTION
    # Auto-detects ID, AGE, and DATE columns
    # ============================================================

    def detect_columns(self) -> bool:
        """
        Automatically detect ID, AGE, and DATE columns.

        Rules:
            - ID   : Always the first column in the file.
            - AGE  : First column containing 'age' or 'alter' (case-insensitive).
            - DATE : First column containing 'date', 'datum', 'study', or 'scan'
                     (case-insensitive). Optional.

        Returns:
            bool: True if ID and AGE columns found, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("COLUMN DETECTION")
        logger.info("=" * 60)

        # --- ID column: first column ---
        self.id_col = self.df_raw.columns[0]
        logger.info(f"ID column detected: '{self.id_col}'")

        # --- AGE column ---
        age_keywords = ['age', 'alter']
        self.age_col = None
        for col in self.df_raw.columns:
            if any(kw in col.lower() for kw in age_keywords):
                self.age_col = col
                break

        if self.age_col is None:
            logger.error("AGE column not found!")
            logger.error("Ensure your file has a column with 'age' or 'alter' in its name (case-insensitive)")
            return False
        logger.info(f"AGE column detected: '{self.age_col}'")

        # --- DATE column (optional) ---
        date_keywords = ['study_date', 'scan_date', 'study', 'scan', 'date', 'datum']
        self.date_col = None
        for col in self.df_raw.columns:
            if any(kw in col.lower() for kw in date_keywords):
                self.date_col = col
                break

        if self.date_col is None:
            logger.warning("DATE column not found. All records will be used without date filtering.")
        else:
            logger.info(f"DATE column detected: '{self.date_col}'")

        return True

    # ============================================================
    # 3. DATA QUALITY CHECKS
    # ============================================================

    def check_missing_data(self) -> pd.DataFrame:
        """
        Analyze missing values across all columns.
        Highlights critical columns (ID, AGE, DATE).

        Returns:
            pd.DataFrame: Summary table of missing values per column.
        """
        logger.info("=" * 60)
        logger.info("MISSING DATA ANALYSIS")
        logger.info("=" * 60)

        missing_summary = []
        for col in self.df_raw.columns:
            missing_count = self.df_raw[col].isna().sum()
            missing_pct = round(100 * missing_count / len(self.df_raw), 2)
            missing_summary.append({
                "Column": col,
                "Missing_Count": missing_count,
                "Missing_Percentage": missing_pct
            })

        missing_df = pd.DataFrame(missing_summary)
        logger.info(f"\n{missing_df.to_string(index=False)}")

        # Critical columns check
        critical_cols = [self.id_col, self.age_col]
        if self.date_col:
            critical_cols.append(self.date_col)

        logger.info("--- Critical Columns ---")
        for col in critical_cols:
            missing = self.df_raw[col].isna().sum()
            if missing > 0:
                logger.warning(f"{col}: {missing} missing values")
            else:
                logger.info(f"{col}: No missing values")

        return missing_df

    def check_data_types(self):
        """
        Validate and convert data types for critical columns.
        - AGE  : Must be numeric. Checks for valid range (0-120).
        - DATE : Converted to datetime if present.
        """
        logger.info("=" * 60)
        logger.info("DATA TYPE VALIDATION")
        logger.info("=" * 60)

        # ID column type
        logger.info(f"ID column ({self.id_col}): {self.df_raw[self.id_col].dtype}")

        # AGE column -> numeric
        try:
            self.df_raw[self.age_col] = pd.to_numeric(self.df_raw[self.age_col], errors='coerce')
            logger.info(f"AGE column ({self.age_col}): Converted to numeric")

            age_min = self.df_raw[self.age_col].min()
            age_max = self.df_raw[self.age_col].max()
            age_mean = self.df_raw[self.age_col].mean()
            logger.info(f"  Age range: {age_min:.2f} - {age_max:.2f} years | Mean: {age_mean:.2f}")

            if age_min < 0 or age_max > 120:
                logger.warning("Suspicious age values detected! Check your data.")
        except Exception as e:
            logger.error(f"Failed to convert AGE to numeric: {e}")

        # DATE column -> datetime
        if self.date_col:
            try:
                self.df_raw[self.date_col] = pd.to_datetime(self.df_raw[self.date_col], errors='coerce')
                logger.info(f"DATE column ({self.date_col}): Converted to datetime")

                date_min = self.df_raw[self.date_col].min()
                date_max = self.df_raw[self.date_col].max()
                logger.info(f"  Date range: {date_min} to {date_max}")
            except Exception as e:
                logger.error(f"Failed to convert DATE to datetime: {e}")

    def check_duplicates(self):
        """
        Check for duplicate participant IDs.
        Reports how duplicates will be handled based on DATE column availability.
        """
        logger.info("=" * 60)
        logger.info("DUPLICATE CHECK")
        logger.info("=" * 60)

        total_rows = len(self.df_raw)
        unique_ids = self.df_raw[self.id_col].nunique()
        duplicates = total_rows - unique_ids

        logger.info(f"Total rows    : {total_rows}")
        logger.info(f"Unique IDs    : {unique_ids}")
        logger.info(f"Duplicate IDs : {duplicates}")

        if duplicates > 0:
            if self.date_col:
                logger.warning(f"Multiple records detected -> will keep most recent based on '{self.date_col}'")
            else:
                logger.warning("Multiple records detected, no DATE column -> will keep first occurrence")
        else:
            logger.info("No duplicate IDs found")

    # ============================================================
    # 4. DATA CLEANING
    # ============================================================

    def clean_data(self) -> bool:
        """
        Clean participant data:
            1. Handle duplicates (keep most recent if DATE exists, else first)
            2. Remove rows with missing AGE
            3. Keep only ID and AGE columns

        Returns:
            bool: True if valid data remains, False if empty after cleaning.
        """
        logger.info("=" * 60)
        logger.info("DATA CLEANING")
        logger.info("=" * 60)

        initial_rows = len(self.df_raw)

        # --- Handle duplicates ---
        if self.date_col:
            self.df_raw[self.date_col] = pd.to_datetime(self.df_raw[self.date_col], errors='coerce')
            df_sorted = self.df_raw.sort_values(self.date_col)
            df_last = df_sorted.groupby(self.id_col).last().reset_index()
            self.df_clean = df_last[[self.id_col, self.age_col]].copy()
            logger.info(f"Kept most recent record per ID based on '{self.date_col}'")
        else:
            self.df_clean = self.df_raw[[self.id_col, self.age_col]].drop_duplicates(
                subset=[self.id_col], keep='first'
            ).copy()
            logger.info("Kept first occurrence per ID (no DATE column)")

        after_dedup = len(self.df_clean)

        # --- Remove missing AGE ---
        self.df_clean = self.df_clean.dropna(subset=[self.age_col])
        after_clean = len(self.df_clean)

        # --- Summary ---
        logger.info("--- Cleaning Summary ---")
        logger.info(f"  Initial rows          : {initial_rows}")
        logger.info(f"  After deduplication   : {after_dedup}")
        logger.info(f"  After removing NaN AGE: {after_clean}")

        if after_clean == 0:
            logger.error("No valid data remaining after cleaning!")
            return False

        logger.info(f"  Final clean data      : {after_clean} participants")
        return True

    # ============================================================
    # 5. MRI FILE HANDLING
    # ============================================================

    def load_mri_files(self) -> bool:
        """
        Scan MRI folder for .nii.gz files.

        Returns:
            bool: True if files found, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("MRI FILE LOADING")
        logger.info("=" * 60)

        if not os.path.exists(self.mri_folder):
            logger.error(f"MRI folder not found: {self.mri_folder}")
            return False

        pattern = os.path.join(self.mri_folder, "*.nii.gz")
        self.mri_files = sorted(glob.glob(pattern))

        if len(self.mri_files) == 0:
            logger.error(f"No .nii.gz files found in: {self.mri_folder}")
            return False

        logger.info(f"Found {len(self.mri_files)} MRI files")

        # Preview first and last 5 files
        logger.info("First 5 files:")
        for i, f in enumerate(self.mri_files[:5], 1):
            logger.info(f"  {i}. {os.path.basename(f)}")

        if len(self.mri_files) > 10:
            logger.info("  ...")
            logger.info("Last 5 files:")
            for i, f in enumerate(self.mri_files[-5:], len(self.mri_files) - 4):
                logger.info(f"  {i}. {os.path.basename(f)}")

        return True

    def extract_mri_ids(self) -> bool:
        """
        Extract numeric IDs from MRI filenames using self.id_regex.

        Example:
            Regex  : DATASET_CONFIGS['IXI']
            File   : mwp1sub-IXI002_T1w.nii.gz
            Result : 2

        Returns:
            bool: True if IDs extracted successfully, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("MRI ID EXTRACTION")
        logger.info("=" * 60)
        logger.info(f"Regex pattern: {self.id_regex}")

        self.mri_ids = []
        failed_files = []

        for file in self.mri_files:
            try:
                filename = os.path.basename(file)
                match = re.search(self.id_regex, filename)
                if match:
                    mri_id = int(match.group(1))
                    self.mri_ids.append(mri_id)
                else:
                    failed_files.append(filename)
            except Exception as e:
                failed_files.append(os.path.basename(file))
                logger.debug(f"Error processing {file}: {e}")

        logger.info(f"Successfully extracted IDs: {len(self.mri_ids)}")

        if failed_files:
            logger.warning(f"Failed to extract ID from {len(failed_files)} files")
            for f in failed_files[:5]:
                logger.warning(f"  - {f}")
            if len(failed_files) > 5:
                logger.warning(f"  ... and {len(failed_files) - 5} more")

        if len(self.mri_ids) == 0:
            logger.error("Could not extract any IDs!")
            logger.error(f"Check your regex pattern: {self.id_regex}")
            logger.error("Available preset patterns:")
            for name, pattern in DATASET_CONFIGS.items():
                logger.error(f"  {name}: {pattern}")
            return False

        # Duplicate check
        unique_count = len(set(self.mri_ids))
        if unique_count < len(self.mri_ids):
            logger.warning(f"Duplicate MRI IDs found! Total: {len(self.mri_ids)}, Unique: {unique_count}")
        else:
            logger.info(f"All MRI IDs are unique: {unique_count}")

        return True

    def check_mri_shapes(self) -> bool:
        """
        Validate that all MRI files have consistent 3D dimensions.
        Inconsistent shapes may cause issues in later processing (Module 2+).

        Returns:
            bool: True if all shapes match, False if inconsistent.
        """
        logger.info("=" * 60)
        logger.info("MRI SHAPE VALIDATION")
        logger.info("=" * 60)
        logger.info("Checking MRI dimensions (this may take a moment)...")

        shapes = []
        failed_files = []

        for file in self.mri_files:
            try:
                img = nib.load(file)
                shapes.append(img.shape)
            except Exception as e:
                failed_files.append((os.path.basename(file), str(e)))

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} MRI files:")
            for f, err in failed_files[:3]:
                logger.warning(f"  - {f}: {err}")

        unique_shapes = set(shapes)

        if len(unique_shapes) == 1:
            logger.info(f"All MRI files have consistent shape: {shapes[0]}")
            return True
        else:
            logger.warning(f"Inconsistent MRI shapes found! ({len(unique_shapes)} different shapes)")
            for shape in unique_shapes:
                count = shapes.count(shape)
                logger.warning(f"  - {shape}: {count} files")
            return False

    # ============================================================
    # 6. DATA-MRI MATCHING
    # ============================================================

    def match_data_mri(self) -> pd.DataFrame:
        """
        Match cleaned participant data with MRI files.
        Only participants present in BOTH data and MRI folder are kept.

        Returns:
            pd.DataFrame: Matched data with columns [ID, AGE, MRI_PATH]
        """
        logger.info("=" * 60)
        logger.info("DATA-MRI MATCHING")
        logger.info("=" * 60)

        data_ids = set(self.df_clean[self.id_col].values)
        mri_ids_set = set(self.mri_ids)

        common_ids = sorted(data_ids.intersection(mri_ids_set))
        only_data = sorted(data_ids - mri_ids_set)
        only_mri = sorted(mri_ids_set - data_ids)

        logger.info(f"Participants in data file : {len(data_ids)}")
        logger.info(f"Participants in MRI folder: {len(mri_ids_set)}")
        logger.info(f"Matched (BOTH)            : {len(common_ids)}")

        if only_data:
            logger.warning(f"In data only (no MRI): {len(only_data)}")
            if len(only_data) <= 10:
                logger.warning(f"  IDs: {only_data}")

        if only_mri:
            logger.warning(f"In MRI only (no data): {len(only_mri)}")
            if len(only_mri) <= 10:
                logger.warning(f"  IDs: {only_mri}")

        if len(common_ids) == 0:
            logger.error("No matching participants found between data and MRI!")
            logger.error("Check your id_regex pattern or data file.")
            return pd.DataFrame()

        # Filter to matched participants only
        df_matched = self.df_clean[self.df_clean[self.id_col].isin(common_ids)].copy()
        df_matched = df_matched.sort_values(self.id_col).reset_index(drop=True)

        # Add MRI file paths
        df_matched['MRI_PATH'] = df_matched[self.id_col].apply(self._get_mri_path)

        # Verify all paths are valid
        empty_paths = df_matched[df_matched['MRI_PATH'] == '']
        if len(empty_paths) > 0:
            logger.warning(f"Could not find MRI path for {len(empty_paths)} participants")

        return df_matched

    def _get_mri_path(self, subject_id: int) -> str:
        """
        Find the MRI file path for a given numeric subject ID.
        Uses the id_prefix extracted from id_regex for matching.

        Args:
            subject_id: Numeric participant ID.

        Returns:
            str: Full file path to the MRI file, or empty string if not found.
        """
        # Zero-padded ID: e.g., IXI002, AIBL018, OASIS001
        formatted_id = f"{self.id_prefix}{subject_id:03d}"

        for file in self.mri_files:
            if formatted_id in os.path.basename(file):
                return file

        return ""

    # ============================================================
    # 7. SAVE RESULTS
    # ============================================================

    def save_results(self, df_final: pd.DataFrame, output_path: str):
        """
        Save the final merged DataFrame to CSV.

        Args:
            df_final: Final matched DataFrame.
            output_path: Path for the output CSV file.
        """
        logger.info("=" * 60)
        logger.info("SAVING RESULTS")
        logger.info("=" * 60)

        try:
            df_final.to_csv(output_path, index=False)
            logger.info(f"Final data saved: {output_path}")
            logger.info(f"Total participants: {len(df_final)}")
            logger.info(f"Columns: {list(df_final.columns)}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")

    # ============================================================
    # 8. MAIN PIPELINE
    # ============================================================

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete Module 1 pipeline:
            1. Load data file
            2. Detect columns (ID, AGE, DATE)
            3. Quality checks (missing, types, duplicates)
            4. Clean data
            5. Load MRI files
            6. Extract MRI IDs
            7. Validate MRI shapes
            8. Match data with MRI

        Returns:
            pd.DataFrame: Final matched data [ID, AGE, MRI_PATH], or empty DataFrame on failure.
        """
        logger.info("=" * 60)
        logger.info("BRAIN AGE PREDICTION - MODULE 1")
        logger.info("Data Loading, Validation & Matching")
        logger.info("=" * 60)

        # Step 1: Load data
        if not self.load_data():
            return pd.DataFrame()

        # Step 2: Detect columns
        if not self.detect_columns():
            return pd.DataFrame()

        # Step 3: Quality checks
        self.check_missing_data()
        self.check_data_types()
        self.check_duplicates()

        # Step 4: Clean data
        if not self.clean_data():
            return pd.DataFrame()

        # Step 5: Load MRI files
        if not self.load_mri_files():
            return pd.DataFrame()

        # Step 6: Extract MRI IDs
        if not self.extract_mri_ids():
            return pd.DataFrame()

        # Step 7: Validate MRI shapes
        self.check_mri_shapes()

        # Step 8: Match data with MRI
        df_final = self.match_data_mri()

        return df_final


# ============================================================
# CLI INTERFACE
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Brain Age Prediction - Module 1: Data Loading & Matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # IXI dataset
  python data_loader.py participants.xlsx /path/to/mri/ --id_regex "IXI(\\d+)"

  # AIBL dataset
  python data_loader.py AIBL_data.csv /path/to/mri/ --dataset AIBL

  # Custom regex
  python data_loader.py data.xlsx /mri/ --id_regex "sub-([0-9]+)"

Supported preset datasets: IXI, AIBL, DLBS, GSP, NKI, OASIS, SALD
        """
    )

    parser.add_argument("data_file", type=str, help="Path to input Excel/CSV file")
    parser.add_argument("mri_folder", type=str, help="Path to MRI folder (.nii.gz files)")
    parser.add_argument("--dataset", type=str, choices=DATASET_CONFIGS.keys(),
                        help="Preset dataset name (sets id_regex automatically)")
    parser.add_argument("--id_regex", type=str, default=None,
                        help="Custom regex pattern to extract ID from MRI filenames")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path (default: <input_name>_merged_final.csv)")

    return parser.parse_args()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main(data_file: str = None, mri_folder: str = None,
         id_regex: str = None, dataset: str = None, output: str = None):
    """
    Main execution function. Supports both CLI and Jupyter Notebook usage.

    Args:
        data_file (str): Path to input Excel/CSV file.
        mri_folder (str): Path to MRI folder.
        id_regex (str): Custom regex for ID extraction. Overrides dataset preset.
        dataset (str): Preset dataset name from DATASET_CONFIGS.
        output (str): Output file path. Auto-generated if None.

    Returns:
        pd.DataFrame: Final merged data, or None on failure.

    Examples:
        # Jupyter - IXI dataset
        df = main("IXI.xlsx", "/mri/ixi/", dataset="IXI")

        # Jupyter - AIBL dataset
        df = main("AIBL.csv", "/mri/aibl/", dataset="AIBL")

        # Jupyter - Custom regex
        df = main("data.xlsx", "/mri/", id_regex=r'sub-([0-9]+)')
    """

    # --- Resolve arguments (CLI or function call) ---
    if data_file is None or mri_folder is None:
        args = parse_arguments()
        data_file = args.data_file
        mri_folder = args.mri_folder
        dataset = args.dataset
        id_regex = args.id_regex
        output = args.output

    # --- Resolve id_regex ---
    if id_regex is not None:
        # Custom regex takes priority
        resolved_regex = id_regex
        logger.info(f"Using custom regex: {resolved_regex}")
    elif dataset is not None:
        # Preset dataset
        resolved_regex = DATASET_CONFIGS[dataset]
        logger.info(f"Using preset dataset '{dataset}': {resolved_regex}")
    else:
        # Default fallback
        resolved_regex = DATASET_CONFIGS["IXI"]
        logger.warning("No dataset or regex specified. Defaulting to IXI pattern.")

    # --- Resolve output path ---
    if output is None:
        base_name = os.path.splitext(os.path.basename(data_file))[0]
        output = f"{base_name}_merged_final.csv"

    # --- Run pipeline ---
    loader = BrainAgeDataLoader(
        data_file=data_file,
        mri_folder=mri_folder,
        id_regex=resolved_regex
    )

    df_final = loader.run_pipeline()

    # --- Save or report failure ---
    if not df_final.empty:
        loader.save_results(df_final, output)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Output file: {output}")
        logger.info("Ready for Module 2: Feature Extraction")

        return df_final
    else:
        logger.error("=" * 60)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 60)
        logger.error("Check log messages above for details.")
        return None


if __name__ == "__main__":
    main()
