"""
============================================================
Tests for Module 1: Data Loader
============================================================
Run:
    python -m pytest tests/test_data_loader.py -v
============================================================
"""

import pytest
import pandas as pd
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import BrainAgeDataLoader, DATASET_CONFIGS


# ============================================================
# FIXTURES - Test Data
# ============================================================

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        "IXI_ID": [1, 2, 3, 4, 5],
        "AGE": [35.5, 42.0, None, 58.2, 29.1],
        "STUDY_DATE": ["2020-01-01", "2020-02-15", "2020-03-10", "2020-04-20", "2020-05-05"],
        "SEX": [1, 2, 1, 2, 1]
    })
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_csv_no_age(tmp_path):
    """CSV without AGE column."""
    df = pd.DataFrame({
        "IXI_ID": [1, 2, 3],
        "SEX": [1, 2, 1]
    })
    file_path = tmp_path / "no_age.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_csv_german_age(tmp_path):
    """CSV with German 'Alter' column."""
    df = pd.DataFrame({
        "Subject_ID": [1, 2, 3],
        "Alter": [30.0, 45.5, 60.2],
        "Datum": ["2020-01-01", "2020-02-01", "2020-03-01"]
    })
    file_path = tmp_path / "german_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def empty_mri_folder(tmp_path):
    """Empty folder (no MRI files)."""
    folder = tmp_path / "empty_mri"
    folder.mkdir()
    return str(folder)


# ============================================================
# TEST: DATASET CONFIGS
# ============================================================

class TestDatasetConfigs:

    def test_all_presets_exist(self):
        """All 7 supported datasets should be in DATASET_CONFIGS."""
        expected = ["IXI", "AIBL", "DLBS", "GSP", "NKI", "OASIS", "SALD"]
        for name in expected:
            assert name in DATASET_CONFIGS

    def test_regex_patterns_valid(self):
        """All preset regex patterns should be compilable."""
        import re
        for name, pattern in DATASET_CONFIGS.items():
            compiled = re.compile(pattern)
            assert compiled is not None


# ============================================================
# TEST: INITIALIZATION
# ============================================================

class TestInitialization:

    def test_init_default_regex(self, sample_csv, empty_mri_folder):
        """Default regex should be IXI pattern."""
        loader = BrainAgeDataLoader(sample_csv, empty_mri_folder)
        assert loader.id_regex == r'IXI(\d+)'
        assert loader.id_prefix == "IXI"

    def test_init_custom_regex(self, sample_csv, empty_mri_folder):
        """Custom regex should be stored correctly."""
        loader = BrainAgeDataLoader(sample_csv, empty_mri_folder, id_regex=r'AIBL(\d+)')
        assert loader.id_regex == r'AIBL(\d+)'
        assert loader.id_prefix == "AIBL"

    def test_init_preset_datasets(self, sample_csv, empty_mri_folder):
        """All preset dataset regex patterns should extract prefix correctly."""
        expected_prefixes = {
            "IXI": "IXI", "AIBL": "AIBL", "DLBS": "DLBS",
            "GSP": "GSP", "NKI": "NKI", "OASIS": "OASIS", "SALD": "SALD"
        }
        for name, pattern in DATASET_CONFIGS.items():
            loader = BrainAgeDataLoader(sample_csv, empty_mri_folder, id_regex=pattern)
            assert loader.id_prefix == expected_prefixes[name]


# ============================================================
# TEST: FILE LOADING
# ============================================================

class TestLoadData:

    def test_load_csv_success(self, sample_csv, empty_mri_folder):
        """CSV file should load successfully."""
        loader = BrainAgeDataLoader(sample_csv, empty_mri_folder)
        assert loader.load_data() == True
        assert loader.df_raw is not None
        assert len(loader.df_raw) == 5

    def test_load_file_not_found(self, empty_mri_folder):
        """Non-existent file should return False."""
        loader = BrainAgeDataLoader("nonexistent.csv", empty_mri_folder)
        assert loader.load_data() == False

    def test_load_unsupported_type(self, tmp_path, empty_mri_folder):
        """Unsupported file type should return False."""
        file_path = tmp_path / "data.txt"
        file_path.write_text("test")
        loader = BrainAgeDataLoader(str(file_path), empty_mri_folder)
        assert loader.load_data() == False


# ============================================================
# TEST: COLUMN DETECTION
# ============================================================

class TestDetectColumns:

    def test_detect_english_columns(self, sample_csv, empty_mri_folder):
        """Should detect ID, AGE, and DATE columns in English."""
        loader = BrainAgeDataLoader(sample_csv, empty_mri_folder)
        loader.load_data()
        assert loader.detect_columns() == True
        assert loader.id_col == "IXI_ID"
        assert loader.age_col == "AGE"
        assert loader.date_col == "STUDY_DATE"

    def test_detect_german_columns(self, sample_csv_german_age, empty_mri_folder):
        """Should detect 'Alter' (German) as AGE column."""
        loader = BrainAgeDataLoader(sample_csv_german_age, empty_mri_folder)
        loader.load_data()
        assert loader.detect_columns() == True
        assert loader.age_col == "Alter"
        assert loader.date_col == "Datum"

    def test_detect_no_age_column(self, sample_csv_no_age, empty_mri_folder):
        """Should return False if no AGE column exists."""
        loader = BrainAgeDataLoader(sample_csv_no_age, empty_mri_folder)
        loader.load_data()
        assert loader.detect_columns() == False


# ============================================================
# TEST: DATA CLEANING
# ============================================================

class TestCleanData:

    def test_clean_removes_missing_age(self, sample_csv, empty_mri_folder):
        """Rows with missing AGE should be removed."""
        loader = BrainAgeDataLoader(sample_csv, empty_mri_folder)
        loader.load_data()
        loader.detect_columns()
        loader.check_data_types()
        assert loader.clean_data() == True
        # Original: 5 rows, 1 with NaN AGE -> 4 remain
        assert len(loader.df_clean) == 4

    def test_clean_keeps_only_id_age(self, sample_csv, empty_mri_folder):
        """Clean data should only have ID and AGE columns."""
        loader = BrainAgeDataLoader(sample_csv, empty_mri_folder)
        loader.load_data()
        loader.detect_columns()
        loader.check_data_types()
        loader.clean_data()
        assert list(loader.df_clean.columns) == ["IXI_ID", "AGE"]


# ============================================================
# TEST: MRI HANDLING
# ============================================================

class TestMRIHandling:

    def test_mri_folder_not_found(self, sample_csv):
        """Non-existent MRI folder should return False."""
        loader = BrainAgeDataLoader(sample_csv, "/nonexistent/path/")
        assert loader.load_mri_files() == False

    def test_mri_folder_empty(self, sample_csv, empty_mri_folder):
        """Empty MRI folder should return False."""
        loader = BrainAgeDataLoader(sample_csv, empty_mri_folder)
        assert loader.load_mri_files() == False

    def test_extract_ids_no_files(self, sample_csv, empty_mri_folder):
        """No MRI files should return False."""
        loader = BrainAgeDataLoader(sample_csv, empty_mri_folder)
        loader.mri_files = []
        assert loader.extract_mri_ids() == False


# ============================================================
# TEST: ID REGEX PATTERNS
# ============================================================

class TestIDRegex:

    def test_ixi_regex(self):
        """IXI regex should extract correct ID."""
        import re
        pattern = DATASET_CONFIGS["IXI"]
        match = re.search(pattern, "mwp1sub-IXI002_T1w.nii.gz")
        assert match is not None
        assert int(match.group(1)) == 2

    def test_aibl_regex(self):
        """AIBL regex should extract correct ID."""
        import re
        pattern = DATASET_CONFIGS["AIBL"]
        match = re.search(pattern, "sub-AIBL0018_T1w.nii.gz")
        assert match is not None
        assert int(match.group(1)) == 18

    def test_oasis_regex(self):
        """OASIS regex should extract correct ID."""
        import re
        pattern = DATASET_CONFIGS["OASIS"]
        match = re.search(pattern, "sub-OASIS001_T1w.nii.gz")
        assert match is not None
        assert int(match.group(1)) == 1


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
