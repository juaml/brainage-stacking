# BrainAge – Module 1: Data Loader

This module prepares input data for brain age prediction.

It matches participant information (CSV / Excel) with CAT12-processed
T1-weighted MRI files (`.nii.gz`) and produces a clean, unified table.

This output is used directly by **Module 2 (Feature Extraction)**.

---

## What this module does

- Loads participant data from `.csv`, `.xls`, or `.xlsx`
- Cleans and validates age and ID columns
- Matches participant IDs with MRI filenames
- Outputs a single, clean CSV file

---

## Input

### 1. Participant file
- Format: `.csv`, `.xls`, `.xlsx`
- Must contain:
  - Participant ID (first column)
  - Age column (`age` or `alter`, case-insensitive)

### 2. MRI folder
- Contains CAT12 gray-matter images  
- File format: `.nii.gz`

---

## Output

A single CSV file:

`<input_name>_merged_final.csv`

Columns:
- `ID`
- `AGE`
- `MRI_PATH`
