# BrainAge Stacking Pipeline

Modular pipeline for brain age prediction using CAT12-processed structural MRI data.

## Modules

### Module 1 – Data Loader
Prepares participant metadata and matches CAT12 gray-matter images.
Location: `module1/`

### Module 2 – Feature Extraction
Extracts regional gray-matter volume features using brain atlases (870 regions).
Location: `module2/`

### Module 3 – Stacking Model Training
Trains a region-wise stacking ensemble for brain age prediction using HTCondor parallelization.
Location: `module3/`

## Pipeline
```
Module 1 → Module 2 → Module 3
CSV/MRI     Features    Brain Age Model
```
