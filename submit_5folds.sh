#!/bin/bash
##############################################
# Batch Submit Script
# Submits 5 parallel folds (870 regions each)
##############################################

echo "============================================"
echo "SUBMITTING 5 PARALLEL FOLDS (870 REGIONS)"
echo "============================================"
echo ""

# Configuration
FEATURES_DIR="/home/fkarateke/IXI_cat12.8.1_mwp1/features"
OUTPUT_DIR="/home/fkarateke/brainage-stacking/models/full_870roi_5folds"
START_REGION=1
END_REGION=870
PRESET="full_paper"

echo "Features dir : ${FEATURES_DIR}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Regions      : ${START_REGION}-${END_REGION} (870 regions)"
echo "Preset       : ${PRESET}"
echo "Folds        : 5 (paralel)"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Submit 5 folds
for fold in {1..5}; do
    echo "Submitting Fold ${fold}/5..."
    
    condor_submit train_stacking_fold.sub \
        features_dir=${FEATURES_DIR} \
        output_dir=${OUTPUT_DIR} \
        fold=${fold} \
        start_region=${START_REGION} \
        end_region=${END_REGION} \
        preset=${PRESET}
    
    echo ""
done

echo "============================================"
echo "✅ ALL 5 FOLDS SUBMITTED!"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  condor_q"
echo ""
echo "Check results:"
echo "  ls -lh ${OUTPUT_DIR}/results_summary_fold*.csv"
echo ""
echo "Estimated time: 24-48 hours per fold"
echo "With parallelization: All done in ~24-48 hours!"
echo ""
