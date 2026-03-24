#!/bin/bash
##############################################
# Batch Submit Script
# Submits 10 parallel jobs (1 region each)
##############################################

echo "Submitting 10 parallel jobs (Region 1-10)..."
echo ""

# Features directory (same for all)
FEATURES_DIR="/home/fkarateke/IXI_cat12.8.1_mwp1/features"
BASE_OUTPUT="/home/fkarateke/brainage-stacking/models/parallel_10roi"
PRESET="fast_test"

# Submit 10 jobs (1 region each)
for region in {1..10}; do
    OUTPUT_DIR="${BASE_OUTPUT}/region_${region}"
    
    echo "Submitting Region ${region}..."
    
    condor_submit train_stacking.sub \
        features_dir=${FEATURES_DIR} \
        output_dir=${OUTPUT_DIR} \
        start_region=${region} \
        end_region=${region} \
        preset=${PRESET}
    
    echo ""
done

echo "✅ All 10 jobs submitted!"
echo ""
echo "Monitor with: condor_q"
echo "Check results: ls -lh ${BASE_OUTPUT}/*/results_summary*.csv"
