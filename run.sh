#!/bin/bash

# ==========================================
# Configuration Parameters
# ==========================================
DATASET="Games"

#Default
ITEM_MAX_LEN=50
USER_MAX_LEN=50
USER_MIN_LEN=5
RW_LENGTH=3
RW_WIDTH=10

VERSION=4
CA="_CA_Gate"


if [ "$VERSION" -eq 1 ]; then
    RW_LENGTH=3
elif [ "$VERSION" -eq 2 ]; then
    RW_LENGTH=5
    RW_WIDTH=10
elif [ "$VERSION" -eq 3 ]; then
    RW_LENGTH=10
    RW_WIDTH=10
elif [ "$VERSION" -eq 4 ]; then
    RW_LENGTH=10
    RW_WIDTH=10
fi


GENERATE_DATA=0   # 0 ou 1
MAX_ROWS=10000
JOBS=3
GPU_ID=0

# Create results folder if it doesn't exist
mkdir -p ./results

# Timestamp for unique logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_GEN="./results/data_${DATASET}_V${VERSION}${CA}_${TIMESTAMP}.log"
LOG_TRAIN="./results/train_${DATASET}_V${VERSION}${CA}_${TIMESTAMP}.log"

# ==========================================
# 1. Run Data Generation (Foreground)
# ==========================================
echo "Starting Data Generation (Version ${VERSION})..."
echo "Logging to: ${LOG_GEN}"

if [ "$GENERATE_DATA" = 1 ]; then

    python -u new_data.py \
    --data=${DATASET} \
    --job=${JOBS} \
    --item_max_length=${ITEM_MAX_LEN} \
    --user_max_length=${USER_MAX_LEN} \
    --user_min_length=${USER_MIN_LEN} \
    --rw_length=${RW_LENGTH} \
    --rw_width=${RW_WIDTH} \
    --version=${VERSION} \
    --max_rows=${MAX_ROWS} \
    --force_graph=True \
    > "${LOG_GEN}" 2>&1

    # Check if data generation was successful
    if [ $? -ne 0 ]; then
        echo "❌ Data generation FAILED. Stopping experiment."
        echo "Check error log: ${LOG_GEN}"
        exit 1
    fi

    echo "✅ Data Generation Complete."

fi

# ==========================================
# 2. Run Model Training (Background)
# ==========================================
echo "Starting Model Training..."
echo "Logging to: ${LOG_TRAIN}"

nohup python -u new_main${CA}.py \
 --data=${DATASET} \
 --gpu=${GPU_ID} \
 --epoch=20 \
 --hidden_size=50 \
 --batchSize=50 \
 --user_long=orgat \
 --user_short=att \
 --item_long=orgat \
 --item_short=att \
 --user_update=rnn \
 --item_update=rnn \
 --lr=0.001 \
 --l2=0.0001 \
 --layer_num=2 \
 --item_max_length=${ITEM_MAX_LEN} \
 --user_max_length=${USER_MAX_LEN} \
 --attn_drop=0.3 \
 --feat_drop=0.3 \
 --rw_length=${RW_LENGTH} \
 --rw_width=${RW_WIDTH} \
 --version=${VERSION} \
 --max_rows=${MAX_ROWS} \
 --record \
 > "${LOG_TRAIN}" 2>&1 &

echo "🚀 Training started with PID $!"