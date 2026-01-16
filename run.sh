#!/bin/bash

# ==========================================
# Configuration Parameters
# ==========================================
DATASET="Games"

#Default
ITEM_MAX_LEN=50
USER_MAX_LEN=50
RW_LENGTH=10
RW_WIDTH=10

VERSION=2
CA="_CA_Gate"
#CA=""

GENERATE_DATA=1   # 0 ou 1
N_USERS=0  #5000       # 0 disabled
JOBS=2
GPU_ID=0
RUN_MODEL=1       # 0 ou 1 


# ==========================================

mkdir -p ./results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_GEN="./results/data_${DATASET}_V${VERSION}${CA}_${TIMESTAMP}.log"
LOG_TRAIN="./results/train_${DATASET}_V${VERSION}${CA}_${TIMESTAMP}.log"

# ==========================================
# 1. Run Data Generation (Foreground)
# ==========================================
if [ "$GENERATE_DATA" = 1 ]; then

    echo "Starting Data Generation (Version ${VERSION})..."
    echo "Logging to: ${LOG_GEN}"

    python -u new_data.py \
    --data=${DATASET} \
    --job=${JOBS} \
    --item_max_length=${ITEM_MAX_LEN} \
    --user_max_length=${USER_MAX_LEN} \
    --rw_length=${RW_LENGTH} \
    --rw_width=${RW_WIDTH} \
    --version=${VERSION} \
    --n_users=${N_USERS} \
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
if [ "$RUN_MODEL" = 1 ]; then

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
    --n_users=${N_USERS} \
    --version=${VERSION} \
    --record \
    > "${LOG_TRAIN}" 2>&1 &

    echo "🚀 Training started with PID $!"

fi