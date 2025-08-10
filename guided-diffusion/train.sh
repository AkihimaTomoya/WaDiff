#!/bin/bash

echo "🚀 Starting training..."

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))
export OMP_NUM_THREADS=11
export PYTHONPATH=/kaggle/working/WaDiff/guided-diffusion:$PYTHONPATH

OUTPUT_DIR="/kaggle/working/outputs"
mkdir -p "$OUTPUT_DIR"

DATA_DIR="/kaggle/working/furniture-split-data/val"
PRETRAINED_DIFFUSION="/kaggle/working/WaDiff/guided-diffusion/models/256x256_diffusion_uncond.pt"
PRETRAINED_DECODER="/kaggle/input/steg/pytorch/default/1/final_decoder_48.pth"

MODEL_FLAGS="--wm_length 48 --attention_resolutions 32,16,8 --class_cond False --image_size 256 --num_channels 256 --learn_sigma True --num_head_channels 64 --num_res_blocks 2 --resblock_updown True"
DIFFUSION_FLAGS="--diffusion_steps 100 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 1 --microbatch 0 --log_interval 100 --save_interval 5000 --use_fp16 True"
WATERMARK_FLAGS="--alpha 0.4 --threshold 400"

LOG_FILE="$OUTPUT_DIR/train_$(date +'%Y%m%d_%H%M%S').log"

python scripts/image_train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --resume_checkpoint $PRETRAINED_DIFFUSION \
    --wm_decoder_path $PRETRAINED_DECODER \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
    $WATERMARK_FLAGS 2>&1 | tee "$LOG_FILE"

echo "📄 Last 10 lines of log:"
tail -n 10 "$LOG_FILE"
