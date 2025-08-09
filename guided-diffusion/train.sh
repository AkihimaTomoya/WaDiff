#!/bin/bash

# Kaggle training script for watermarked diffusion model
# This script runs on Kaggle T4 x2 GPUs without MPI

# Set up environment
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

# Model configuration
MODEL_FLAGS="--wm_length 48 --attention_resolutions 32,16,8 --class_cond False --image_size 256 --num_channels 256 --learn_sigma True --num_head_channels 64 --num_res_blocks 2 --resblock_updown True"

# Diffusion configuration  
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

# Training configuration
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --microbatch 1 --log_interval 100 --save_interval 5000 --use_fp16 True"

# Watermark configuration
WATERMARK_FLAGS="--alpha 0.4 --threshold 400"

# Paths
DATA_DIR="/kaggle/working/furniture-split-data/val"
OUTPUT_DIR="/kaggle/working/outputs"
PRETRAINED_DIFFUSION="/kaggle/working/WaDiff/guided-diffusion/models/256x256_diffusion_uncond.pt"
PRETRAINED_DECODER="/kaggle/input/steg/pytorch/default/1/final_decoder_48.pth"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python scripts/image_train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --resume_checkpoint $PRETRAINED_DIFFUSION \
    --wm_decoder_path $PRETRAINED_DECODER \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
    $WATERMARK_FLAGS

echo "Training completed!"
