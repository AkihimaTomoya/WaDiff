export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MODEL_FLAGS="--wm_length 48 --attention_resolutions 32,16,8 --class_cond False --image_size 256 --num_channels 256 --learn_sigma True --num_head_channels 64 --num_res_blocks 2 --resblock_updown True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4"
NUM_GPUS=1
mpiexec -n $NUM_GPUS python scripts/image_train.py --alpha 0.4 --threshold 400 --wm_decoder_path ./ --data_dir ../../../../data/imagenet/val --resume_checkpoint models/256x256_diffusion_uncond.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

