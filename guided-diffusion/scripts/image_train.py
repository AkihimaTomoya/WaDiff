"""
Train a diffusion model on images for Kaggle environment.
"""

import argparse
import sys
import os
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
import atexit

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.stega_model import StegaStampDecoder


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except:
            pass


def main_worker(rank, world_size, args):
    """Main worker function for distributed training."""
    try:
        # Setup environment variables
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
        # Set CUDA device for this process
        if th.cuda.is_available():
            th.cuda.set_device(rank % th.cuda.device_count())
        
        # Setup distributed training
        dist_util.setup_dist()
        logger.configure(dir=args.output_dir)

        if rank == 0:
            logger.log("creating model and diffusion...")
        
        # Create watermarked model
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys()), 
            wm_length=args.wm_length
        )

        # Create the original model architecture (without watermark)
        ori_model, _ = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys()), 
            wm_length=0
        )
        
        # Freeze original model parameters
        for param in ori_model.parameters():
            param.requires_grad = False
        
        ori_model.eval()
        ori_model.to(dist_util.dev())
        model.to(dist_util.dev())

        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
        
        # Define watermark decoder
        wm_decoder = None
        if args.wm_length > 0 and isinstance(args.wm_length, int):
            wm_decoder = StegaStampDecoder(
                args.image_size,
                3,
                args.wm_length,
            )
            wm_decoder.to(dist_util.dev())

        if rank == 0:
            logger.log("creating data loader...")
        
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
        
        if rank == 0:
            logger.log("training...")
            logger.log(f"GPU Memory allocated: {th.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.log(f"GPU Memory reserved: {th.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            ori_model=ori_model,
            wm_length=args.wm_length,
            alpha=args.alpha,
            threshold=args.threshold,
            wm_decoder=wm_decoder,
            wm_decoder_path=args.wm_decoder_path
        ).run_loop()
        
    except Exception as e:
        if rank == 0:
            logger.log(f"Error in main_worker: {e}")
            if th.cuda.is_available():
                logger.log(f"GPU Memory allocated: {th.cuda.memory_allocated() / 1024**3:.2f} GB")
                logger.log(f"GPU Memory reserved: {th.cuda.memory_reserved() / 1024**3:.2f} GB")
        raise e
    finally:
        # Clean up
        if th.cuda.is_available():
            th.cuda.empty_cache()
        cleanup_distributed()


def main():
    args = create_argparser().parse_args()
    
    # Register cleanup function
    atexit.register(cleanup_distributed)
    
    # Validate required arguments
    if not args.data_dir:
        raise ValueError("--data_dir is required")
    
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Watermark length: {args.wm_length}")
    print(f"Alpha: {args.alpha}")
    print(f"Threshold: {args.threshold}")
    print(f"Resume checkpoint: {args.resume_checkpoint}")
    print(f"Watermark decoder path: {args.wm_decoder_path}")
    print(f"Use FP16: {args.use_fp16}")
    print("="*60)
    
    # Check if CUDA is available
    if not th.cuda.is_available():
        print("CUDA is not available. Training on CPU...")
        world_size = 1
        main_worker(0, world_size, args)
    else:
        # Get number of available GPUs
        world_size = min(th.cuda.device_count(), 2)  # Limit to 2 GPUs max for Kaggle
        print(f"Available GPUs: {world_size}")
        
        if world_size > 1:
            print(f"Starting distributed training on {world_size} GPUs...")
            try:
                # Use spawn method for better compatibility
                mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
            finally:
                cleanup_distributed()
        else:
            print("Starting single GPU training...")
            main_worker(0, 1, args)


def create_argparser():
    defaults = dict(
        data_dir="",
        output_dir="./outputs",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,  # Reduced default batch size
        microbatch=1,   # Set microbatch to 1 for memory efficiency
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=True,  # Enable FP16 by default
        fp16_scale_growth=1e-3,
        wm_length=48,
        alpha=0.4,
        threshold=400,
        wm_decoder_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()
