#Change for kaggle training
import copy
import functools
import os
import gc

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        ori_model=None,
        wm_length=48,
        alpha=0,
        threshold=0,
        wm_decoder=None,
        wm_decoder_path=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.alpha = alpha
        self.threshold = threshold
        self.wm_decoder = wm_decoder
        self.wm_decoder_path = wm_decoder_path

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist_util.get_world_size()
        self.wm_length = wm_length
        self.sync_cuda = th.cuda.is_available()

        # MEMORY OPTIMIZATION: Clear cache before loading models
        if th.cuda.is_available():
            th.cuda.empty_cache()
            gc.collect()

        if wm_length > 0 and isinstance(wm_length, int):
            self.ori_model = ori_model
            self._load_and_sync_parameters(load_wm_model=True)
        else:
            self.ori_model = None
            self._load_and_sync_parameters()
        
        # Load watermark decoder if provided
        if self.wm_decoder is not None and self.wm_decoder_path and os.path.exists(self.wm_decoder_path):
            if dist_util.get_rank() == 0:
                logger.log(f"loading watermark decoder from: {self.wm_decoder_path}")
                # Load on CPU first, then move to device
                decoder_state = th.load(self.wm_decoder_path, map_location='cpu')
                self.wm_decoder.load_state_dict(decoder_state)
                del decoder_state  # Free memory
                
            # Sync decoder parameters across all processes
            if dist_util.get_world_size() > 1:
                dist_util.sync_params(self.wm_decoder.parameters())
                
            self.wm_decoder.eval()
            # MEMORY OPTIMIZATION: Use half precision for decoder
            if self.use_fp16:
                self.wm_decoder.half()
        
        # MEMORY OPTIMIZATION: Clear cache after model loading
        if th.cuda.is_available():
            th.cuda.empty_cache()
            gc.collect()
        
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        self.opt = AdamW(
            self.mp_trainer.master_params_train, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available() and dist_util.get_world_size() > 1:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=32,  # REDUCED from 128 to 32 for memory
                find_unused_parameters=False,
            )
        else:
            if dist_util.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        

    def _load_and_sync_parameters(self, load_wm_model=False):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist_util.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                # Always load to CPU first to avoid device conflicts
                model_dict = dist_util.load_state_dict(resume_checkpoint, map_location='cpu')
                
                # modify the model dict for watermark model
                if load_wm_model:
                    model_dict_ori = copy.deepcopy(model_dict)
                    # Expand input channels for watermark
                    if 'input_blocks.0.0.weight' in model_dict:
                        original_weight = model_dict['input_blocks.0.0.weight']
                        # Add additional channels for watermark
                        additional_channels = self.model.input_blocks[0][0].weight.shape[1] - original_weight.shape[1]
                        if additional_channels > 0:
                            additional_weight = self.model.input_blocks[0][0].weight[:, -additional_channels:, ...].cpu()
                            model_dict['input_blocks.0.0.weight'] = th.cat((original_weight, additional_weight), 1)
                    
                    # Initialize watermark decoder weights if they exist in the model
                    if hasattr(self.model, 'secret_dense'):
                        model_dict['secret_dense.weight'] = self.model.secret_dense.weight.cpu()
                        model_dict['secret_dense.bias'] = self.model.secret_dense.bias.cpu()

                    # Load with strict=False to handle missing keys
                    missing_keys, unexpected_keys = self.model.load_state_dict(model_dict, strict=False)
                    if missing_keys:
                        logger.log(f"Missing keys in watermark model: {missing_keys}")
                    if unexpected_keys:
                        logger.log(f"Unexpected keys in watermark model: {unexpected_keys}")
                    
                    # Load original model
                    if self.ori_model is not None:
                        missing_keys_ori, unexpected_keys_ori = self.ori_model.load_state_dict(model_dict_ori, strict=False)
                        if missing_keys_ori:
                            logger.log(f"Missing keys in original model: {missing_keys_ori}")
                        del model_dict_ori
                else:
                    missing_keys, unexpected_keys = self.model.load_state_dict(model_dict, strict=False)
                    if missing_keys:
                        logger.log(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.log(f"Unexpected keys: {unexpected_keys}")

                del model_dict
                if dist_util.get_rank() == 0:
                    logger.log('Successfully Loaded.')

        # Ensure models are moved to correct device and have consistent dtype
        self.model.to(dist_util.dev())
        if self.ori_model is not None:
            self.ori_model.to(dist_util.dev())
            # Ensure ori_model uses same precision as main model
            if self.use_fp16:
                # Convert ori_model to FP16 to match main model precision
                self.ori_model.convert_to_fp16()

        # Sync parameters across all processes
        dist_util.sync_params(self.model.parameters())
        if self.ori_model is not None:
            dist_util.sync_params(self.ori_model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist_util.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                # MEMORY OPTIMIZATION: Clear cache periodically
                if th.cuda.is_available() and self.step % (self.log_interval * 10) == 0:
                    th.cuda.empty_cache()
                    gc.collect()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        
        # MEMORY OPTIMIZATION: Process data in smaller chunks
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev()) # -1 ~ 1
            
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                self.ori_model,
                self.alpha,
                self.threshold,
                self.wm_decoder,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            
            # MEMORY OPTIMIZATION: Clear intermediate tensors
            del micro, micro_cond, t, weights, losses, loss
            if th.cuda.is_available():
                th.cuda.empty_cache()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        # Trước khi lưu checkpoint, prune cũ
        logdir = get_blob_logdir()
        prune_old_checkpoints(logdir, keep=2)

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist_util.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                filename = (
                    f"model{self.step + self.resume_step:06d}.pt"
                    if not rate
                    else f"ema_{rate}_{self.step + self.resume_step:06d}.pt"
                )
                path = bf.join(logdir, filename)
                with bf.BlobFile(path, 'wb') as f:
                    th.save(state_dict, f)

        # Lưu model và EMA
        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Lưu optimizer state
        if dist_util.get_rank() == 0:
            opt_path = bf.join(logdir, f"opt{self.step + self.resume_step:06d}.pt")
            with bf.BlobFile(opt_path, 'wb') as f:
                th.save(self.opt.state_dict(), f)

        if dist.is_initialized():
            dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{step:06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    return path if bf.exists(path) else None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def prune_old_checkpoints(logdir: str, keep: int = 2):
    try:
        files = bf.ls(logdir)
    except Exception:
        return
    ckpts = sorted([f for f in files if f.endswith('.pt')])
    if len(ckpts) > keep:
        for old in ckpts[:-keep]:
            try:
                bf.rm(bf.join(logdir, old))
            except Exception:
                pass
