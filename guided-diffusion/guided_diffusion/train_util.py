#Change for kaggle training - Optimized checkpoint management
import copy
import functools
import os
import gc
import re
from pathlib import Path
import shutil

class FileHandler:
    """Replacement for blobfile using standard Python libraries"""
    
    @staticmethod
    def ls(path):
        """List files in directory"""
        try:
            if os.path.isdir(path):
                return os.listdir(path)
            else:
                return []
        except (OSError, FileNotFoundError):
            return []

    @staticmethod
    def rm(path):
        """Remove file or directory"""
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
        except (OSError, FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not remove {path}: {e}")

    @staticmethod
    def stat(path):
        """Get file statistics"""
        class StatResult:
            def __init__(self, size):
                self.size = size
        
        try:
            return StatResult(os.path.getsize(path))
        except (OSError, FileNotFoundError):
            return StatResult(0)

    @staticmethod
    def exists(path):
        """Check if path exists"""
        return os.path.exists(path)

    @staticmethod
    def join(*args):
        """Join path components"""
        return os.path.join(*args)

    @staticmethod
    def dirname(path):
        """Get directory name"""
        return os.path.dirname(path)

    @staticmethod
    def copy(src, dst):
        """Copy file from src to dst"""
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        except (OSError, FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not copy {src} to {dst}: {e}")
            raise

    @staticmethod
    def move(src, dst):
        """Move file from src to dst"""
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        except (OSError, FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not move {src} to {dst}: {e}")
            raise

    class BlobFile:
        """File context manager replacement"""
        def __init__(self, path, mode='rb'):
            self.path = path
            self.mode = mode
            self.file = None
            
        def __enter__(self):
            # Ensure directory exists when writing
            if 'w' in self.mode or 'a' in self.mode:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.file = open(self.path, self.mode)
            return self.file
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.file:
                self.file.close()

# Create global instance
bf = FileHandler()

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
        
        # Kiểm tra disk space mỗi nửa save_interval
        if self.step > 0 and self.step % (self.save_interval // 2) == 0:
            check_disk_space(get_blob_logdir(), min_free_gb=2.0)

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
        """
        Improved save method with better checkpoint management and error handling.
        """
        logdir = get_blob_logdir()
        
        # Kiểm tra disk space trước khi save
        check_disk_space(logdir, min_free_gb=2.6)
        
        # Cleanup checkpoint cũ TRƯỚC khi lưu checkpoint mới
        prune_old_checkpoints(logdir, keep=2)  # Giữ 2 checkpoint mới nhất
        
        def save_checkpoint_safely(rate, params):
            """Save checkpoint với error handling và atomic operations."""
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            
            if dist_util.get_rank() == 0:
                filename = (
                    f"model{self.step + self.resume_step:06d}.pt"
                    if not rate
                    else f"ema_{rate}_{self.step + self.resume_step:06d}.pt"
                )
                
                logger.log(f"💾 Saving {filename}...")
                
                final_path = bf.join(logdir, filename)
                temp_path = final_path + ".tmp"
                
                try:
                    # Ghi vào file tạm thời
                    with bf.BlobFile(temp_path, 'wb') as f:
                        th.save(state_dict, f)
                    
                    # Kiểm tra file tạm có được ghi đầy đủ không
                    if bf.exists(temp_path):
                        temp_size = bf.stat(temp_path).size
                        if temp_size > 0:
                            # Xóa file cũ nếu tồn tại
                            if bf.exists(final_path):
                                bf.rm(final_path)
                            
                            # Rename atomic - use move instead of copy+delete
                            bf.move(temp_path, final_path)
                            
                            # Verify file sau khi lưu
                            final_size = bf.stat(final_path).size
                            logger.log(f"✅ Successfully saved {filename} ({final_size / (1024*1024):.1f} MB)")
                        else:
                            raise ValueError(f"Temporary file {temp_path} is empty")
                    else:
                        raise ValueError(f"Temporary file {temp_path} was not created")
                        
                except Exception as e:
                    logger.log(f"❌ Failed to save {filename}: {e}")
                    
                    # Cleanup temp file nếu có
                    if bf.exists(temp_path):
                        try:
                            bf.rm(temp_path)
                        except:
                            pass
                    
                    raise e  # Re-raise để caller biết có lỗi

        # Lưu model và EMA checkpoints
        try:
            save_checkpoint_safely(0, self.mp_trainer.master_params)
            
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint_safely(rate, params)
                
        except Exception as e:
            logger.log(f"Error saving model checkpoints: {e}")
            # Có thể continue để ít nhất lưu được optimizer state

        # Lưu optimizer state
        if dist_util.get_rank() == 0:
            opt_filename = f"opt{self.step + self.resume_step:06d}.pt"
            opt_final_path = bf.join(logdir, opt_filename)
            opt_temp_path = opt_final_path + ".tmp"
            
            logger.log(f"💾 Saving {opt_filename}...")
            
            try:
                with bf.BlobFile(opt_temp_path, 'wb') as f:
                    th.save(self.opt.state_dict(), f)
                
                if bf.exists(opt_temp_path) and bf.stat(opt_temp_path).size > 0:
                    if bf.exists(opt_final_path):
                        bf.rm(opt_final_path)
                    bf.move(opt_temp_path, opt_final_path)
                    
                    opt_size = bf.stat(opt_final_path).size
                    logger.log(f"✅ Successfully saved {opt_filename} ({opt_size / (1024*1024):.1f} MB)")
                else:
                    raise ValueError("Optimizer temp file is empty or not created")
                    
            except Exception as e:
                logger.log(f"❌ Failed to save optimizer state: {e}")
                if bf.exists(opt_temp_path):
                    try:
                        bf.rm(opt_temp_path)
                    except:
                        pass

        # Sync tất cả processes
        if dist.is_initialized():
            dist.barrier()
        
        # Memory cleanup sau khi save
        if th.cuda.is_available():
            th.cuda.empty_cache()
        
        logger.log(f"📁 Checkpoint save completed at step {self.step + self.resume_step}")


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


def prune_old_checkpoints(logdir: str, keep: int = 3):
    """
    Giữ lại tối đa `keep` checkpoint mới nhất cho mỗi loại file, xóa các file cũ hơn.
    Phân loại chính xác: model, EMA (theo rate), optimizer
    """
    if dist_util.get_rank() != 0:
        return  # Chỉ rank 0 thực hiện cleanup
    
    try:
        # Kiểm tra thư mục có tồn tại không
        if not bf.exists(logdir):
            logger.log(f"Warning: Logdir {logdir} does not exist")
            return
            
        # Lấy danh sách file
        try:
            files = bf.ls(logdir)
        except Exception as e:
            logger.log(f"Warning: Could not list files in {logdir}: {e}")
            return
            
        # Phân loại các file checkpoint theo loại
        model_checkpoints = []      # [(step, filename)]
        ema_checkpoints = {}        # {rate: [(step, filename)]}
        optimizer_checkpoints = []  # [(step, filename)]
        other_files = []           # Các file khác
        
        logger.log(f"Found {len(files)} files in {logdir}")
        
        for filename in files:
            if not filename.endswith('.pt'):
                continue
                
            # Model checkpoints: modelXXXXXX.pt
            model_match = re.match(r'^model(\d{6})\.pt$', filename)
            if model_match:
                step = int(model_match.group(1))
                model_checkpoints.append((step, filename))
                continue
            
            # EMA checkpoints: ema_RATE_XXXXXX.pt
            ema_match = re.match(r'^ema_([0-9.]+)_(\d{6})\.pt$', filename)
            if ema_match:
                rate = ema_match.group(1)
                step = int(ema_match.group(2))
                if rate not in ema_checkpoints:
                    ema_checkpoints[rate] = []
                ema_checkpoints[rate].append((step, filename))
                continue
            
            # Optimizer checkpoints: optXXXXXX.pt
            opt_match = re.match(r'^opt(\d{6})\.pt$', filename)
            if opt_match:
                step = int(opt_match.group(1))
                optimizer_checkpoints.append((step, filename))
                continue
            
            # Các file .pt khác
            other_files.append(filename)
        
        # Thống kê trước khi cleanup
        logger.log(f"Before cleanup:")
        logger.log(f"  Model checkpoints: {len(model_checkpoints)}")
        logger.log(f"  EMA checkpoints: {sum(len(v) for v in ema_checkpoints.values())}")
        logger.log(f"  Optimizer checkpoints: {len(optimizer_checkpoints)}")
        logger.log(f"  Other .pt files: {len(other_files)}")
        
        total_deleted = 0
        deleted_size_mb = 0
        
        # Xóa model checkpoints cũ
        if len(model_checkpoints) > keep:
            model_checkpoints.sort(key=lambda x: x[0])  # Sort by step
            files_to_delete = model_checkpoints[:-keep]
            
            for step, filename in files_to_delete:
                try:
                    file_path = bf.join(logdir, filename)
                    
                    # Tính size file trước khi xóa
                    try:
                        file_size = bf.stat(file_path).size
                        deleted_size_mb += file_size / (1024 * 1024)
                    except:
                        pass
                    
                    bf.rm(file_path)
                    logger.log(f"✓ Deleted old model checkpoint: {filename} (step {step})")
                    total_deleted += 1
                except Exception as e:
                    logger.log(f"✗ Failed to delete {filename}: {e}")
        
        # Xóa EMA checkpoints cũ cho mỗi rate
        for rate, checkpoints in ema_checkpoints.items():
            if len(checkpoints) > keep:
                checkpoints.sort(key=lambda x: x[0])  # Sort by step
                files_to_delete = checkpoints[:-keep]
                
                for step, filename in files_to_delete:
                    try:
                        file_path = bf.join(logdir, filename)
                        
                        # Tính size file trước khi xóa
                        try:
                            file_size = bf.stat(file_path).size
                            deleted_size_mb += file_size / (1024 * 1024)
                        except:
                            pass
                        
                        bf.rm(file_path)
                        logger.log(f"✓ Deleted old EMA checkpoint: {filename} (rate {rate}, step {step})")
                        total_deleted += 1
                    except Exception as e:
                        logger.log(f"✗ Failed to delete {filename}: {e}")
        
        # Xóa optimizer checkpoints cũ
        if len(optimizer_checkpoints) > keep:
            optimizer_checkpoints.sort(key=lambda x: x[0])  # Sort by step
            files_to_delete = optimizer_checkpoints[:-keep]
            
            for step, filename in files_to_delete:
                try:
                    file_path = bf.join(logdir, filename)
                    
                    # Tính size file trước khi xóa
                    try:
                        file_size = bf.stat(file_path).size
                        deleted_size_mb += file_size / (1024 * 1024)
                    except:
                        pass
                    
                    bf.rm(file_path)
                    logger.log(f"✓ Deleted old optimizer checkpoint: {filename} (step {step})")
                    total_deleted += 1
                except Exception as e:
                    logger.log(f"✗ Failed to delete {filename}: {e}")
        
        # Thống kê sau cleanup
        remaining_model = max(0, len(model_checkpoints) - max(0, len(model_checkpoints) - keep))
        remaining_ema = sum(max(0, len(v) - max(0, len(v) - keep)) for v in ema_checkpoints.values())
        remaining_opt = max(0, len(optimizer_checkpoints) - max(0, len(optimizer_checkpoints) - keep))
        
        logger.log(f"After cleanup:")
        logger.log(f"  Remaining model checkpoints: {remaining_model}")
        logger.log(f"  Remaining EMA checkpoints: {remaining_ema}")
        logger.log(f"  Remaining optimizer checkpoints: {remaining_opt}")
        logger.log(f"  Total files deleted: {total_deleted}")
        logger.log(f"  Disk space freed: {deleted_size_mb:.1f} MB")
        
        if total_deleted > 0:
            logger.log(f"🧹 Checkpoint cleanup completed successfully!")
        else:
            logger.log("ℹ️  No old checkpoints to clean up")
            
    except Exception as e:
        logger.log(f"Error in prune_old_checkpoints: {e}")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}")


def check_disk_space(logdir: str, min_free_gb: float = 2.0):
    """
    Kiểm tra dung lượng disk còn trống và thực hiện cleanup tích cực nếu cần.
    """
    if dist_util.get_rank() != 0:
        return
        
    try:
        import shutil
        total, used, free = shutil.disk_usage(logdir)
        free_gb = free / (1024**3)
        used_gb = used / (1024**3)
        total_gb = total / (1024**3)
        
        logger.log(f"💾 Disk space - Free: {free_gb:.2f} GB / Total: {total_gb:.2f} GB ({free_gb/total_gb*100:.1f}% free)")
        
        if free_gb < min_free_gb:
            logger.log(f"⚠️  WARNING: Low disk space! Free: {free_gb:.2f} GB < {min_free_gb:.2f} GB threshold")
            logger.log("🧹 Performing aggressive cleanup - keeping only 1 most recent checkpoint of each type")
            
            # Cleanup tích cực - chỉ giữ 1 checkpoint gần nhất
            prune_old_checkpoints(logdir, keep=1)
            
            # Kiểm tra lại sau cleanup
            total, used, free = shutil.disk_usage(logdir)
            free_gb_after = free / (1024**3)
            freed_space = free_gb_after - free_gb
            
            if freed_space > 0:
                logger.log(f"✅ Freed {freed_space:.2f} GB. New free space: {free_gb_after:.2f} GB")
            
            if free_gb_after < min_free_gb * 0.5:  # Nếu vẫn còn rất ít
                logger.log(f"🚨 CRITICAL: Disk space still very low after cleanup!")
        else:
            logger.log("✅ Disk space is sufficient")
            
    except Exception as e:
        logger.log(f"Could not check disk space: {e}")
