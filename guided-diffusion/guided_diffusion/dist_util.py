"""
Helpers for distributed training on Kaggle without MPI.
"""
import io
import os
import socket
import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2  # T4 x2 on Kaggle
SETUP_RETRY_COUNT = 3

def setup_dist():
    """
    Setup a distributed process group for Kaggle environment.
    """
    if dist.is_initialized():
        return
    
    # Get environment variables or set defaults
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set CUDA device
    if th.cuda.is_available():
        th.cuda.set_device(rank % th.cuda.device_count())
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % th.cuda.device_count())
    
    # Setup distributed training if multiple GPUs
    if world_size > 1:
        backend = "nccl" if th.cuda.is_available() else "gloo"
        
        # Set master address and port
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(_find_free_port()))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        # Single GPU setup - initialize with gloo backend for compatibility
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        
        backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{th.cuda.current_device()}")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without MPI dependency.
    Always load to CPU first to avoid device conflicts in distributed training.
    """
    # Override map_location to CPU if not specified to avoid device conflicts
    if 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu'
        
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        for p in params:
            with th.no_grad():
                dist.broadcast(p, 0)

def _find_free_port():
    """Find a free port for distributed training."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return port
    finally:
        s.close()

def get_rank():
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """Get the world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
