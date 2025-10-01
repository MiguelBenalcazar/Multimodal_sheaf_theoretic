import torch
import os
import glob
import logging
import json

logger = logging.getLogger(__name__)

def save_checkpoint(P12, 
                    P21, 
                    theta_1,
                    theta_2,
                    batch:int, 
                    monitor_value:float,
                    monitor:str="discrepancy",   # "loss" or "discrepancy"
                    prefix="restriction_maps", 
                    keep_last=2,
                    is_best=False):
    """
    Save a checkpoint containing restriction maps P12, P21 and latent embeddings theta_1, theta_2.

    Args:
        P12, P21: RestrictionMap instances with nn.Linear Pij
        theta_1, theta_2: torch.Tensor, latent embeddings for img/text
        batch: int, epoch or step number
        monitor_value: float, value of monitored metric (e.g., discrepancy)
        monitor: str, metric name
        prefix: str, prefix for filename
        keep_last: int, number of rolling checkpoints to keep
        is_best: bool, whether this is the best checkpoint
    """
    os.makedirs("./saved_restrictions", exist_ok=True)

    filename = f"{prefix}_batch_{batch}_{monitor}_{monitor_value:.4f}.pth"
    path = os.path.join("./saved_restrictions", filename)

    checkpoint = {
        "batch": batch,
        monitor: monitor_value,
        "P12": P12.Pij.state_dict(),
        "P21": P21.Pij.state_dict(),
        "theta_1": theta_1.detach().cpu(),  # save tensors on CPU
        "theta_2": theta_2.detach().cpu(),
    }

    torch.save(checkpoint, path)
    logger.info(f"Saved restriction maps + thetas checkpoint at {path}")

    # Save best checkpoint separately
    if is_best:
        best_path = os.path.join("./saved_restrictions", f"best_{prefix}.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Updated BEST restriction maps checkpoint at {best_path}")

    # Keep only last N rolling checkpoints
    ckpts = sorted(glob.glob(f"./saved_restrictions/{prefix}_batch*.pth"), 
                   key=os.path.getmtime, reverse=True)
    for old_ckpt in ckpts[keep_last:]:
        try:
            os.remove(old_ckpt)
            logger.info(f"Deleted old checkpoint: {old_ckpt}")
        except Exception as e:
            logger.warning(f"Could not delete {old_ckpt}: {e}")


def load_checkpoint(path, P12, P21, device="cpu"):
    """
    Load restriction maps and theta embeddings from checkpoint.

    Args:
        path: str, checkpoint file path
        P12, P21: RestrictionMap instances to restore
        device: str or torch.device, where to load tensors (default="cpu")
    Returns:
        ckpt: dict containing metadata and theta embeddings
    """
    if not os.path.isfile(path):
        logger.error(f"Checkpoint file not found: {path}")
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        ckpt = torch.load(path, map_location=device)
        P12.Pij.load_state_dict(ckpt["P12"])
        P21.Pij.load_state_dict(ckpt["P21"])

        theta_1 = ckpt["theta_1"].to(device)
        theta_2 = ckpt["theta_2"].to(device)

        logger.info(f"Restriction maps + thetas loaded successfully from {path}")
        return ckpt, theta_1, theta_2
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise RuntimeError(f"Error loading checkpoint: {e}")




# def save_checkpoint(P12, 
#                     P21, 
#                     batch:int, 
#                     monitor_value:float,
#                     monitor:str="discrepancy",   # "loss" or "discrepancy"
#                     prefix="restriction_maps", 
#                     keep_last=2,
#                     is_best=False):
#     """
#     Save a checkpoint containing only restriction maps P12 and P21.

#     Args:
#         P12, P21: RestrictionMap instances with nn.Linear Pij
#         batch: int, epoch or step number
#         monitor_value: float, value of monitored metric (e.g., loss)
#         monitor: str, metric name
#         prefix: str, prefix for filename
#         keep_last: int, number of rolling checkpoints to keep
#         is_best: bool, whether this is the best checkpoint
#     """
#     os.makedirs("./saved_restrictions", exist_ok=True)

#     filename = f"{prefix}_batch_{batch}_{monitor}_{monitor_value:.4f}.pth"
#     path = os.path.join("./saved_restrictions", filename)

#     checkpoint = {
#         "batch": batch,
#         monitor: monitor_value,
#         "P12": P12.Pij.state_dict(),
#         "P21": P21.Pij.state_dict(),
#     }

#     torch.save(checkpoint, path)
#     logger.info(f"Saved restriction maps checkpoint at {path}")

#     # Save best checkpoint separately
#     if is_best:
#         best_path = os.path.join("./saved_restrictions", f"best_{prefix}.pth")
#         torch.save(checkpoint, best_path)
#         logger.info(f"Updated BEST restriction maps checkpoint at {best_path}")

#     # Keep only last N rolling checkpoints
#     ckpts = sorted(glob.glob(f"./saved_restrictions/{prefix}_batch*.pth"), 
#                    key=os.path.getmtime, reverse=True)
#     for old_ckpt in ckpts[keep_last:]:
#         try:
#             os.remove(old_ckpt)
#             logger.info(f"Deleted old checkpoint: {old_ckpt}")
#         except Exception as e:
#             logger.warning(f"Could not delete {old_ckpt}: {e}")


# def load_checkpoint(path, P12, P21):
#     """
#     Load restriction maps from checkpoint.

#     Args:
#         path: str, checkpoint file path
#         P12, P21: RestrictionMap instances to restore
#     """
#     if not os.path.isfile(path):
#         logger.error(f"Checkpoint file not found: {path}")
#         raise FileNotFoundError(f"Checkpoint file not found: {path}")

#     try:
#         ckpt = torch.load(path, map_location="cpu")
#         P12.Pij.load_state_dict(ckpt["P12"])
#         P21.Pij.load_state_dict(ckpt["P21"])
#         logger.info(f"Restriction maps loaded successfully from {path}")
#         return ckpt  # return metadata (batch, loss, etc.)
#     except Exception as e:
#         logger.error(f"Error loading checkpoint: {e}")
#         raise RuntimeError(f"Error loading checkpoint: {e}")
