import torch
import torch.distributed as dist
from training import metrics
import torch.nn.functional as F

def recall_metrics(all_img_embs, all_txt_embs, all_img_ids, all_cap_ids, Ks=[1,5,10]):
    sims = all_img_embs @ all_txt_embs.T
    recalls = {}
    
    for K in Ks:
        # Image -> Text
        correct_i2t = 0
        for i, img_id in enumerate(all_img_ids):
            topk = sims[i].topk(K).indices.cpu().numpy()
            if any(all_cap_ids[j] == img_id for j in topk):
                correct_i2t += 1
        R_i2t = correct_i2t / len(all_img_ids)
        
        # Text -> Image
        correct_t2i = 0
        for j, cap_id in enumerate(all_cap_ids):
            topk = sims[:, j].topk(K).indices.cpu().numpy()
            if any(all_img_ids[i] == cap_id for i in topk):
                correct_t2i += 1
        R_t2i = correct_t2i / len(all_cap_ids)
        
        recalls[f'R@{K}_i2t'] = R_i2t
        recalls[f'R@{K}_t2i'] = R_t2i
    return recalls

def gather_tensor(tensor, args=None):
    """
    Gather a tensor from all processes in DDP.
    If not distributed, return the tensor itself.
    """
    if args is None or not getattr(args, "distributed", False):
        return tensor

    if not dist.is_available() or not dist.is_initialized():
        return tensor
    

    world_size = args.world_size if hasattr(args, "world_size") else dist.get_world_size()

    # Make sure tensor is on GPU
    if not tensor.is_cuda:
        tensor = tensor.to(args.device)
    
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
   
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

# def gather_list(lst, args):
#     """
#     Gather a list (of ints) across processes in DDP.
#     """
#     tensor = torch.tensor(lst, device=args.device, dtype=torch.long)
#     gathered = gather_tensor(tensor, args=args)
#     return gathered.cpu().tolist()


def recall_metrics_distributed(args, all_img_embs, all_txt_embs, all_img_ids, all_cap_ids, Ks=[1, 5, 10]):
    """
    Handles gathering across processes and computes recall metrics.
    """
    # Concatenate local batches first
    all_img_embs = torch.cat(all_img_embs, dim=0)
    all_txt_embs = torch.cat(all_txt_embs, dim=0)
    all_img_ids = torch.tensor(all_img_ids, device=args.device, dtype=torch.long)
    all_cap_ids = torch.tensor(all_cap_ids, device=args.device, dtype=torch.long)

    if args.distributed:
        all_img_embs = gather_tensor(all_img_embs, args=args)
        all_txt_embs = gather_tensor(all_txt_embs, args=args)
        all_img_ids = gather_tensor(all_img_ids, args=args).cpu().tolist()
        all_cap_ids = gather_tensor(all_cap_ids, args=args).cpu().tolist()
    else:
        all_img_ids = all_img_ids.cpu().tolist()
        all_cap_ids = all_cap_ids.cpu().tolist()

    # Only compute recalls on main process
    if not args.distributed or (dist.get_rank() == 0):
        return recall_metrics(all_img_embs, all_txt_embs, all_img_ids, all_cap_ids, Ks)
    else:
        return None



def evaluate_model(
        img_model, 
        text_model, 
        test_loader,
        P12, 
        P21,
        device,
        args):

    img_model.eval()
    text_model.eval()
   

    
    total_sheaf = torch.tensor(0.0, device=device)
    total_discrepancy_norm =torch.tensor(0.0, device=device)
    total_cosine_similarity = torch.tensor(0.0, device=device)

    for i, (imgs, encodings) in enumerate(test_loader):
        imgs = imgs.to(device, non_blocking=True)
        encodings = {k: v.to(device, non_blocking=True) for k, v in encodings.items()}

        embeddings_imgs = img_model(**imgs).last_hidden_state[:, 0, :]
        embeddings_text = text_model(**encodings).last_hidden_state[:, 0, :]

        theta_proj_discrepancy, sheaf_loss, discrepancy_norm, cos_sim = metrics.evaluation(P12, P21, embeddings_imgs, embeddings_text)
        
        sheaf_loss = torch.tensor(sheaf_loss, device=device)
        discrepancy_norm = torch.tensor(discrepancy_norm, device=device)
        cos_sim = torch.tensor(cos_sim, device=device)

        total_sheaf += sheaf_loss
        total_discrepancy_norm += discrepancy_norm
        total_cosine_similarity += cos_sim

        # if args.test_run:
        #     break

    # Reduce across GPUs if distributed
    if args.distributed:
        total_sheaf= reduce_tensor(total_sheaf)
        total_discrepancy_norm=reduce_tensor(total_discrepancy_norm)
        total_cosine_similarity = reduce_tensor(total_cosine_similarity)


    # Calculate average loss
    num_batches = len(test_loader)
    avg_train_total_sheaf = total_sheaf / num_batches
    avg_train_discrepancy = total_discrepancy_norm / num_batches
    avg_train_cosine_sim = total_cosine_similarity / num_batches
    return avg_train_total_sheaf.item(), avg_train_discrepancy.item(), avg_train_cosine_sim.item()


def reduce_tensor(tensor, average=True):
    """Reduce tensor across all processes (sum or mean)."""
    rt = tensor.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        if average:
            rt /= dist.get_world_size()
    return rt





# def evaluate_model(
#         img_model, 
#         text_model, 
#         test_loader,
#         P12, 
#         P21,
#         epoch, 
#         device,
#         args):

#     img_model.eval()
#     text_model.eval()
   

#     total_sheaf = torch.tensor(0.0, device=device)
#     total_elements = torch.tensor(0, device=device)

#     if distributed_mode.is_main_process():
#             logger.info(f"-------TESTING Epoch: {epoch} -------")


    
#     total_sheaf = 0.0
#     total_discrepancy_norm = 0.0
#     total_cosine_similarity = 0.0
#     for i, (imgs, encodings) in enumerate(test_loader):
#         imgs = imgs.to(device, non_blocking=True)
#         encodings = {k: v.to(device, non_blocking=True) for k, v in encodings.items()}

#         embeddings_imgs = img_model(**imgs).last_hidden_state[:, 0, :]
#         embeddings_text = text_model(**encodings).last_hidden_state[:, 0, :]

#         theta_proj_discrepancy, sheaf_loss, discrepancy_norm, cos_sim = metrics.evaluation(P12, P21, theta1, theta2)
        

#         # Forward pass through restriction maps
#         proj1 = P12.Pij(embeddings_imgs)  # Use forward() method
#         proj2 = P21.Pij(embeddings_text)
            
#         theta_proj_discrepancy = proj1 - proj2

#         # Sum over all elements for correct weighting across GPUs
#         batch_loss = torch.sum(theta_proj_discrepancy ** 2)
#         batch_elements = torch.tensor(theta_proj_discrepancy.numel(), device=device)
            
#         total_sheaf += batch_loss
#         total_elements += batch_elements

#         if args.test_run:
#             break

#     # Reduce across GPUs if distributed
#     if args.distributed:
#         dist.all_reduce(total_sheaf, op=dist.ReduceOp.SUM)
#         dist.all_reduce(total_elements, op=dist.ReduceOp.SUM)

#     # Calculate average loss
#     avg_loss = (total_sheaf / total_elements).item() if total_elements > 0 else 0.0
#     return avg_loss


