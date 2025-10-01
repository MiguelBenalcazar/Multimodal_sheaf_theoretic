import torch
import torch.distributed as dist
from training import metrics


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
    total_var_theta1 = torch.tensor(0.0, device=device)
    total_var_theta2 = torch.tensor(0.0, device=device)
    

    for i, (imgs, encodings) in enumerate(test_loader):
        imgs = imgs.to(device, non_blocking=True)
        encodings = {k: v.to(device, non_blocking=True) for k, v in encodings.items()}

        embeddings_imgs = img_model(**imgs).last_hidden_state[:, 0, :]
        embeddings_text = text_model(**encodings).last_hidden_state[:, 0, :]

        _, sheaf_loss, discrepancy_norm, cos_sim, var_theta1,var_theta2 = metrics.evaluation(P12, P21, embeddings_imgs, embeddings_text)

        loss_sheaf_reg = sheaf_loss - args.alpha * (var_theta1 + var_theta2)

    

        
        # loss_sheaf_reg = torch.tensor(loss_sheaf_reg, device=device)
        # discrepancy_norm = torch.tensor(discrepancy_norm, device=device)
        # cos_sim = torch.tensor(cos_sim, device=device)

        total_sheaf += loss_sheaf_reg.item()
        total_discrepancy_norm += discrepancy_norm.item()
        total_cosine_similarity += cos_sim.item()
        total_var_theta1 += var_theta1.item()
        total_var_theta2 += var_theta2.item()

        # if args.test_run:
        #     break

    # Reduce across GPUs if distributed
    if args.distributed:
        total_sheaf= reduce_tensor(total_sheaf)
        total_discrepancy_norm=reduce_tensor(total_discrepancy_norm)
        total_cosine_similarity = reduce_tensor(total_cosine_similarity)
        total_var_theta1 = reduce_tensor(total_var_theta1)
        total_var_theta2 = reduce_tensor(total_var_theta2)


    # Calculate average loss
    num_batches = len(test_loader)
    avg_train_total_sheaf = total_sheaf / num_batches
    avg_train_discrepancy = total_discrepancy_norm / num_batches
    avg_train_cosine_sim = total_cosine_similarity / num_batches
    avg_var_theta1 = total_var_theta1 / num_batches
    avg_var_theta2 = total_var_theta2 / num_batches


    return (avg_train_total_sheaf.item(), 
        avg_train_discrepancy.item(), 
        avg_train_cosine_sim.item(),
        avg_var_theta1.item(),
        avg_var_theta2.item())
    # return avg_train_total_sheaf.item(), avg_train_discrepancy.item(), avg_train_cosine_sim.item()


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


