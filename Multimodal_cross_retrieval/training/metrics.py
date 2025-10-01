import torch
import numpy as np

def evaluation(P12, P21, theta1, theta2):
    # theta_proj_discrepancy = P12.Pij(theta1) -  P21.Pij(theta2)
    theta_proj_discrepancy = P12.Pij(theta1) - P21.Pij(theta2)
    
     # compute scalar for backward (sheaf regularization)
    sheaf_loss = torch.mean(theta_proj_discrepancy ** 2)
    discrepancy_norm = torch.norm(theta_proj_discrepancy, p=2) / theta_proj_discrepancy.numel()
    cos_sim = torch.nn.functional.cosine_similarity(P12.Pij(theta1), P21.Pij(theta2), dim=-1).mean()

    if np.isnan(sheaf_loss.item()):
        raise ValueError(f"NaN detected in sheaf_loss: {sheaf_loss}")

    return theta_proj_discrepancy, sheaf_loss.item(), discrepancy_norm.item(), cos_sim.item()