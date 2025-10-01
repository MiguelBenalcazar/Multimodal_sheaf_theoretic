import torch
import numpy as np

def evaluation(P12, P21, theta1, theta2):
    # theta_proj_discrepancy = P12.Pij(theta1) -  P21.Pij(theta2)
    theta1_proj = P12.Pij(theta1)
    theta2_proj = P21.Pij(theta2)
    # theta_proj_discrepancy = P12.Pij(theta1) - P21.Pij(theta2)
    theta_proj_discrepancy =theta1_proj - theta2_proj
    
     # compute scalar for backward (sheaf regularization)
    sheaf_loss = torch.mean(theta_proj_discrepancy ** 2)
    discrepancy_norm = torch.norm(theta_proj_discrepancy, p=2) / theta_proj_discrepancy.numel()
    cos_sim = torch.nn.functional.cosine_similarity(P12.Pij(theta1), P21.Pij(theta2), dim=-1).mean()


    var_theta1 = torch.var(theta1_proj, dim=0).mean()
    var_theta2 = torch.var(theta2_proj, dim=0).mean()

    if np.isnan(sheaf_loss.item()):
        raise ValueError(f"NaN detected in sheaf_loss: {sheaf_loss}")

    return theta_proj_discrepancy, sheaf_loss, discrepancy_norm, cos_sim, var_theta1, var_theta2