import torch
import torch.nn as nn
import torch.nn.functional as F


class RestrictionMap:  # Inherit from nn.Module for DDP compatibility
    def __init__(self, 
                 di_dj: int, 
                 dij: int = 64, 
                 requires_grad: bool = False):  # Default to True for training
    
        super(RestrictionMap, self).__init__()
        
        self.dim_i_j = di_dj  # dimension di or dj
        self.dim = dij  # common dimensionality
        self.Pij = torch.nn.Linear(self.dim_i_j, self.dim, bias=False)
        self.Pij.weight.requires_grad = requires_grad

        self.__init_parameters__()
    
    def __init_parameters__(self):
        torch.nn.init.uniform_(self.Pij.weight, -0.1, 0.1)
        # torch.nn.init.orthogonal_(self.Pij.weight)

    def forward(self, x):
        """Forward pass for nn.Module compatibility"""
        return self.Pij(x)
    
    def get_size(self):
        """Return the shape of the restriction map weight matrix"""
        return self.Pij.weight.shape
        
    @torch.no_grad()  # disable gradient tracking during manual update
    def step(self, Pji, theta1, theta2, eta: float = 0.01, lambda_val: float = 0.01):
        """Manual step function (if you still want to use it alongside optimizer)"""
        # Compute discrepancy
        proj1 = self.Pij(theta1)   # P12 θ1
        proj2 = Pji(theta2)        # P21 θ2

        discrepancy_ij = proj1 - proj2
      
         # Update rule
         # Gradient descent step on the weights of Pij
         #
        self.Pij.weight -= eta * lambda_val * discrepancy_ij.T @ theta1

        # proj1 = self.Pij.weight @ theta1  # P12 θ1
        # proj2 = Pji.weight @ theta2        # P21 θ2

        # discrepancy_ij = proj1 - proj2
        # self.Pij.weight -= eta * lambda_val * discrepancy_ij @ theta1.T
