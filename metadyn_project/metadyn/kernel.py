# metadyn/kernel.py
import torch
import torch.nn as nn

class MetadynamicKernel(nn.Module):
    """
    A learned optimizer for iterative inference, implemented as a PyTorch Module.

    This module takes the current state of a variable `z` and the gradient
    of an objective function with respect to `z`, and computes an optimal update
    step. It is designed to replace fixed, hand-coded update rules (like
    vanilla gradient descent) in models that use iterative refinement.

    By learning the update rule, the kernel can adapt to the specific problem
    landscape, learning strategies like momentum, filtering, and second-order-like
    behavior to achieve faster and more robust convergence, especially under

    noisy or ill-conditioned scenarios.

    Args:
        state_dim (int): The dimensionality of the state variable `z` to be optimized.
        grad_dim (int): The dimensionality of the gradient of the objective.
        hidden_dim (int, optional): The size of the hidden layer in the kernel network.
                                   Defaults to 128.
    """
    def __init__(self, state_dim: int, grad_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.grad_dim = grad_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim + grad_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Computes the optimal state update.

        Args:
            state (torch.Tensor): The current state variable (e.g., `z`).
            grad (torch.Tensor): The gradient of the objective w.r.t. the state.

        Returns:
            torch.Tensor: The computed state update (`delta_z`).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if grad.dim() == 1:
            grad = grad.unsqueeze(0)
            
        x = torch.cat([state, grad], dim=1)
        return self.net(x)