# Torch-MetaDyn: Learned Inference Dynamics

[![PyPI version](https://badge.fury.io/py/torch-metadyn.svg)](https://badge.fury.io/py/torch-metadyn) <!-- Placeholder -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`torch-metadyn`** provides a PyTorch implementation of a **Metadynamic Kernel** (`f_ψ`), a small neural network that learns to perform optimization for iterative inference tasks.

Instead of relying on fixed update rules like `z_new = z - lr * grad`, this library allows you to learn the update rule itself, leading to significantly faster and more robust convergence, especially in challenging, noisy environments.

## The Core Idea

Many models involve an iterative process to infer some latent variable `z`. Standard methods use a fixed gradient-based approach. This is often slow and highly sensitive to noise.

A Metadynamic Kernel replaces the fixed rule with a learned function: `delta_z = f_psi(z, grad)`. This kernel learns to be a sophisticated, problem-specific optimizer, discovering strategies like momentum and noise filtering implicitly.

### The Proof: Robustness Under Uncertainty

The value of this approach is most clear in a noisy environment. We trained two models to infer a latent variable `z` for a VAE on MNIST, but we injected significant noise into the guiding gradient signal.

*   **System A (Fixed Gradient):** Fails to converge, getting stuck in a poor local minimum.
*   **System B (Our Metadynamic Kernel):** Learns to filter the noise and robustly finds a superior solution in fewer steps.

![Inference Comparison](https://raw.githubusercontent.com/user/torch-metadyn/main/assets/inference_comparison_noisy.png) <!-- Placeholder -->

This demonstrates that our learned kernel is not just a complex alternative, but a fundamentally more powerful mechanism for inference.

## Installation

```bash
pip install torch-metadyn # Placeholder for actual PyPI name
```

## Quick Start

Replace your fixed update rule with our learned kernel.

```python
import torch
from metadyn import MetadynamicKernel

# Setup
state_dim = 32
latent_z = torch.randn(1, state_dim, requires_grad=True)
objective = compute_some_loss(latent_z)
grad = torch.autograd.grad(objective, latent_z)[0]

# --- Old Way (Fixed Dynamic) ---
# lr = 0.1
# delta_z_fixed = -lr * grad

# --- New Way (Learned Dynamic) ---
# Kernel is a torch.nn.Module that you include in your model's parameters
f_psi = MetadynamicKernel(state_dim=32, grad_dim=32)
delta_z_learned = f_psi(latent_z.detach(), grad)

# The kernel f_psi is trained jointly with the rest of your model.
# See the /examples directory for a complete, working demo.
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{archon2024metadyn,
  author       = {Archon},
  title        = {Torch-MetaDyn: Learned Inference Dynamics},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/user/torch-metadyn}}
}
```

## License

This project is licensed under the MIT License.