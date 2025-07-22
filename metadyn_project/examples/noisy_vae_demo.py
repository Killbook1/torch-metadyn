# examples/noisy_vae_demo.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# --- IMPORTANT: We now import our MetadynamicKernel from the local library structure ---
import sys
# Add the parent directory to the path to find the 'metadyn' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metadyn import MetadynamicKernel

# --- Configuration ---
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 2e-4
    LATENT_DIM = 32
    INFERENCE_STEPS_T = 8
    LEAK = 0.5
    NOISE_LEVEL = 1.5

cfg = Config()

# --- Re-usable VAE Components ---
# These are identical to our experiment, they are part of the "host" model
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim * 2)
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.upsample = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
                                      nn.Linear(256, 64 * 7 * 7), nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )
    def forward(self, z): return self.net(self.upsample(z).view(-1, 64, 7, 7))

# --- The Host Model Demonstrating the Library ---
class DynamicVAE(nn.Module):
    def __init__(self, latent_dim, dynamic_type='learned'):
        super().__init__()
        self.latent_dim = latent_dim
        self.dynamic_type = dynamic_type
        self.encoder = Encoder(latent_dim) 
        self.decoder = Decoder(latent_dim)
        
        if self.dynamic_type == 'learned':
            # This is where we instantiate our library's component
            self.f_psi = MetadynamicKernel(state_dim=latent_dim, grad_dim=latent_dim)

    def forward(self, x, T):
        mu_target, log_var_target = torch.chunk(self.encoder(x), 2, dim=1)
        z = torch.randn(x.size(0), self.latent_dim, device=x.device)
        history = []
        is_grad_enabled = torch.is_grad_enabled()

        for t in range(T):
            z_step = z.clone().requires_grad_(True)
            objective = -0.5 * torch.sum(1 + log_var_target - (z_step - mu_target).pow(2) - log_var_target.exp(), dim=1)
            history.append(objective.mean().item())

            if is_grad_enabled:
                grad_objective = torch.autograd.grad(objective.sum(), z_step, retain_graph=True)[0]
                if self.training and cfg.NOISE_LEVEL > 0:
                    grad_objective += torch.randn_like(grad_objective) * cfg.NOISE_LEVEL
            else:
                grad_objective = torch.zeros_like(z)

            if self.dynamic_type == 'learned':
                delta_z = self.f_psi(z.detach(), grad_objective)
            else:
                delta_z = -grad_objective
            
            z = (1 - cfg.LEAK) * z + cfg.LEAK * delta_z

        recon_x = self.decoder(z)
        final_kl_div = -0.5 * torch.sum(1 + log_var_target - (z - mu_target).pow(2) - log_var_target.exp())
        return recon_x, final_kl_div, history

# --- Main Execution Block ---
def main():
    print(f"Executing Demo with config: {cfg}")
    output_dir = "demo_results"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    models_to_train = {
        "VAE_FixedDynamic_Noisy": DynamicVAE(cfg.LATENT_DIM, dynamic_type='fixed'),
        "VAE_LearnedDynamic_Noisy": DynamicVAE(cfg.LATENT_DIM, dynamic_type='learned')
    }
    histories = {}

    for name, model in models_to_train.items():
        print(f"\n--- Training Demo Model: {name} ---")
        model.to(cfg.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
        recon_loss_fn = nn.BCELoss(reduction='sum')
        
        for epoch in range(cfg.EPOCHS):
            model.train()
            progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
            for i, (inputs, _) in enumerate(progress_bar):
                inputs = inputs.to(cfg.DEVICE)
                optimizer.zero_grad(set_to_none=True)
                recon_x, kl_div, history = model(inputs, T=cfg.INFERENCE_STEPS_T)
                if i == 0 and epoch == 0:
                    histories[name] = history
                loss = (recon_loss_fn(recon_x, inputs) + kl_div) / inputs.size(0)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(Loss=f"{loss.item():.2f}")

    # --- Plotting and Saving Results ---
    print("\n--- Generating Comparison Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range(cfg.INFERENCE_STEPS_T), histories["VAE_FixedDynamic_Noisy"], 'o-', label='Fixed Dynamic (Noisy Gradient)', color='crimson', alpha=0.7)
    ax.plot(range(cfg.INFERENCE_STEPS_T), histories["VAE_LearnedDynamic_Noisy"], 'o-', label='Metadynamic Kernel', color='royalblue', linewidth=2.5, markersize=8)
    ax.set_title('Inference Trajectory Under Noisy Conditions', fontsize=18, fontweight='bold')
    ax.set_xlabel('Inference Timestep (T)', fontsize=14)
    ax.set_ylabel('Objective Value (KL Divergence)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "inference_comparison_demo.png")
    plt.savefig(plot_path)
    print(f"Saved comparison plot to {plot_path}")
    plt.show()

if __name__ == '__main__':
    main()