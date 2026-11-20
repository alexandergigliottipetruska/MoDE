from models.transformer import ActionTransformer
from diffusion.edm import edm_loss
import torch
from math import pi
import torch.nn as nn
from diffusion.edm import add_noise, get_scalings
from diffusion.sampler import reverse_diffusion_sampling
import matplotlib.pyplot as plt

# Define the model and optimizer
model = ActionTransformer(hidden_dim=128, num_blocks=2, num_heads=4, causal=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
B = 16
optim_steps = 0

# Create synthetic action demonstrations
t = torch.arange(0, 10, 1).reshape((1, 10, 1))
i = torch.arange(0, B, 1).reshape((B, 1, 1))
a = torch.arange(0, 6, 1).reshape((1, 1, 6)) # action dims

A_continuous = torch.sin((2 * pi * t / 10) + (i * pi / 8) + (a * pi) / 6) # continuous arm action trajectories
G_discrete = torch.where(i % 2 == 0, 1, -1).expand(-1, 10, -1) # discrete gripper trajectories

A = torch.cat([A_continuous, G_discrete], dim=2)
B = A.shape[0]

# Sample a new batch of noise levels using log-uniform distribution.
sigma_min = torch.tensor(0.01).expand(B)
sigma_max = torch.tensor(2.0).expand(B)

while optim_steps <= 2000:
    # Zero gradients
    optimizer.zero_grad()

    mu = torch.empty((B))
    mu.uniform_(0, 1)
    noise_level = torch.exp((1 - mu) * torch.log(sigma_min) + mu * torch.log(sigma_max))

    # Calculate loss
    loss = edm_loss(model, A, noise_level)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()
    if optim_steps == 0:
        print(f"Loss {loss:.2f} at optim step {optim_steps}")
    optim_steps += 1

    if optim_steps % 10 == 0:
        print(f"Loss {loss:.2f} at optim step {optim_steps}")

model.eval()

with torch.no_grad():
    for sigma in [0.1, 0.5, 1.0, 2.0]:

        # One noise level for every action sequence
        noise_level = torch.full((A.shape[0],), sigma)

        # Corrupt clean actions
        noisy_action = add_noise(A, noise_level)

        # EDM coefficients
        c_skip, c_in, c_out = get_scalings(noise_level)

        c_skip = c_skip.reshape(-1, 1, 1)
        c_out = c_out.reshape(-1, 1, 1)
        c_in = c_in.reshape(-1, 1, 1)

        # Reconstruct clean actions
        reconstructed_action = (
            c_skip * noisy_action
            + c_out * model(c_in * noisy_action, noise_level)
        )

        # Compare reconstruction errors
        noisy_mse = (noisy_action - A).square().mean()
        skip_mse = (c_skip * noisy_action - A).square().mean()
        denoised_mse = (reconstructed_action - A).square().mean()

        print(f"Sigma: {sigma}")
        print(f"Noisy MSE: {noisy_mse.item():.4f}")
        print(f"Skip-only MSE: {skip_mse.item():.4f}")
        print(f"Denoised MSE: {denoised_mse.item():.4f}")


import math

@torch.no_grad()
def sample_edm(model, x0, num_steps, sigma_min=0.01, sigma_max=2.0):
    model.eval()

    x = x0.clone()
    B = x.shape[0]

    # Exponentially decreasing noise schedule, ending at zero
    sigmas = torch.logspace(
        math.log10(sigma_max),
        math.log10(sigma_min),
        steps=num_steps,
        device=x.device,
        dtype=x.dtype,
    )
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

    for k in range(num_steps):
        sigma = sigmas[k]
        sigma_next = sigmas[k + 1]

        noise_level = torch.full(
            (B,), sigma.item(), device=x.device, dtype=x.dtype
        )

        c_skip, c_in, c_out = get_scalings(noise_level)

        c_skip = c_skip.reshape(B, 1, 1)
        c_out = c_out.reshape(B, 1, 1)
        c_in = c_in.reshape(B, 1, 1)

        # Full EDM denoising prediction
        denoised = (
            c_skip * x
            + c_out * model(c_in * x, noise_level)
        )

        # Euler derivative
        d = (x - denoised) / sigma

        # Euler update
        x = x + (sigma_next - sigma) * d

    return x


# ----------------------------------
# Compare 10, 20, and 40 Euler steps
# ----------------------------------

torch.manual_seed(42)

device = next(model.parameters()).device
B_eval = 64

# SAME starting noise for every experiment
epsilon = torch.randn(B_eval, 10, 7, device=device)
x0 = 2.0 * epsilon

step_counts = [10, 20, 40]
results = {}

for steps in step_counts:
    output = sample_edm(model, x0, num_steps=steps)
    results[steps] = output.detach().cpu()

    assert output.shape == (64, 10, 7)
    assert torch.isfinite(output).all()

    print(f"{steps} steps: sampling complete")


# ----------------------------------
# Plot results: 3 rows x 6 columns
# ----------------------------------

fig, axes = plt.subplots(3, 6, figsize=(21, 10))

time = range(10)
num_trajectories = 4

for row, steps in enumerate(step_counts):
    output = results[steps]

    for dim in range(6):
        ax = axes[row, dim]

        for traj in range(num_trajectories):
            ax.plot(
                time,
                output[traj, :, dim].numpy(),
                label=f"Trajectory {traj}"
            )

        ax.set_title(f"{steps} steps | Action dim {dim}")
        ax.set_xlabel("Timestep")

        if dim == 0:
            ax.set_ylabel("Action value")

        ax.grid(True, linestyle=":", alpha=0.6)

        if row == 0 and dim == 0:
            ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("sampling_comparison.png", dpi=200, bbox_inches="tight")
plt.show()