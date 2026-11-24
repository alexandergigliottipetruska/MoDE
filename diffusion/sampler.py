# Reverse diffusion sampling
# Generative Inference
import torch
import torch.nn as nn
from diffusion.edm import get_scalings

def reverse_diffusion_sampling(model, B, s, g, l, H=10, D=7, noise_min=0.001, noise_max=80.0, sampling_steps=10, sigmas=None):
    """
    Initial sample x0: [B, H, D]
    s: static images [B, 768]
    g: gripper images [B, 768]
    l: language instruction embeddings [B, 512]
    p: proprioception: [B, 15]
    """
    sigma_max = torch.tensor(noise_max)
    sigma_min = torch.tensor(noise_min)
    # Sample Gaussian Noise
    noise = torch.empty((B, H, D))
    noise.normal_(mean=0.0, std=1.0)

    # Create initial sample x0
    x_0 = sigma_max * noise

    # Create a Exponential noise scheduler for sigma_k
    k = torch.arange(0, sampling_steps, 1)
    N = sampling_steps
    if sigmas is None:
        sigmas = sigma_max * (sigma_min / sigma_max) ** (k / (N - 1))

    x_k = x_0

    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)], dim=-1)

    # First-order Euler Integration
    for k in range(len(sigmas) - 1):
        sigma_k = sigmas[k].expand(B)
        
        c_skip, c_in, c_out = get_scalings(sigma_k)
        c_skip = c_skip.reshape(B, 1, 1)
        c_in = c_in.reshape(B, 1, 1)
        c_out = c_out.reshape(B, 1, 1)
        F_theta =  model(c_in * x_k, sigma_k, s, g, l)

        # Reconstruct clean action estimate using EDM coefficients
        A_hat = c_skip * x_k + c_out * F_theta # reconstructed action

        # Convert clean action estimate into ODE Derivative
        d_k = (x_k - A_hat) / sigmas[k]

        # Perform Euler Update
        x_next = x_k + (sigmas[k+1] - sigmas[k]) * d_k
        x_k = x_next

    output = x_k

    return output