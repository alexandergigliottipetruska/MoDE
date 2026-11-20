# Add noise, EDM preconditioning and loss
# Diffusion mathematics and training objective
import torch
from math import sqrt
import torch.nn as nn

def add_noise(A, sigma):
    """
    Accepts clean robotic action chunks and noise levels and produces noisy action chunks.
    A: clean action sequences, [B, H, D]
    Sigma: noise levels [B]

    """
    B, H, D = A.shape[0], A.shape[1], A.shape[2]
    # Initialize Gaussian noise [B, H, D]
    noise = torch.empty((B, H, D))
    noise.normal_(mean=0, std=1)

    # Reshape noise levels [B, 1, 1] for broadcasting
    sigma = sigma.reshape(B, 1, 1)

    # Forward corruption equation
    noisy_actions = A + sigma * noise

    # Return noisy actions
    return noisy_actions # shape [B, H, D]

def get_scalings(sigma):
    """
    Return EDM coefficients to scale the input and construct training targets
    """
    B = sigma.shape[0]
    sigma_data = torch.full((B,), fill_value=0.5)

    c_skip = (sigma_data**2) / (sigma ** 2 + sigma_data ** 2)
    c_out = (sigma * sigma_data) / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_in = 1 / (torch.sqrt(sigma ** 2 + sigma_data ** 2))
    return c_skip, c_in, c_out

def edm_loss(model, A, sigma, s, g, l, p):
    """
    A: clean action sequences [B, H, D]
    sigma: noise levels [B]
    s: static images [B, 768]
    g: gripper images [B, 768]
    l: language instructions embeddings [B, 512]
    p: proprioception [B, 15]
    """
    B, H, D = A.shape[0], A.shape[1], A.shape[2]
    # Get EDM coefficients
    c_skip, c_in, c_out = get_scalings(sigma)

    ## Get noisy actions
    noisy_actions = add_noise(A, sigma)

    # Input to Action Transformer
    x_in = c_in.reshape(B, 1, 1) * noisy_actions

    # Construct Training Targets
    y_target = (A - c_skip.reshape(B, 1, 1) * noisy_actions) / c_out.reshape(B, 1, 1)

    # Get model output
    output = model(x_in, sigma, s, g, l, p)

    # Compute Mean Squared Loss (MSE)
    MSE_loss = (output - y_target).square().mean()

    return MSE_loss