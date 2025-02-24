import torch
import numpy as np
from diffusers import CogVideoXDDIMScheduler

def get_sigma_sid_edm(batch_size=100, tmax=800, device='cpu'):
    """Implementation from SiD paper"""
    sigma_min = 0.002
    sigma_max = 80
    rho = 7.0
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)

    rnd_t = torch.linspace(0, 1, batch_size, device=device).view(-1, 1, 1, 1) * tmax/1000
    sigma = (max_inv_rho + (1-rnd_t) * (min_inv_rho - max_inv_rho)) ** rho

    return sigma.squeeze(), rnd_t.squeeze()

def get_sigma_cogvideo(timesteps, scheduler):
    """Implementation from CogVideo distillation"""
    alphas_cumprod = scheduler.alphas_cumprod[timesteps]
    sigma = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    sigma = (sigma ** 2 + 0.5 ** 2) / (sigma * 0.5) ** 2
    return sigma



def main():
    # Setup
    batch_size = 100

    # Get sigmas from both implementations
    sigma_edm, t_edm = get_sigma_sid_edm(batch_size)

    scheduler = CogVideoXDDIMScheduler.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="scheduler")
    timesteps = torch.linspace(0, len(scheduler.alphas_cumprod)-1, batch_size, dtype=torch.long)
    sigma_cogvideo = get_sigma_cogvideo(timesteps, scheduler)

    # Convert to numpy
    sigma_edm = sigma_edm.numpy()
    sigma_cogvideo = sigma_cogvideo.numpy()
    t_edm = t_edm.numpy()

    # Print numerical comparisons
    print("\nNumerical Comparison (at 10 time points):")
    print("Time      SiD EDM      CogVideo     Ratio")
    print("-" * 50)

    for i in range(0, batch_size, batch_size//10):
        ratio = sigma_edm[i] / sigma_cogvideo[i]
        print(f"{t_edm[i]:6.3f}   {sigma_edm[i]:10.6f}   {sigma_cogvideo[i]:10.6f}   {ratio:8.6f}")

    # Print statistics
    print("\nStatistics:")
    ratio = sigma_edm / sigma_cogvideo
    print(f"Correlation coefficient: {np.corrcoef(sigma_edm, sigma_cogvideo)[0,1]:.6f}")
    print(f"Mean ratio: {ratio.mean():.6f}")
    print(f"Std ratio: {ratio.std():.6f}")
    print(f"Min ratio: {ratio.min():.6f}")
    print(f"Max ratio: {ratio.max():.6f}")

    # Print range information
    print("\nValue Ranges:")
    print(f"SiD EDM range:    [{sigma_edm.min():.6f}, {sigma_edm.max():.6f}]")
    print(f"CogVideo range:   [{sigma_cogvideo.min():.6f}, {sigma_cogvideo.max():.6f}]")

if __name__ == "__main__":
    main()
