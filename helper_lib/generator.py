import torch
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# VAE utilities
# ---------------------------

def generate_samples(model, device, num_samples=10, latent_dim=20):
    """VAE: sample z ~ N(0, I) and decode."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        samples = model.decode(z).cpu()

    n = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=(n*2, n*2))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n*n):
        axes[i].axis('off')
        if i < num_samples:
            axes[i].imshow(samples[i].squeeze().numpy(), cmap='gray')
    plt.tight_layout(); plt.show()

def reconstruct_images(model, data_loader, device, num_images=10):
    """Reconstruct a few images with a trained VAE."""
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    with torch.no_grad():
        recon_images, _, _ = model(images)
    images = images.cpu(); recon_images = recon_images.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axes[0, i].imshow(images[i].squeeze().numpy(), cmap='gray');  axes[0, i].axis('off')
        axes[1, i].imshow(recon_images[i].squeeze().numpy(), cmap='gray'); axes[1, i].axis('off')
    axes[0,0].set_ylabel('Original'); axes[1,0].set_ylabel('Reconstructed')
    plt.tight_layout(); plt.show()

def interpolate_latent_space(model, data_loader, device, steps=10):
    """Linear interpolation between two images in latent space."""
    model.eval()
    images, _ = next(iter(data_loader))
    img1, img2 = images[0:1].to(device), images[1:2].to(device)
    with torch.no_grad():
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        grids = []
        for a in np.linspace(0, 1, steps):
            z = (1 - a) * mu1 + a * mu2
            grids.append(model.decode(z).cpu())
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
    for i in range(steps):
        axes[i].imshow(grids[i].squeeze().numpy(), cmap='gray')
        axes[i].axis('off'); axes[i].set_title(f'{i/(steps-1):.1f}')
    plt.suptitle('Latent Space Interpolation'); plt.tight_layout(); plt.show()

def visualize_latent_space_2d(model, data_loader, device, num_batches=10):
    """2D latent space visualization (use with latent_dim=2)."""
    model.eval()
    z_list, label_list = [], []
    with torch.no_grad():
        for i, (data, labels) in enumerate(data_loader):
            if i >= num_batches: break
            data = data.to(device)
            mu, _ = model.encode(data)
            z_list.append(mu.cpu().numpy())
            label_list.append(labels.numpy())
    z = np.concatenate(z_list, 0)
    labels = np.concatenate(label_list, 0)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(z[:,0], z[:,1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(sc, label='Class'); plt.xlabel('z1'); plt.ylabel('z2')
    plt.title('2D Latent Space'); plt.grid(True, alpha=0.3); plt.show()

# ---------------------------
# GAN utility
# ---------------------------

def generate_gan_samples(model, device='cpu', num_samples=16, save_path=None):
    """
    Generate a grid of samples from a trained DCGAN (model is GANBundle).
    Assumes the generator outputs in [-1, 1].
    """
    assert hasattr(model, "G") and hasattr(model, "z_dim")
    G = model.G.to(device).eval()
    z_dim = model.z_dim

    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, 1, 1, device=device)
        imgs = G(z).cpu()                # [-1, 1]
        imgs = (imgs.clamp(-1, 1) + 1)/2 # -> [0, 1]

    n = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=(n*2, n*2))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n*n):
        axes[i].axis('off')
        if i < num_samples:
            if imgs.shape[1] == 1:
                axes[i].imshow(imgs[i, 0].numpy(), cmap='gray')
            else:
                axes[i].imshow(np.transpose(imgs[i].numpy(), (1,2,0)))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"Saved samples to {save_path}")
    plt.show()
    

# ---------------------------
# EBM sampling utility
# ---------------------------

def _langevin_sample(model, x_init, device='cpu',
                     steps=60, step_size=1e-2, noise_std=1e-2):
    """
    Internal helper: Langevin dynamics for sampling at test time.
    """
    x = x_init.to(device)
    x.requires_grad_(True)

    for k in range(steps):
        energy = model(x).sum()
        grad_x, = torch.autograd.grad(energy, x, create_graph=False)
        x = x - 0.5 * step_size * grad_x + noise_std * torch.randn_like(x)
        x = x.clamp(0.0, 1.0)
        x = x.detach()
        x.requires_grad_(True)

    return x.detach()


def generate_ebm_samples(model, device='cpu', num_samples=16,
                         steps=60, step_size=1e-2, noise_std=1e-2,
                         image_size=32, channels=3):
    """
    Generate samples from a trained EBM on CIFAR-10 using Langevin dynamics.
    """
    model = model.to(device)
    model.eval()

    # start from uniform noise in [0,1]
    x_init = torch.rand(num_samples, channels, image_size, image_size, device=device)
    x_samples = _langevin_sample(
        model, x_init, device=device,
        steps=steps, step_size=step_size, noise_std=noise_std
    ).cpu()

    n = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=(n*2, n*2))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n*n):
        axes[i].axis('off')
        if i < num_samples:
            img = x_samples[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)

    plt.tight_layout()
    plt.show()


# ---------------------------
# Diffusion utilities
# ---------------------------

# keep schedule consistent with trainer
# ---------------------------
# Diffusion utilities
# ---------------------------

DIFFUSION_T = 100
DIFFUSION_BETAS = torch.linspace(1e-4, 0.02, DIFFUSION_T)
DIFFUSION_ALPHAS = 1.0 - DIFFUSION_BETAS
DIFFUSION_ALPHA_BARS = torch.cumprod(DIFFUSION_ALPHAS, dim=0)

def _p_sample_step(model, x_t, t, device):
    B = x_t.size(0)
    t_batch = torch.full((B,), t, device=device, dtype=torch.long)
    t_norm = t_batch.float() / (DIFFUSION_T - 1)

    eps_hat = model(x_t, t_norm)

    beta_t = DIFFUSION_BETAS[t].to(device)
    alpha_t = DIFFUSION_ALPHAS[t].to(device)
    alpha_bar_t = DIFFUSION_ALPHA_BARS[t].to(device)

    beta_t = beta_t.view(1, 1, 1, 1)
    alpha_t = alpha_t.view(1, 1, 1, 1)
    alpha_bar_t = alpha_bar_t.view(1, 1, 1, 1)

    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat
    )

    if t > 0:
        z = torch.randn_like(x_t)
        sigma_t = torch.sqrt(beta_t)
        x_prev = mean + sigma_t * z
    else:
        x_prev = mean

    return x_prev


def generate_diffusion_samples(model, device, num_samples=10, diffusion_steps=100):
    """
    Generate CIFAR-10-like samples with a trained diffusion model.
    Assumes 3-channel 32x32 images.
    """
    channels = 3
    image_size = 32

    model = model.to(device)
    model.eval()

    # *** IMPORTANT: start from 3×32×32 noise, not 1×28×28 ***
    x_t = torch.randn(num_samples, channels, image_size, image_size, device=device)
    print("DEBUG: x_t initial shape:", x_t.shape)

    start_t = min(diffusion_steps, DIFFUSION_T) - 1
    end_t = 0

    with torch.no_grad():
        for t in range(start_t, end_t - 1, -1):
            x_t = _p_sample_step(model, x_t, t, device)

    samples = x_t.clamp(0, 1).cpu()

    n = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n * n):
        axes[i].axis("off")
        if i < num_samples:
            img = samples[i].permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C)
            axes[i].imshow(img)

    plt.tight_layout()
    plt.show()