import torch
import torch.nn.functional as F
from torch import nn, optim

# ---------------------------
# Classifier trainer (unchanged)
# ---------------------------
def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(data_loader):.4f}, '
              f'Accuracy: {100*correct/total:.2f}%')
    return model

# ---------------------------
# VAE loss + trainer (unchanged)
# ---------------------------
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae_model(model, data_loader, optimizer, device='cpu', epochs=10):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item() / len(data):.4f}')
        print(f'====> Epoch {epoch+1}/{epochs} - Average loss: {train_loss/len(data_loader.dataset):.4f}')
    return model

# ---------------------------
# DCGAN trainer (robust)
# ---------------------------
@torch.no_grad()
def _smooth_targets(y, eps=0.0):
    return y * (1 - eps) + 0.5 * eps if eps > 0 else y

def train_gan(model, data_loader, device='cpu', epochs=2,
              lr=2e-4, betas=(0.5, 0.999), label_smooth=0.0, log_every=100):
    """
    Train a DCGAN. 'model' is a GANBundle from get_model('GAN').
    Images should be normalized to [-1,1].
    """
    assert hasattr(model, "G") and hasattr(model, "D") and hasattr(model, "z_dim")
    G, D, z_dim = model.G.to(device), model.D.to(device), model.z_dim

    G.train(); D.train()
    optG = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optD = optim.Adam(D.parameters(), lr=lr, betas=betas)
    bce  = nn.BCEWithLogitsLoss()

    history, step = {"d_loss": [], "g_loss": []}, 0

    for epoch in range(1, epochs+1):
        for x, _ in data_loader:
            x = x.to(device)
            B = x.size(0)

            # ---- Train D ----
            z = torch.randn(B, z_dim, 1, 1, device=device)
            fake = G(z).detach()
            y_real = _smooth_targets(torch.ones(B, device=device), eps=label_smooth)
            y_fake = torch.zeros(B, device=device)
            d_loss = bce(D(x), y_real) + bce(D(fake), y_fake)
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()

            # ---- Train G ----
            z = torch.randn(B, z_dim, 1, 1, device=device)
            gen = G(z)
            g_loss = bce(D(gen), torch.ones(B, device=device))
            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()

            history["d_loss"].append(d_loss.item())
            history["g_loss"].append(g_loss.item())
            step += 1
            if step % log_every == 0:
                print(f"[{epoch}/{epochs}] step {step}  D:{d_loss.item():.3f}  G:{g_loss.item():.3f}")

    return model, history

# ---------------------------
# Diffusion trainer
# ---------------------------

# diffusion schedule (DDPM-style)
DIFFUSION_T = 100
DIFFUSION_BETAS = torch.linspace(1e-4, 0.02, DIFFUSION_T)
DIFFUSION_ALPHAS = 1.0 - DIFFUSION_BETAS
DIFFUSION_ALPHA_BARS = torch.cumprod(DIFFUSION_ALPHAS, dim=0)

def _q_sample(x0, t, noise=None):
    """
    Forward process q(x_t | x_0).
    x0 : (B,C,H,W)
    t  : (B,) integer timesteps in [0, T-1]
    """
    if noise is None:
        noise = torch.randn_like(x0)
    a_bar_t = DIFFUSION_ALPHA_BARS[t].view(-1, 1, 1, 1).to(x0.device)
    return torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * noise, noise

def train_diffusion(model, data_loader, criterion, optimizer,
                    device='cpu', epochs=10):
    """
    Train diffusion model to predict noise ε.
    model must take (x_t, t_norm) where t_norm ∈ [0,1].
    criterion: usually nn.MSELoss()
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        n_samples = 0

        for batch_idx, (x0, _) in enumerate(data_loader):
            x0 = x0.to(device)
            B = x0.size(0)
            n_samples += B

            # random timestep per example
            t = torch.randint(0, DIFFUSION_T, (B,), device=device)
            x_t, noise = _q_sample(x0, t)

            t_norm = t.float() / (DIFFUSION_T - 1)
            noise_hat = model(x_t, t_norm)

            loss = criterion(noise_hat, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B

            if batch_idx % 100 == 0:
                print(f"[Diffusion] Epoch {epoch+1}/{epochs}, "
                      f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / n_samples
        print(f"[Diffusion] Epoch {epoch+1}/{epochs} - Avg loss: {epoch_loss:.4f}")

    return model

# ---------------------------
# EBM: Langevin dynamics + trainer
# ---------------------------
import torch.autograd as autograd

def langevin_dynamics(model, x_init, device='cpu',
                      steps=30, step_size=1e-2, noise_std=1e-2):
    """
    Run Langevin dynamics to sample approximately from the EBM.
    x_init: (B, C, H, W) initial noise in [0,1]
    Returns: x_K (detached, no grad) in [0,1]
    """
    x = x_init.to(device)
    x.requires_grad_(True)

    for k in range(steps):
        energy = model(x).sum()               # scalar
        grad_x, = autograd.grad(energy, x, create_graph=False)

        # gradient descent on energy + small Gaussian noise
        x = x - 0.5 * step_size * grad_x + noise_std * torch.randn_like(x)
        x = x.clamp(0.0, 1.0)

        # detach between steps to avoid backprop through the whole chain
        x = x.detach()
        x.requires_grad_(True)

    return x.detach()


def train_ebm(model, data_loader, device='cpu', epochs=10,
              lr=1e-4, steps_langevin=30, step_size=1e-2, noise_std=1e-2):
    """
    Simple contrastive EBM training on CIFAR-10.

    Loss:  L = E_theta(x_pos) - E_theta(x_neg)
    where x_neg is obtained by Langevin dynamics from random noise.
    """
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_idx, (x_pos, _) in enumerate(data_loader):
            x_pos = x_pos.to(device)

            # positive energy on real data
            pos_energy = model(x_pos).mean()

            # negative samples: start from uniform noise in [0,1]
            x_init = torch.rand_like(x_pos)
            x_neg = langevin_dynamics(
                model, x_init, device=device,
                steps=steps_langevin,
                step_size=step_size,
                noise_std=noise_std
            )

            neg_energy = model(x_neg).mean()

            loss = pos_energy - neg_energy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"[EBM] Epoch {epoch}/{epochs}, "
                    f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                    f"E_pos: {pos_energy.item():.3f}, E_neg: {neg_energy.item():.3f}"
                )

        avg_loss = running_loss / len(data_loader)
        print(f"[EBM] Epoch {epoch}/{epochs} - Avg loss: {avg_loss:.4f}")

    return model