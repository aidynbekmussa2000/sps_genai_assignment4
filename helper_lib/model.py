import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# ---------------- DCGAN modules (size-robust D) ----------------
def _block(in_c, out_c, k, s, p, *, transposed=False, bn=True):
    op = nn.ConvTranspose2d if transposed else nn.Conv2d
    layers = [op(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(out_c))
    return nn.Sequential(*layers)

class DCGANGenerator(nn.Module):
    """
    z ~ N(0,1), (B, z_dim, 1, 1) -> (C, 28, 28) in [-1, 1]
    """
    def __init__(self, z_dim=100, g_feat=64, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            _block(z_dim,     g_feat*4, 3, 2, 0, transposed=True), nn.ReLU(True),  # 1x1 -> 3x3
            _block(g_feat*4,  g_feat*2, 4, 2, 1, transposed=True), nn.ReLU(True),  # 3x3 -> 6x6
            _block(g_feat*2,  g_feat,   4, 2, 1, transposed=True), nn.ReLU(True),  # 6x6 -> 12x12
            nn.ConvTranspose2d(g_feat,  out_channels, 4, 2, 3, bias=False),        # -> 28x28
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class DCGANDiscriminator(nn.Module):
    """
    (C, H, W) in [-1,1] -> real/fake logits (robust to H,W via AdaptiveAvgPool2d)
    Designed for H=W≈28 but tolerant to small deviations.
    """
    def __init__(self, d_feat=64, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, d_feat,   4, 2, 1, bn=False), nn.LeakyReLU(0.2, True),
            _block(d_feat,      d_feat*2, 4, 2, 1),           nn.LeakyReLU(0.2, True),
            _block(d_feat*2,    d_feat*4, 3, 1, 1),           nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1)  # -> (B, d_feat*4, 1, 1)
        )
        self.head = nn.Conv2d(d_feat*4, 1, kernel_size=1, bias=False)

    def forward(self, x):
        h = self.features(x)
        return self.head(h).view(-1)  # logits

class GANBundle(nn.Module):
    """Wrapper so the trainer/sampler can pass around one object."""
    def __init__(self, G, D, z_dim):
        super().__init__()
        self.G = G
        self.D = D
        self.z_dim = z_dim

# ---------------- Diffusion model ----------------
class DiffusionModel(nn.Module):
    """
    Simple epsilon-prediction network for 1-channel 28x28 images.
    Input: x_t (B,1,28,28), t_norm (B,) or (B,1) in [0,1]
    Output: predicted noise ε̂ with same shape as x_t.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.is_diffusion = True  # easy flag for helper code

        self.down1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.down3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        # t: (B,) or (B,1)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        t_emb = self.time_mlp(t)                 # (B, C)
        t_emb = t_emb.view(t.shape[0], -1, 1, 1)

        d1 = F.silu(self.down1(x))
        d2 = F.silu(self.down2(F.avg_pool2d(d1, 2)))
        d3 = F.silu(self.down3(F.avg_pool2d(d2, 2)))

        d3 = d3 + t_emb                          # inject time info

        u1 = F.silu(self.up1(d3))
        u2 = F.silu(self.up2(u1))

        out = self.out(u2)
        return out


class EnergyModel(nn.Module):
    """
    Simple CNN energy function for EBM on CIFAR-10.

    Input:  x in R^{B x C x H x W}, e.g. 3x32x32
    Output: scalar energy per sample, shape (B,)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),   # -> (B, base_channels*4, 1, 1)
        )
        self.fc = nn.Linear(base_channels * 4, 1)

    def forward(self, x):
        h = self.features(x)          # (B, C, 1, 1)
        h = h.view(h.size(0), -1)     # (B, C)
        energy = self.fc(h)           # (B, 1)
        return energy.squeeze(-1)     # (B,)



# ---------------- your existing models ----------------
def get_model(model_name, latent_dim=20, input_channels=1):
    """
    Return the appropriate model based on model_name
    model_name ∈ {'FCNN','CNN','EnhancedCNN','VAE','GAN','Diffusion'}
    """
    if model_name == 'FCNN':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    elif model_name == 'CNN':
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = F.relu(self.fc1(x))
                x = self.dropout2(x)
                return self.fc2(x)
        model = CNN()

    elif model_name == 'EnhancedCNN':
        # If you see a deprecation warning, switch to ResNet18_Weights.DEFAULT
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)

    elif model_name == 'VAE':
        class VAE(nn.Module):
            def __init__(self, latent_dim=20, input_channels=1):
                super().__init__()
                self.latent_dim = latent_dim
                self.encoder = nn.Sequential(
                    nn.Conv2d(input_channels, 32, 3, 2, 1), nn.ReLU(),  # 28->14
                    nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),              # 14->7
                    nn.Flatten()
                )
                self.fc_mu     = nn.Linear(64*7*7, latent_dim)
                self.fc_logvar = nn.Linear(64*7*7, latent_dim)
                self.decoder_input = nn.Linear(latent_dim, 64*7*7)
                self.decoder = nn.Sequential(
                    nn.Unflatten(1, (64,7,7)),
                    nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),  # 7->14
                    nn.ConvTranspose2d(32, input_channels, 3, 2, 1, output_padding=1),
                    nn.Sigmoid()
                )

            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = self.decoder_input(z)
                return self.decoder(h)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar
        model = VAE(latent_dim=latent_dim, input_channels=input_channels)

    elif model_name == 'GAN':
        z_dim = 100
        G = DCGANGenerator(z_dim=z_dim, g_feat=64, out_channels=input_channels)
        D = DCGANDiscriminator(d_feat=64, in_channels=input_channels)
        model = GANBundle(G, D, z_dim)

    elif model_name == 'Diffusion':
        model = DiffusionModel(in_channels=input_channels)
    
    elif model_name == 'EBM':
        model = EnergyModel(in_channels=input_channels)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model