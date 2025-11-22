import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True,
                    dataset_type='imagefolder', for_vae=False, for_gan=False):
    """
    Create and return a data loader for training or testing.

    Args:
        data_dir: Path to data directory or built-in dataset root.
        batch_size: Batch size.
        train: Whether to load train or test split.
        dataset_type: 'mnist', 'cifar10', 'fashionmnist', or 'imagefolder'.
        for_vae: If True, use transforms suitable for VAE (range [0,1]).
        for_gan: If True, use transforms suitable for GAN (range [-1,1]).
    """
    # Choose normalization depending on purpose
    if for_gan:
        # GAN → scale images to [-1,1]
        def _norm(channels):
            return transforms.Normalize((0.5,) * channels, (0.5,) * channels)
    elif for_vae:
        # VAE → no normalization (keep [0,1])
        _norm = lambda c: None
    else:
        # Default classification normalization
        def _norm(channels):
            if channels == 1:
                return transforms.Normalize((0.1307,), (0.3081,))
            else:
                return transforms.Normalize((0.5,) * channels, (0.5,) * channels)

    # --- Dataset selection ---
    if dataset_type == 'mnist':
        channels = 1
        tfms = [transforms.ToTensor()]
        norm = _norm(channels)
        if norm: tfms.append(norm)
        dataset = datasets.MNIST(root='./data', train=train, download=True,
                                 transform=transforms.Compose(tfms))

    elif dataset_type == 'fashionmnist':
        channels = 1
        tfms = [transforms.ToTensor()]
        norm = _norm(channels)
        if norm: tfms.append(norm)
        dataset = datasets.FashionMNIST(root='./data', train=train, download=True,
                                        transform=transforms.Compose(tfms))

    elif dataset_type == 'cifar10':
        channels = 3
        tfms = [transforms.ToTensor()]
        norm = _norm(channels)
        if norm: tfms.append(norm)
        dataset = datasets.CIFAR10(root='./data', train=train, download=True,
                                   transform=transforms.Compose(tfms))

    else:  # custom image folder
        channels = 3
        if for_gan or for_vae:
            size = (28, 28)
        else:
            size = (224, 224)
        tfms = [transforms.Resize(size), transforms.ToTensor()]
        norm = _norm(channels)
        if norm: tfms.append(norm)
        dataset = datasets.ImageFolder(root=data_dir, transform=transforms.Compose(tfms))

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )
    return loader