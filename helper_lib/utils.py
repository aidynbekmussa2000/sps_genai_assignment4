import torch

def save_model(model, path='model.pth'):
    """
    Save model state dictionary to file
    
    Args:
        model: PyTorch model
        path: File path to save model
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path='model.pth'):
    """
    Load model state dictionary from file
    
    Args:
        model: PyTorch model (architecture should match saved model)
        path: File path to load model from
    
    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')
    return model