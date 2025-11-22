import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained PyTorch model
        data_loader: Test data loader
        criterion: Loss function
        device: 'cpu' or 'cuda'
    
    Returns:
        avg_loss: Average loss on test set
        accuracy: Accuracy percentage
    """
    model = model.to(device)
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = test_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    print(f'Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy