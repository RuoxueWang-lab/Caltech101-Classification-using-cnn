import torch

def train_with_validation(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()
        
        # Print training loss for the epoch
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)  # Get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Print validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

   
    