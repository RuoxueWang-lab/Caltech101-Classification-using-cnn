from network import CNN
from train import train_with_validation
from test import test
from torchvision import models
import data_loader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

def main(hyperparameters_grid, model):
    train_set, test_loader = data_loader.load_data()
    
    best_accuracy = 0
    
    for lr in hyperparameters_grid['learning_rate']:
        for batch_size in hyperparameters_grid['batch_size']:
            for epochs in hyperparameters_grid['epochs']:
                print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}:")

                # Perform k-fold cross-validation
                fold_accuracies = []
                for fold, (train_loader, val_loader) in enumerate(data_loader.CrossValidation(train_set, batch_size)):
                    print(f"\nStarting Fold {fold + 1}")
                    criterion = nn.CrossEntropyLoss() 
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    fold_accruacy = train_with_validation(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)
                    fold_accuracies.append(fold_accruacy)
    
                avg_accuracy = np.mean(fold_accuracies)
                print(f"Average Accuracy: {avg_accuracy}%")
                
                if avg_accuracy < best_accuracy:
                    best_accuracy = avg_accuracy
                    best_hyperparameters = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }
    print(f"Best parameters: {best_hyperparameters}")
    print(f"Best average accuracy: {best_accuracy}%")
        
    # Test the model
    # test(model, test_loader)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Self-designed CNN
    # model = CNN()
    
    # ResNet
    model = models.resnet50(pretrained=True)
    
    # There's only one fully connected layer at the end of ResNet netowrk
    # We extract the number of input features of the last fully connected layer
    num_features = model.fc.in_features
    # Then we change the output features of this layer into 102, which matches the number of classes Caltech101 dataset
    model.fc = nn.Linear(num_features, 102)
    
    model = model.to(device)
    
    hyperparameters_grid = {
    'learning_rate': [0.001, 0.01],
    'batch_size': [32, 64],
    'epochs': [5, 10]}
    
    main(hyperparameters_grid, model)
             
    
    