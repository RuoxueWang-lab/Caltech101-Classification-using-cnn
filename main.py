from network import CNN
from train import train_with_validation
from test import test
import data_loader
import torch.optim as optim
import torch.nn as nn
import numpy as np

def main(hyperparameters_grid):
    model = CNN()
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
    hyperparameters_grid = {
    'learning_rate': [0.001, 0.01],
    'batch_size': [32, 64],
    'epochs': [5, 10]}
    
    main(hyperparameters_grid)
             
    
    