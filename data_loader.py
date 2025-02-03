from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold

import Transform

# Define data augmentation transformations for the training set
transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
    ])

# Define the minimal transformations for validation set and test set
transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
    ])   


def load_data():
    # Download dataset without any transform
    dataset = datasets.Caltech101(root="./data", download=True, transform=None)
    
    # Slpit the dataset
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    
    test_dataset = Transform.TransformSubset(test_set, transform_test)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)
    
    return train_set, test_loader
    
def CrossValidation(train_set, batch_size):
    # Split training set for Cross Validation
    # Get the length of the train_set
    length = len(train_set)
    # Initialize a 5-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(length))):
        train_subset = Subset(train_set, train_idx)
        val_set = Subset(train_set, val_idx)
        
        train_dataset = Transform.TransformSubset(train_subset, transform_train)
        val_dataset = Transform.TransformSubset(val_set, transform_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        yield train_loader, val_loader
        
        

