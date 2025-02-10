import torch

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        """
        DataLoader would need the length of the subset. 
        Pytorch will call this function automatically when we pass our instance into DataLoader.
        """
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
            return img, label